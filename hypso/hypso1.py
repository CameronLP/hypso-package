from datetime import datetime
from dateutil import parser
from importlib.resources import files
#from osgeo import gdal, osr
from pathlib import Path
from typing import Literal, Union

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pyproj as prj
#import rasterio
#import re
import xarray as xr

from .hypso import Hypso
from hypso.atmospheric import run_py6s, run_acolite, run_machi
from hypso.calibration import read_coeffs_from_file, \
                              run_radiometric_calibration, \
                              run_destriping_correction, \
                              run_smile_correction, \
                              make_mask, \
                              make_overexposed_mask, \
                              get_destriping_correction_matrix, \
                              run_destriping_correction_with_computed_matrix
from hypso.chlorophyll import run_tuned_chlorophyll_estimation, run_band_ratio_chlorophyll_estimation, validate_tuned_model
from hypso.geometry import interpolate_at_frame, \
                           geometry_computation, \
                           get_nearest_pixel
from hypso.georeference import georeferencing
from hypso.georeference.utils import check_star_tracker_orientation
from hypso.masks import run_global_land_mask, \
                        run_ndwi_land_mask, \
                        run_threshold_land_mask

from hypso.masks import run_cloud_mask, \
                        run_quantile_threshold_cloud_mask

from hypso.reading import load_l1a_nc_cube, load_l1a_nc_metadata

from hypso.writing import l1a_nc_writer, l1b_nc_writer

from hypso.DataArrayValidator import DataArrayValidator
from hypso.DataArrayDict import DataArrayDict

from satpy import Scene
from satpy.dataset.dataid import WavelengthRange
from pyresample.geometry import SwathDefinition

SUPPORTED_PRODUCT_LEVELS = ["l1a", "l1b", "l2a"]

ATM_CORR_PRODUCTS = Literal["6sv1", "acolite", "machi"]
CHL_EST_PRODUCTS = Literal["band_ratio", "6sv1_aqua", "acolite_aqua"]
LAND_MASK_PRODUCTS = Literal["global", "ndwi", "threshold"]
CLOUD_MASK_PRODUCTS = Literal["default"]

DEFAULT_ATM_CORR_PRODUCT = "6sv1"
DEFAULT_CHL_EST_PRODUCT = "band_ratio"
DEFAULT_LAND_MASK_PRODUCT = "global"
DEFAULT_CLOUD_MASK_PRODUCT = "default"

UNIX_TIME_OFFSET = 20 # TODO: Verify offset validity. Sivert had 20 here

# TODO: store latitude and longitude as xarray
# TODO: setattr, hasattr, getattr for setting class variables, especially those from geometry

class Hypso1(Hypso):

    def __init__(self, hypso_path: Union[str, Path], points_path: Union[str, Path, None] = None, verbose=False) -> None:
        
        """
        Initialization of HYPSO-1 Class.

        :param hypso_path: Absolute path to "L1a.nc" file
        :param points_path: Absolute path to the corresponding ".points" files generated with QGIS for manual geo
            referencing. (Optional. Default=None)

        """

        super().__init__(hypso_path=hypso_path, points_path=points_path)

        # General -----------------------------------------------------
        self.platform = 'hypso1'
        self.sensor = 'hypso1_hsi'
        self.VERBOSE = verbose

        # Booleans to check if certain processes have been run
        self.georeferencing_has_run = False
        self.calibration_has_run = False
        self.geometry_computation_has_run = False
        self.write_l1a_nc_file_has_run = False
        self.write_l1b_nc_file_has_run = False
        self.write_l2a_nc_file_has_run = False
        #self.atmospheric_correction_has_run = False
        self.land_mask_has_run = False
        self.cloud_mask_has_run = False
        self.chlorophyll_estimation_has_run = False
        self.toa_reflectance_has_run = False

        self._load_l1a_nc_file()
        self._run_georeferencing()

        chl_attributes = {}
        product_attributes = {}

        self.chl = DataArrayDict(dims_shape=self.spatial_dimensions, 
                                 attributes=chl_attributes, 
                                 dims_names=self.dim_names_2d,
                                 num_dims=2
                                 )
        
        self.products = DataArrayDict(dims_shape=self.spatial_dimensions, 
                                      attributes=product_attributes, 
                                      dims_names=self.dim_names_2d,
                                      num_dims=2
                                      )

        return None

        
    # Setters

    def _set_capture_datetime(self) -> None:
        """
        Format and set the datetime of the capture using information derived from the capture name.

        :return: None.
        """

        parts = self.capture_name.split("_", 1)
        dt = datetime.strptime(parts[1], "%Y-%m-%d_%H%MZ")
        self.capture_datetime = dt

        return None

    def _set_capture_name(self) -> None:
        """
        Format and set the capture name using information derived from the capture file path.

        :return: None.
        """

        capture_name = self.hypso_path.stem

        for pl in SUPPORTED_PRODUCT_LEVELS:

            if "-" + pl in capture_name:
                capture_name = capture_name.replace("-" + pl, "")

        self.capture_name = capture_name

        return None

    def _set_capture_region(self) -> None:
        """
        Format and set the capture region using information derived from the capture name.

        :return: None.
        """

        self.capture_region = self.capture_name.split('_')[0].strip('_')

        return None

    def _set_capture_type(self) -> None:
        """
        Format and set the capture region using information derived from the capture name.

        :return: None.
        """

        if self.capture_config["frame_count"] == self.standard_dimensions["nominal"]:

            self.capture_type = "nominal"

        elif self.image_height == self.standard_dimensions["wide"]:

            self.capture_type = "wide"
        else:
            # EXPERIMENTAL_FEATURES
            if self.VERBOSE:
                print("[WARNING] Number of Rows (AKA frame_count) Is Not Standard.")
            
            self.capture_type = "custom"

        if self.VERBOSE:
            print(f"[INFO] Capture capture type: {self.capture_type}")

        return None

    def _set_adcs_dataframes(self) -> None:

        self._set_adcs_pos_dataframe()
        self._set_adcs_quat_dataframe()

        return None

    # TODO move DataFrame formatting related code to geometry
    def _set_adcs_pos_dataframe(self) -> None:

        
        position_headers = ["timestamp", "eci x [m]", "eci y [m]", "eci z [m]"]
        
        timestamps = self.adcs["timestamps"]
        pos_x = self.adcs["position_x"]
        pos_y = self.adcs["position_y"]
        pos_z = self.adcs["position_z"]

        pos_array = np.column_stack((timestamps, pos_x, pos_y, pos_z))
        pos_df = pd.DataFrame(pos_array, columns=position_headers)

        self.adcs_pos_df = pos_df

        return None

    # TODO move DataFrame formatting related code to geometry
    def _set_adcs_quat_dataframe(self) -> None:

        
        quaternion_headers = ["timestamp", "quat_0", "quat_1", "quat_2", "quat_3", "Control error [deg]"]

        timestamps = self.adcs["timestamps"]
        quat_s = self.adcs["quaternion_s"]
        quat_x = self.adcs["quaternion_x"]
        quat_y = self.adcs["quaternion_y"]
        quat_z = self.adcs["quaternion_z"]
        control_error = self.adcs["control_error"]

        quat_array = np.column_stack((timestamps, quat_s, quat_x, quat_y, quat_z, control_error))
        quat_df = pd.DataFrame(quat_array, columns=quaternion_headers)

        self.adcs_quat_df = quat_df

        return None

    def _set_nc_files(self) -> None:

        self.nc_file = self.hypso_path
        self.nc_name = self.hypso_path.stem

        file_path = str(self.hypso_path)

        self.l1a_nc_file = Path(file_path)
        self.l1b_nc_file = Path(file_path.replace("-l1a", "-l1b"))
        self.l2a_nc_file = Path(file_path.replace("-l1a", "-l2a"))

        self.l1a_nc_name = self.l1a_nc_file.stem
        self.l1b_nc_name = self.l1b_nc_file.stem
        self.l2a_nc_name = self.l2a_nc_file.stem

        return None

    def _set_tmp_dir(self) -> None:

        self.tmp_dir = Path(self.nc_file.parent.absolute(), self.nc_name.replace("-l1a", "") + "_tmp")

        return None
    
    def _set_nc_dir(self) -> None:

        self.nc_dir = Path(self.nc_file.parent.absolute())

        return None

    def _set_background_value(self) -> None:

        self.background_value = 8 * self.capture_config["bin_factor"]

        return None

    def _set_exposure(self) -> None:

        self.exposure = self.capture_config["exposure"] / 1000  # in seconds

        return None

    def _set_aoi(self) -> None:

        self.x_start = self.capture_config["aoi_x"]
        self.x_stop = self.capture_config["aoi_x"] + self.capture_config["column_count"]
        self.y_start = self.capture_config["aoi_y"]
        self.y_stop = self.capture_config["aoi_y"] + self.capture_config["row_count"]

    def _set_dimensions(self) -> None:

        self.bin_factor = self.capture_config["bin_factor"]

        self.row_count = self.capture_config["row_count"]
        self.frame_count = self.capture_config["frame_count"]
        self.column_count = self.capture_config["column_count"]

        self.image_height = self.capture_config["row_count"]
        self.image_width = int(self.capture_config["column_count"] / self.capture_config["bin_factor"])
        self.im_size = self.image_height * self.image_width

        self.bands = self.image_width
        self.lines = self.capture_config["frame_count"]  # AKA Frames AKA Rows
        self.samples = self.image_height  # AKA Cols

        self.spatial_dimensions = (self.capture_config["frame_count"], self.image_height)

        if self.VERBOSE:
            print(f"[INFO] Capture spatial dimensions: {self.spatial_dimensions}")

        return None

    def _set_timestamps(self) -> None:

        self.start_timestamp_capture = int(self.timing['capture_start_unix']) + UNIX_TIME_OFFSET

        # Get END_TIMESTAMP_CAPTURE
        # can't compute end timestamp using frame count and frame rate
        # assuming some default value if fps and exposure not available
        try:
            self.end_timestamp_capture = self.start_timestamp_capture + self.capture_config["frame_count"] / self.capture_config["fps"] + self.capture_config["exposure"] / 1000.0
        except:
            if self.VERBOSE:
                print("[WARNING] FPS or exposure values not found. Assuming 20.0 for each.")
            self.end_timestamp_capture = self.start_timestamp_capture + self.capture_config["frame_count"] / 20.0 + 20.0 / 1000.0

        # using 'awk' for floating point arithmetic ('expr' only support integer arithmetic): {printf \"%.2f\n\", 100/3}"
        time_margin_start = 641.0  # 70.0
        time_margin_end = 180.0  # 70.0

        self.start_timestamp_adcs = self.start_timestamp_capture - time_margin_start
        self.end_timestamp_adcs = self.end_timestamp_capture + time_margin_end

        self.unixtime = self.start_timestamp_capture
        self.iso_time = datetime.utcfromtimestamp(self.unixtime).isoformat()

        return None


    # Loading L1a

    def _load_l1a_nc_file(self, path: str = None) -> None:

        if path is not None:
            self.hypso_path = path

        self._validate_l1a_file()

        self._set_capture_name()
        self._set_capture_region()
        self._set_capture_datetime()

        
        self._load_l1a_metadata()

        self._set_nc_files()
        self._set_tmp_dir()
        self._set_nc_dir()

        self._set_background_value()
        self._set_exposure()
        self._set_aoi()
        self._set_dimensions()
        self._set_timestamps()
        self._set_capture_type()
        self._set_adcs_dataframes()
        #self._set_target_area()

        self._load_l1a_cube()

        return None

    def _validate_l1a_file(self) -> None:

        # Check that hypso_path file is a NetCDF file:
        #if not self.hypso_path.suffix == '.nc':
        #    raise Exception("Incorrect HYPSO Path. Only .nc files supported")

        match self.hypso_path.suffix:
            case '.nc':
                return None
            case '.bip':
                raise Exception("Incorrect HYPSO Path. Only .nc files supported")
            case _:
                raise Exception("Incorrect HYPSO Path. Only .nc files supported")
    
        return None

    def _load_l1a_cube(self) -> None:

        self.l1a_cube = load_l1a_nc_cube(self.hypso_path)

        return None

    # TODO: use setattr here?
    def _load_l1a_metadata(self) -> None:
        
        self.capture_config, \
            self.timing, \
            target_coords, \
            self.adcs, \
            dimensions = load_l1a_nc_metadata(self.hypso_path)
        
        return None

    def _get_l1a_cube(self) -> xr.DataArray:

        return self.l1a_cube


    # L1b functions

    # TODO
    def _load_l1b_file(self) -> None:
        return None

    # TODO
    def _load_l1b_cube(self) -> None:
        return None
    
    # TODO
    def _load_l1b_metadata(self) -> None:
        return None
    
    def _get_l1b_cube(self) -> xr.DataArray:

        return self.l1b_cube



    # L2a functions

    # TODO
    def _load_l2a_file(self) -> None:
        return None

    def _create_l2a_cube_xarray(self, l2a_cube: np.ndarray) -> xr.DataArray:

        l2a_cube = self._create_3d_xarray(data=l2a_cube)
        l2a_cube = self._set_l2a_cube_xarray_attrs(l2a_cube=l2a_cube)

        return l2a_cube

    def _set_l2a_cube_xarray_attrs(self, l2a_cube: xr.DataArray) -> xr.DataArray:

        l2a_cube.attrs['level'] = "L2a"
        l2a_cube.attrs['units'] = "a.u."
        l2a_cube.attrs['description'] = "Reflectance (Rrs)"

        return l2a_cube

    # TODO
    def _load_l2a_cube(self) -> None:
        return None
    
    # TODO
    def _load_l2a_metadata(self) -> None:
        return None


    # Georeferencing functions

    def _run_georeferencing(self, overwrite: bool = False) -> None:

        if self.georeferencing_has_run and not overwrite:

            if self.VERBOSE:
                    print("[INFO] Georeferencing has already been run. Skipping.")

            return None

        # Compute latitude and longitudes arrays if a points file is available
        if self.points_path is not None:

            if self.VERBOSE:
                print('[INFO] Running georeferencing...')

            gr = georeferencing.Georeferencer(filename=self.points_path,
                                              cube_height=self.spatial_dimensions[0],
                                              cube_width=self.spatial_dimensions[1],
                                              image_mode=None,
                                              origin_mode='qgis')
            
            # Update latitude and longitude arrays with computed values from Georeferencer
            self.latitudes = gr.latitudes[:, ::-1]
            self.longitudes = gr.longitudes[:, ::-1]
            
            self._run_georeferencing_orientation()
            self._compute_bbox()
            self._compute_gsd()
            self._compute_resolution()

            self.latitudes_original = self.latitudes
            self.longitudes_original = self.longitudes


        else:

            if self.VERBOSE:
                print('[INFO] No georeferencing .points file provided. Skipping georeferencing.')

        self.georeferencing_has_run = True

        return None

    # TODO: works with xarray?
    def _run_georeferencing_orientation(self) -> None:

        datacube_flipped = check_star_tracker_orientation(adcs_samples=self.adcs['adcssamples'],
                                                         quaternion_s=self.adcs['quaternion_s'],
                                                         quaternion_x=self.adcs['quaternion_x'],
                                                         quaternion_y=self.adcs['quaternion_y'],
                                                         quaternion_z=self.adcs['quaternion_z'],
                                                         velocity_x=self.adcs['velocity_x'],
                                                         velocity_y=self.adcs['velocity_y'],
                                                         velocity_z=self.adcs['velocity_z'])

        if not datacube_flipped:

            if self.l1a_cube is not None:
                self.l1a_cube = self.l1a_cube[:, ::-1, :]

            if self.l1b_cube is not None:  
                self.l1b_cube = self.l1b_cube[:, ::-1, :]
                
            if self.l2a_cube is not None:  
                self.l2a_cube = self.l2a_cube[:, ::-1, :]


        self.datacube_flipped = datacube_flipped

        return None
    
    def _compute_gsd(self) -> None:

        frame_count = self.frame_count
        image_height = self.image_height

        latitudes = self.latitudes
        longitudes = self.longitudes

        try:
            bbox = self.bbox
        except:
            self._compute_bbox()

        aoi = prj.aoi.AreaOfInterest(west_lon_degree=bbox[0],
                                     south_lat_degree=bbox[1],
                                     east_lon_degree=bbox[2],
                                     north_lat_degree=bbox[3], 
                                    )

        utm_crs_list = prj.database.query_utm_crs_info(datum_name="WGS 84", area_of_interest=aoi)


        #bbox_geodetic = [np.min(latitudes), 
        #                 np.max(latitudes), 
        #                 np.min(longitudes), 
        #                 np.max(longitudes)]

        #utm_crs_list = prj.database.query_utm_crs_info(datum_name="WGS 84",
        #                                                area_of_interest=prj.aoi.AreaOfInterest(
        #                                                west_lon_degree=bbox_geodetic[2],
        #                                                south_lat_degree=bbox_geodetic[0],
        #                                                east_lon_degree=bbox_geodetic[3],
        #                                                north_lat_degree=bbox_geodetic[1], )
        #                                            )
        
        if self.VERBOSE:
            print(f'[INFO] Using UTM map: ' + utm_crs_list[0].name, 'EPSG:', utm_crs_list[0].code)

        # crs_25832 = prj.CRS.from_epsg(25832) # UTM32N
        # crs_32717 = prj.CRS.from_epsg(32717) # UTM17S
        crs_4326 = prj.CRS.from_epsg(4326)  # Unprojected [(lat,lon), probably]
        source_crs = crs_4326
        destination_epsg = int(utm_crs_list[0].code)
        destination_crs = prj.CRS.from_epsg(destination_epsg)
        latlon_to_proj = prj.Transformer.from_crs(source_crs, destination_crs)


        pixel_coords_map = np.zeros([frame_count, image_height, 2])

        for i in range(frame_count):
            for j in range(image_height):
                pixel_coords_map[i, j, :] = latlon_to_proj.transform(latitudes[i, j], 
                                                                     longitudes[i, j])

        # time line x and y differences
        a = np.diff(pixel_coords_map[:, image_height // 2, 0])
        b = np.diff(pixel_coords_map[:, image_height // 2, 1])
        along_track_gsd = np.sqrt(a * a + b * b)
        along_track_mean_gsd = np.mean(along_track_gsd)

        # detector line x and y differences
        a = np.diff(pixel_coords_map[frame_count // 2, :, 0])
        b = np.diff(pixel_coords_map[frame_count // 2, :, 1])
        across_track_gsd = np.sqrt(a * a + b * b)
        across_track_mean_gsd = np.mean(across_track_gsd)


        self.along_track_gsd = along_track_gsd
        self.across_track_gsd = across_track_gsd

        self.along_track_mean_gsd = along_track_mean_gsd
        self.across_track_mean_gsd = across_track_mean_gsd

        return None

    def _compute_resolution(self) -> None:

        distances = [self.along_track_mean_gsd, 
                     self.across_track_mean_gsd]

        filtered_distances = [d for d in distances if d is not None]

        try:
            resolution = max(filtered_distances)
        except ValueError:
            resolution = 0

        self.resolution = resolution

        return None

    def _compute_bbox(self) -> None:

        lon_min = self.longitudes.min()
        lon_max = self.longitudes.max()
        lat_min = self.latitudes.min()
        lat_max = self.latitudes.max()

        bbox = (lon_min,lat_min,lon_max,lat_max)
        
        self.bbox = bbox

        return None


            




    # Calibration functions
        
    def _run_calibration(self, 
                         overwrite: bool = False,
                         rad_cal = True,
                         smile_corr = True,
                         destriping_corr = True,
                         **kwargs) -> None:
        """
        Get calibrated and corrected cube. Includes Radiometric, Smile and Destriping Correction.
            Assumes all coefficients has been adjusted to the frame size (cropped and
            binned), and that the data cube contains 12-bit values.

        :return: None
        """

        if self.calibration_has_run and not overwrite:

            if self.VERBOSE:
                    print("[INFO] Calibration has already been run. Skipping.")

            return None

        if self.VERBOSE:
            print('[INFO] Running calibration routines...')

        self._set_calibration_coeff_files()
        self._set_calibration_coeffs()
        self._set_wavelengths()
        self._set_srf()

        l1a_cube = self.l1a_cube.to_numpy()

        if rad_cal:
            l1b_cube = self._run_radiometric_calibration(cube=l1a_cube)
        if smile_corr:
            l1b_cube = self._run_smile_correction(cube=l1b_cube)
        if destriping_corr:
            l1b_cube = self._run_destriping_correction(cube=l1b_cube, **kwargs)


        #l1b_cube = self._format_l1b_cube(data=l1b_cube)

        self.l1b_cube = l1b_cube
 
        self.calibration_has_run = True

        return None

    def _run_radiometric_calibration(self, cube: np.ndarray) -> np.ndarray:

        # Radiometric calibration

        if self.VERBOSE:
            print("[INFO] Running radiometric calibration...")

        #cube = self._get_flipped_cube(cube=cube)

        cube = run_radiometric_calibration(cube=cube, 
                                           background_value=self.background_value,
                                           exp=self.exposure,
                                           image_height=self.image_height,
                                           image_width=self.image_width,
                                           frame_count=self.frame_count,
                                           rad_coeffs=self.rad_coeffs)

        #cube = self._get_flipped_cube(cube=cube)

        # TODO: The factor by 10 is to fix a bug in which the coeff have a factor of 10
        cube = cube / 10

        return cube

    def _run_smile_correction(self, cube: np.ndarray) -> np.ndarray:

        # Smile correction

        if self.VERBOSE:
            print("[INFO] Running smile correction...")

        #cube = self._get_flipped_cube(cube=cube)

        cube = run_smile_correction(cube=cube, 
                                    smile_coeffs=self.smile_coeffs)

        #cube = self._get_flipped_10cube(cube=cube)

        return cube

    def _run_destriping_correction(self, cube: np.ndarray, compute_destriping_matrix=False) -> np.ndarray:
        """
        Apply destriping correction to L1a datacube.

        :param cube: Radiometrically and smile corrected L1a datacube.

        :return: Destriped datacube.
        """

        if self.VERBOSE:
            print("[INFO] Running destriping correction...")

        #cube = self._get_flipped_cube(cube=cube)

        if compute_destriping_matrix:
            water_mask = make_mask(cube=cube, sat_val_scale=0.25)
            
            #overexposed_mask = make_overexposed_mask(cube=cube)
            #overexposed_mask = overexposed_mask.astype(bool)
            #self.water_mask = water_mask
            #self.overexposed_mask = overexposed_mask
            #mask = water_mask | ~overexposed_mask
            #mask = np.full(self.spatial_dimensions, False)
            #self.mask = mask

            self.destriping_coeffs = get_destriping_correction_matrix(cube, water_mask=water_mask)

            cube = run_destriping_correction_with_computed_matrix(cube=cube, 
                                         destriping_coeffs=self.destriping_coeffs)
            
        else:

            cube = run_destriping_correction(cube=cube, destriping_coeffs=self.destriping_coeffs[:,:])
        
        #cube = self._get_flipped_cube(cube=cube)

        return cube

    def _set_calibration_coeffs(self) -> None:
        """
        Set the calibration coefficients included in the package. This includes radiometric,
        smile and destriping correction.

        :return: None.
        """
        
        self._set_radiometric_coeffs()
        self._set_smile_coeffs()
        self._set_destriping_coeffs()
        self._set_spectral_coeffs()
        
        return None

    def _set_radiometric_coeffs(self) -> None:
        """
        Set the radiometric calibration coefficients included in the package.

        :return: None.
        """

        self.rad_coeffs = read_coeffs_from_file(self.rad_coeff_file)

        return None

    def _set_smile_coeffs(self) -> None:
        """
        Set the smile calibration coefficients included in the package.

        :return: None.
        """

        self.smile_coeffs = read_coeffs_from_file(self.smile_coeff_file)

        return None
    
    def _set_destriping_coeffs(self) -> None:
        """
        Set the destriping calibration coefficients included in the package.

        :return: None.
        """

        self.destriping_coeffs = read_coeffs_from_file(self.destriping_coeff_file)

        return None
    
    def _set_spectral_coeffs(self) -> None:
        """
        Set the spectral calibration coefficients included in the package.

        :return: None.
        """

        self.spectral_coeffs = read_coeffs_from_file(self.spectral_coeff_file)

        return None

    def _set_calibration_coeff_files(self) -> None:
        """
        Set the absolute path for the calibration coefficients included in the package. This includes radiometric,
        smile and destriping correction.

        :return: None.
        """

        self._set_rad_coeff_file()
        self._set_smile_coeff_file()
        self._set_destriping_coeff_file()
        self._set_spectral_coeff_file()

        return None

    def _set_rad_coeff_file(self, rad_coeff_file: Union[str, Path, None] = None) -> None:
        """
        Set the absolute path for the radiometric coefficients based on the detected capture type (wide, nominal, or custom).

        :param rad_coeff_file: Path to radiometric coefficients file (optional)

        :return: None.
        """

        if rad_coeff_file:
            self.rad_coeff_file = rad_coeff_file
            return None

        match self.capture_type:

            case "custom":
                csv_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_full_v1.csv"

            case "nominal":
                csv_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_nominal_v1.csv"

            case "wide":
                csv_file_radiometric = "radiometric_calibration_matrix_HYPSO-1_wide_v1.csv"

            case _:
                csv_file_radiometric = None

        if csv_file_radiometric:
            rad_coeff_file = files('hypso.calibration').joinpath(f'data/{csv_file_radiometric}')
        else: 
            rad_coeff_file = None

        self.rad_coeff_file = rad_coeff_file

        return None

    def _set_smile_coeff_file(self, smile_coeff_file: Union[str, Path, None] = None) -> None:

        """
        Set the absolute path for the smile coefficients based on the detected capture type (wide, nominal, or custom).

        :param smile_coeff_file: Path to smile coefficients file (optional)

        :return: None.
        """

        if smile_coeff_file:
            self.smile_coeff_file = smile_coeff_file
            return None

        match self.capture_type:

            case "custom":
                csv_file_smile = "spectral_calibration_matrix_HYPSO-1_full_v1.csv"   

            case "nominal":
                csv_file_smile = "smile_correction_matrix_HYPSO-1_nominal_v1.csv"

            case "wide":
                csv_file_smile = "smile_correction_matrix_HYPSO-1_wide_v1.csv"

            case _:
                csv_file_smile = None

        if csv_file_smile:
            smile_coeff_file = files('hypso.calibration').joinpath(f'data/{csv_file_smile}')
        else:
            smile_coeff_file = csv_file_smile

        self.smile_coeff_file = smile_coeff_file

        return None

    def _set_destriping_coeff_file(self, destriping_coeff_file: Union[str, Path, None] = None) -> None:

        """
        Set the absolute path for the destriping coefficients based on the detected capture type (wide, nominal, or custom).

        :param destriping_coeff_file: Path to destriping coefficients file (optional)

        :return: None.
        """

        if destriping_coeff_file:
            self.destriping_coeff_file = destriping_coeff_file
            return None

        match self.capture_type:

            case "custom":
                csv_file_destriping = None

            case "nominal":
                csv_file_destriping = "destriping_matrix_HYPSO-1_nominal_v1.csv"

            case "wide":
                csv_file_destriping = "destriping_matrix_HYPSO-1_wide_v1.csv"

            case _:
                csv_file_destriping = None

        if csv_file_destriping:
            destriping_coeff_file = files('hypso.calibration').joinpath(f'data/{csv_file_destriping}')
        else:
            destriping_coeff_file = csv_file_destriping

        self.destriping_coeff_file = destriping_coeff_file

        return None

    def _set_spectral_coeff_file(self, spectral_coeff_file: Union[str, Path, None] = None) -> None:
        """
        Set the absolute path for the spectral coefficients (wavelengths) based on the detected capture type (wide, nominal, or custom).

        :param spectral_coeff_file: Path to spectral coefficients file (optional)

        :return: None.
        """

        if spectral_coeff_file:
            self.spectral_coeff_file = spectral_coeff_file
            return None
        
        csv_file_spectral = "spectral_bands_HYPSO-1_v1.csv"

        spectral_coeff_file = files('hypso.calibration').joinpath(f'data/{csv_file_spectral}')

        self.spectral_coeff_file = spectral_coeff_file

        return None

    def _set_wavelengths(self) -> None:
        """
        Set the wavelengths corresponding to each of the 120 bands of the HYPSO datacube using information from the spectral coefficients.

        :return: None.
        """

        if self.spectral_coeffs is not None:
            self.wavelengths = self.spectral_coeffs
        else:
            self.wavelengths = None

        return None

    # TODO: move to calibration module?
    def _set_srf(self) -> None:
        """
        Set Spectral Response Functions (SRF) from HYPSO for each of the 120 bands. Theoretical FWHM of 3.33nm is
        used to estimate Sigma for an assumed gaussian distribution of each SRF per band.

        :return: None.
        """

        if not any(self.wavelengths):
            self.srf = None

        fwhm_nm = 3.33
        sigma_nm = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))

        srf = []
        for band in self.wavelengths:
            center_lambda_nm = band
            start_lambda_nm = np.round(center_lambda_nm - (3 * sigma_nm), 4)
            soft_end_lambda_nm = np.round(center_lambda_nm + (3 * sigma_nm), 4)

            srf_wl = [center_lambda_nm]
            lower_wl = []
            upper_wl = []
            for ele in self.wavelengths:
                if start_lambda_nm < ele < center_lambda_nm:
                    lower_wl.append(ele)
                elif center_lambda_nm < ele < soft_end_lambda_nm:
                    upper_wl.append(ele)

            # Make symmetric
            while len(lower_wl) > len(upper_wl):
                lower_wl.pop(0)
            while len(upper_wl) > len(lower_wl):
                upper_wl.pop(-1)

            srf_wl = lower_wl + srf_wl + upper_wl

            good_idx = [(True if ele in srf_wl else False) for ele in self.wavelengths]

            # Delta based on Hypso Sampling (Wavelengths)
            gx = None
            if len(srf_wl) == 1:
                gx = [0]
            else:
                gx = np.linspace(-3 * sigma_nm, 3 * sigma_nm, len(srf_wl))
            gaussian_srf = np.exp(
                -(gx / sigma_nm) ** 2 / 2)  # Not divided by the sum, because we want peak to 1.0

            # Get final wavelength and SRF
            srf_wl_single = self.wavelengths
            srf_single = np.zeros_like(srf_wl_single)
            srf_single[good_idx] = gaussian_srf

            srf.append([srf_wl_single, srf_single])

        self.srf = srf

        return None


    # Geometry computation functions

    def _run_geometry(self, overwrite: bool = False) -> None:

        if self.geometry_computation_has_run and not overwrite:

            if self.VERBOSE:
                    print("[INFO] Geometry computation has already been run. Skipping.")

            return None

        if self.VERBOSE:
            print("[INFO] Running geometry computation...")

        framepose_data = interpolate_at_frame(adcs_pos_df=self.adcs_pos_df,
                                              adcs_quat_df=self.adcs_quat_df,
                                              timestamps_srv=self.timing['timestamps_srv'],
                                              frame_count=self.capture_config['frame_count'],
                                              fps=self.capture_config['fps'],
                                              exposure=self.capture_config['exposure'],
                                              verbose=self.VERBOSE
                                              )


        wkt_linestring_footprint, \
           prj_file_contents, \
           local_angles, \
           geometric_meta_info, \
           pixels_lat, \
           pixels_lon, \
           sun_azimuth, \
           sun_zenith, \
           sat_azimuth, \
           sat_zenith = geometry_computation(framepose_data=framepose_data,
                                             image_height=self.image_height,
                                             verbose=self.VERBOSE
                                             )

        self.framepose_df = framepose_data

        self.wkt_linestring_footprint = wkt_linestring_footprint
        self.prj_file_contents = prj_file_contents
        self.local_angles = local_angles
        self.geometric_meta_info = geometric_meta_info

        self.solar_zenith_angles = sun_zenith.reshape(self.spatial_dimensions)
        self.solar_azimuth_angles = sun_azimuth.reshape(self.spatial_dimensions)

        self.sat_zenith_angles = sat_zenith.reshape(self.spatial_dimensions)
        self.sat_azimuth_angles = sat_azimuth.reshape(self.spatial_dimensions)

        #self.lat_original = pixels_lat.reshape(self.spatial_dimensions)
        #self.lon_original = pixels_lon.reshape(self.spatial_dimensions)

        self.latitudes_original = pixels_lat.reshape(self.spatial_dimensions)
        self.longitudes_original = pixels_lon.reshape(self.spatial_dimensions)

        self.geometry_computation_has_run = True

        return None




    # Atmospheric correction functions

    def _run_atmospheric_correction(self, product_name: str) -> None:

        match product_name.lower():
            case "6sv1":
                self.l2a_cube = self._run_6sv1_atmospheric_correction()
            case "acolite":
                self.l2a_cube = self._run_acolite_atmospheric_correction()
            case "machi":
                self.l2a_cube = self._run_machi_atmospheric_correction() 
            case _:
                print("[ERROR] No such atmospheric correction product supported!")
                return None
        return None

    def _run_6sv1_atmospheric_correction(self, **kwargs) -> xr.DataArray:

        # Py6S Atmospheric Correction
        # aot value: https://neo.gsfc.nasa.gov/view.php?datasetId=MYDAL2_D_AER_OD&date=2023-01-01
        # alternative: https://giovanni.gsfc.nasa.gov/giovanni/
        # atmos_dict = {
        #     'aot550': 0.01,
        #     # 'aeronet': r"C:\Users\alvar\Downloads\070101_151231_Autilla.dubovik"
        # }
        # AOT550 parameter gotten from: https://giovanni.gsfc.nasa.gov/giovanni/

        self._run_calibration(**kwargs)
        self._run_geometry()

        if self.VERBOSE: 
            print("[INFO] Running 6SV1 atmospheric correction")

        # TODO: which values should we use?
        if self.latitudes is None:
            latitudes = self.latitudes_original # fall back on geometry computed values
        else:
            latitudes = self.latitudes

        if self.longitudes is None:
            longitudes = self.longitudes_original # fall back on geometry computed values
        else:
            longitudes = self.longitudes

        atmos_params = {
            'aot550': 0.0580000256
        }

        time_capture = parser.parse(self.iso_time)

        cube = self.l1b_cube.to_numpy()

        cube = run_py6s(wavelengths=self.wavelengths, 
                        hypercube_L1=cube, 
                        lat_2d_array=latitudes,
                        lon_2d_array=longitudes,
                        solar_azimuth_angles=self.solar_azimuth_angles,
                        solar_zenith_angles=self.solar_zenith_angles,
                        sat_azimuth_angles=self.sat_azimuth_angles,
                        sat_zenith_angles=self.sat_zenith_angles,
                        iso_time=self.iso_time,
                        py6s_dict=atmos_params, 
                        time_capture=time_capture,
                        srf=self.srf)
        
        cube = self._format_l2a_dataarray(cube)
        cube.attrs['correction'] = "6sv1"

        return cube

    def _run_acolite_atmospheric_correction(self) -> xr.DataArray:

        self._run_calibration()
        self._run_geometry()
        self._write_l1b_nc_file()

        if not self._check_write_l1b_nc_file_has_run():

            raise RuntimeError(
                f"[ERROR] No L1b file found. Please generate it before attemping to run ACOLITE."
                )

            return None

        if self.VERBOSE: 
            print("[INFO] Running ACOLITE atmospheric correction")

        # user and password from https://urs.earthdata.nasa.gov/profile
        # optional but good
        atmos_params = {
            'user':'alvarof',
            'password':'nwz7xmu8dak.UDG9kqz'
        }

        if not self.l1b_nc_file.is_file():
            raise Exception("No -l1b.nc file found")
        
        cube = run_acolite(output_path=self.tmp_dir, 
                           atmos_dict=atmos_params, 
                           nc_file_acoliteready=self.l1b_nc_file)

        cube = self._format_l2a_dataarray(cube)
        cube.attrs['correction'] = "acolite"

        return cube
    
    # TODO
    def _run_machi_atmospheric_correction(self) -> xr.DataArray:

        print("[WARNING] Minimal Atmospheric Compensation for Hyperspectral Imagers (MACHI) atmospheric correction has not been enabled.")

        return None

        self._run_calibration()
        self._run_geometry()

        if self.VERBOSE: 
            print("[INFO] Running MACHI atmospheric correction")

        cube = self.l1b_cube.to_numpy()
        
        #T, A, objs = run_machi(cube=cube, verbose=self.VERBOSE)

        cube = self._format_l2a_dataarray(cube)
        cube.attrs['correction'] = "machi"

        return cube
    

    # Top of atmosphere reflectance functions

    # TODO: run product through validator, add proprty
    # TODO: move code to atmospheric module
    # TODO: add set functions
    def _run_toa_reflectance(self) -> None:

        self._run_calibration()
        self._run_geometry()
        
        # Get Local variables
        srf = self.srf
        toa_radiance = self.l1b_cube.to_numpy()

        scene_date = parser.isoparse(self.iso_time)
        julian_day = scene_date.timetuple().tm_yday
        solar_zenith = self.solar_zenith_angles

        # Read Solar Data
        solar_data_path = str(files('hypso.atmospheric').joinpath("Solar_irradiance_Thuillier_2002.csv"))
        solar_df = pd.read_csv(solar_data_path)

        # Create new solar X with a new delta
        solar_array = np.array(solar_df)
        current_num = solar_array[0, 0]
        delta = 0.01
        new_solar_x = [solar_array[0, 0]]
        while current_num <= solar_array[-1, 0]:
            current_num = current_num + delta
            new_solar_x.append(current_num)

        # Interpolate for Y with original solar data
        new_solar_y = np.interp(new_solar_x, solar_array[:, 0], solar_array[:, 1])

        # Replace solar Dataframe
        solar_df = pd.DataFrame(np.column_stack((new_solar_x, new_solar_y)), columns=solar_df.columns)

        # Estimation of TOA Reflectance
        band_number = 0
        toa_reflectance = np.empty_like(toa_radiance)
        for single_wl, single_srf in srf:
            # Resample HYPSO SRF to new solar wavelength
            resamp_srf = np.interp(new_solar_x, single_wl, single_srf)
            weights_srf = resamp_srf / np.sum(resamp_srf)
            ESUN = np.sum(solar_df['mW/m2/nm'].values * weights_srf)  # units matche HYPSO from device.py

            # Earth-Sun distance (from day of year) using julian date
            # http://physics.stackexchange.com/questions/177949/earth-sun-distance-on-a-given-day-of-the-year
            distance_sun = 1 - 0.01672 * np.cos(0.9856 * (
                    julian_day - 4))

            # Get toa_reflectance
            solar_angle_correction = np.cos(np.radians(solar_zenith))
            multiplier = (ESUN * solar_angle_correction) / (np.pi * distance_sun ** 2)
            toa_reflectance[:, :, band_number] = toa_radiance[:, :, band_number] / multiplier

            band_number = band_number + 1

        self.toa_reflectance = xr.DataArray(toa_reflectance, dims=("y", "x", "band"))
        self.toa_reflectance.attrs['units'] = "a.u."
        self.toa_reflectance.attrs['description'] = "Top of atmosphere (TOA) reflectance"

        self.toa_reflectance_has_run = True

        return None



    # Land mask functions

    def _run_land_mask(self, land_mask_name: str="global", **kwargs) -> None:

        land_mask_name = land_mask_name.lower()

        match land_mask_name:
            case "global":
                self.land_mask = self._run_global_land_mask(**kwargs)
            case "ndwi":
                self.land_mask = self._run_ndwi_land_mask(**kwargs)
            case "threshold":
                self.land_mask = self._run_threshold_land_mask(**kwargs)

            case _:

                print("[WARNING] No such land mask supported!")
                return None

        self.land_mask_has_run = True

        return None

    def _run_global_land_mask(self) -> np.ndarray:

        self._run_georeferencing()

        if self.VERBOSE:
            print("[INFO] Running global land mask generation...")

        land_mask = run_global_land_mask(spatial_dimensions=self.spatial_dimensions,
                                        latitudes=self.latitudes,
                                        longitudes=self.longitudes
                                        )
        
        land_mask = self._format_land_mask_dataarray(land_mask)
        land_mask.attrs['method'] = "global"

        return land_mask

    def _run_ndwi_land_mask(self) -> np.ndarray:

        self._run_calibration()

        if self.VERBOSE:
            print("[INFO] Running NDWI land mask generation...")

        cube = self.l1b_cube.to_numpy()

        land_mask = run_ndwi_land_mask(cube=cube, 
                                       wavelengths=self.wavelengths,
                                       verbose=self.VERBOSE)

        land_mask = self._format_land_mask_dataarray(land_mask)
        land_mask.attrs['method'] = "ndwi"

        return land_mask
    
    def _run_threshold_land_mask(self) -> xr.DataArray:

        self._run_calibration()

        if self.VERBOSE:
            print("[INFO] Running threshold land mask generation...")

        cube = self.l1b_cube.to_numpy()

        land_mask = run_threshold_land_mask(cube=cube,
                                            wavelengths=self.wavelengths,
                                            verbose=self.VERBOSE)
        
        land_mask = self._format_land_mask_dataarray(land_mask)
        land_mask.attrs['method'] = "threshold"

        return land_mask
     
    def _format_land_mask(self, land_mask: Union[np.ndarray, xr.DataArray]) -> None:

        land_mask_attributes = {
                                'description': "Land mask"
                               }
        
        v = DataArrayValidator(dims_shape=self.spatial_dimensions, dims_names=self.dim_names_2d, num_dims=2)

        data = v.validate(data=land_mask)
        data = data.assign_attrs(land_mask_attributes)

        return data



    # Cloud mask functions
        
    def _run_cloud_mask(self, cloud_mask_name: str="quantile_threshold", **kwargs) -> None:

        cloud_mask_name = cloud_mask_name.lower()

        match cloud_mask_name:
            case "quantile_threshold":

                if self.VERBOSE:
                    print("[INFO] Running quantile threshold cloud mask generation...")

                self.cloud_mask = self._run_quantile_threshold_cloud_mask(**kwargs)

            case _:
                print("[WARNING] No such cloud mask supported!")
                return None

        self.cloud_mask_has_run = True

        return None

    def _run_quantile_threshold_cloud_mask(self, quantile: float = 0.075) -> None:

        self._run_calibration()

        cloud_mask = run_quantile_threshold_cloud_mask(cube=self.l1b_cube,
                                                        quantile=quantile)

        cloud_mask = self._format_cloud_mask_dataarray(cloud_mask)
        cloud_mask.attrs['method'] = "quantile threshold"
        cloud_mask.attrs['quantile'] = quantile

        return cloud_mask 



    # Chlorophyll estimation functions

    def _run_chlorophyll_estimation(self, 
                                    product_name: str, 
                                    model: Union[str, Path] = None,
                                    factor: float = None,
                                    #overwrite: bool = False,
                                    **kwargs) -> None:

        match product_name.lower():

            case "band_ratio":

                if self.VERBOSE:
                    print("[INFO] Running band ratio chlorophyll estimation...")

                self.chl[product_name] = self._run_band_ratio_chlorophyll_estimation(factor=factor, **kwargs)
                
            case "6sv1_aqua":

                if self.VERBOSE:
                    print("[INFO] Running 6SV1 AQUA Tuned chlorophyll estimation...")

                self.chl[product_name] = self._run_6sv1_aqua_tuned_chlorophyll_estimation(model=model, **kwargs)

            case "acolite_aqua":

                if self.VERBOSE:
                    print("[INFO] Running ACOLITE AQUA Tuned chlorophyll estimation...")

                self.chl[product_name] = self._run_acolite_aqua_tuned_chlorophyll_estimation(model=model, **kwargs)

            case _:
                print("[ERROR] No such chlorophyll estimation product supported!")
                return None

        self.chlorophyll_estimation_has_run = True

        return None

    def _run_band_ratio_chlorophyll_estimation(self, factor: float = None) -> xr.DataArray:

        self._run_calibration()

        cube = self.l1b_cube.to_numpy()

        try:
            mask = self.unified_mask.to_numpy()
        except:
            mask = None

        chl = run_band_ratio_chlorophyll_estimation(cube = cube,
                                                    mask = mask, 
                                                    wavelengths = self.wavelengths,
                                                    spatial_dimensions = self.spatial_dimensions,
                                                    factor = factor
                                                    )

        chl_attributes = {
                        'method': "549 nm over 663 nm band ratio",
                        'factor': factor,
                        'units': "a.u."
                        }

        chl = self._format_chl(chl)
        chl = chl.assign_attrs(chl_attributes)

        return chl

    def _run_6sv1_aqua_tuned_chlorophyll_estimation(self, model: Path = None) -> xr.DataArray:

        self._run_calibration()
        self._run_geometry()

        if self.l2a_cube is None or self.l2a_cube.attrs['correction'] != '6sv1':
            self._run_atmospheric_correction(product_name='6sv1')

        model = Path(model)

        if not validate_tuned_model(model = model):
            print("[ERROR] Invalid model.")
            return None
        
        if self.spatial_dimensions is None:
            print("[ERROR] No spatial dimensions provided.")
            return None
        
        cube = self.l2a_cube.to_numpy()

        try:
            mask = self.unified_mask.to_numpy()
        except:
            mask = None

        chl = run_tuned_chlorophyll_estimation(l2a_cube = cube,
                                               model = model,
                                               mask = mask,
                                               spatial_dimensions = self.spatial_dimensions
                                               )
        
        chl_attributes = {
                        'method': "6SV1 AQUA Tuned",
                        'model': model,
                        'units': r'$mg \cdot m^{-3}$'
                        }

        chl = self._format_chl(chl)
        chl = chl.assign_attrs(chl_attributes)

        return chl

    def _run_acolite_aqua_tuned_chlorophyll_estimation(self, model: Path = None) -> xr.DataArray:

        self._run_calibration()
        self._run_geometry()

        if self.l2a_cube is None or self.l2a_cube.attrs['correction'] != 'acolite':
            self._run_atmospheric_correction(product_name='acolite')

        model = Path(model)

        if not validate_tuned_model(model = model):
            print("[ERROR] Invalid model.")
            return None
        
        if self.spatial_dimensions is None:
            print("[ERROR] No spatial dimensions provided.")
            return None

        cube = self.l2a_cube.to_numpy()

        try:
            mask = self.unified_mask.to_numpy()
        except:
            mask = None
        
        chl = run_tuned_chlorophyll_estimation(l2a_cube = cube,
                                               model = model,
                                               mask = mask,
                                               spatial_dimensions = self.spatial_dimensions
                                               )

        chl_attributes = {
                        'method': "ACOLITE AQUA Tuned",
                        'model': model,
                        'units': r'$mg \cdot m^{-3}$'
                        }

        chl = self._format_chl(chl)
        chl = chl.assign_attrs(chl_attributes)

        return chl

    def _format_chl(self, chl: Union[np.ndarray, xr.DataArray]) -> None:

        cloud_mask_attributes = {
                                'description': "Chlorophyll estimates"
                                }
        
        v = DataArrayValidator(dims_shape=self.spatial_dimensions, dims_names=self.dim_names_2d, num_dims=2)

        data = v.validate(data=chl)
        data = data.assign_attrs(cloud_mask_attributes)

        return data



    # SatPy functions

    def _generate_satpy_scene(self) -> Scene:

        scene = Scene()

        latitudes, longitudes = self._generate_satpy_latlons()

        latitude_attrs = {
                         'file_type': None,
                         'resolution': self.resolution,
                         'standard_name': 'latitude',
                         'units': 'degrees_north',
                         'start_time': self.capture_datetime,
                         'end_time': self.capture_datetime,
                         'modifiers': (),
                         'ancillary_variables': []
                         }

        longitude_attrs = {
                          'file_type': None,
                          'resolution': self.resolution,
                          'standard_name': 'longitude',
                          'units': 'degrees_east',
                          'start_time': self.capture_datetime,
                          'end_time': self.capture_datetime,
                          'modifiers': (),
                          'ancillary_variables': []
                          }

        scene['latitude'] = latitudes
        scene['latitude'].attrs.update(latitude_attrs)
        #scn['latitude'].attrs['area'] = swath_def

        scene['longitude'] = longitudes
        scene['longitude'].attrs.update(longitude_attrs)
        #scn['longitude'].attrs['area'] = swath_def

        return scene

    def _generate_satpy_latlons(self) -> tuple[xr.DataArray, xr.DataArray]:

        latitudes = xr.DataArray(self.latitudes, dims=self.dim_names_2d)
        longitudes = xr.DataArray(self.longitudes, dims=self.dim_names_2d)

        return latitudes, longitudes

    def _generate_swath_definition(self) -> SwathDefinition:

        latitudes, longitudes = self._generate_satpy_latlons()
        swath_def = SwathDefinition(lons=longitudes, lats=latitudes)

        return swath_def

    def _generate_l1a_satpy_scene(self) -> Scene:

        scene = self._generate_satpy_scene()
        swath_def= self._generate_swath_definition()

        try:
            cube = self.l1a_cube
        except:
            return None

        attrs = {
                'file_type': None,
                'resolution': self.resolution,
                'name': None,
                'standard_name': cube.attrs['description'],
                'coordinates': ['latitude', 'longitude'],
                'units': cube.attrs['units'],
                'start_time': self.capture_datetime,
                'end_time': self.capture_datetime,
                'modifiers': (),
                'ancillary_variables': []
                }   

        wavelengths = range(0,120)

        for i, wl in enumerate(wavelengths):

            data = cube[:,:,i]
                
            name = 'band_' + str(i+1)
            scene[name] = data
            scene[name].attrs.update(attrs)
            scene[name].attrs['wavelength'] = WavelengthRange(min=wl, central=wl, max=wl, unit="band")
            scene[name].attrs['band'] = i
            scene[name].attrs['area'] = swath_def

        return scene
    
    def _generate_l1b_satpy_scene(self) -> Scene:

        scene = self._generate_satpy_scene()
        swath_def= self._generate_swath_definition()

        try:
            cube = self.l1b_cube
            wavelengths = self.wavelengths
        except:
            return None

        attrs = {
                'file_type': None,
                'resolution': self.resolution,
                'name': None,
                'standard_name': cube.attrs['description'],
                'coordinates': ['latitude', 'longitude'],
                'units': cube.attrs['units'],
                'start_time': self.capture_datetime,
                'end_time': self.capture_datetime,
                'modifiers': (),
                'ancillary_variables': []
                }   

        for i, wl in enumerate(wavelengths):

            data = cube[:,:,i]

            name = 'band_' + str(i+1)
            scene[name] = xr.DataArray(data, dims=self.dim_names_2d)
            scene[name].attrs.update(attrs)
            scene[name].attrs['wavelength'] = WavelengthRange(min=wl, central=wl, max=wl, unit="nm")
            scene[name].attrs['band'] = i
            scene[name].attrs['area'] = swath_def

        return scene

    def _generate_l2a_satpy_scene(self) -> Scene:

        scene = self._generate_satpy_scene()
        swath_def= self._generate_swath_definition()

        try:
            cube = self.l2a_cube
            wavelengths = self.wavelengths
        except:
            return None

        attrs = {
                'file_type': None,
                'resolution': self.resolution,
                'name': None,
                'standard_name': cube.attrs['description'],
                'coordinates': ['latitude', 'longitude'],
                'units': cube.attrs['units'],
                'start_time': self.capture_datetime,
                'end_time': self.capture_datetime,
                'modifiers': (),
                'ancillary_variables': []
                }   

        for i, wl in enumerate(wavelengths):

            data = cube[:,:,i]

            name = 'band_' + str(i+1)
            scene[name] = data
            scene[name].attrs.update(attrs)
            scene[name].attrs['wavelength'] = WavelengthRange(min=wl, central=wl, max=wl, unit="nm")
            scene[name].attrs['band'] = i
            scene[name].attrs['area'] = swath_def

        return scene

    def _generate_chlorophyll_satpy_scene(self) -> Scene:

        scene = self._generate_satpy_scene()
        swath_def= self._generate_swath_definition()

        attrs = {
                'file_type': None,
                'resolution': self.resolution,
                'name': None,
                #'standard_name': cube.attrs['description'],
                'coordinates': ['latitude', 'longitude'],
                #'units': cube.attrs['units'],
                'start_time': self.capture_datetime,
                'end_time': self.capture_datetime,
                'modifiers': (),
                'ancillary_variables': []
                }   

        for key, chl in self.chl.items():

            if self._validate_array_dims(chl, ndim=2):

                #chl = item[1]
                #key = item[0]
                data = chl.to_numpy()
                name = 'chl_' + key
                scene[name] = xr.DataArray(data, dims=self.dim_names_2d)
                scene[name].attrs.update(attrs)
                scene[name].attrs['standard_name'] = chl.attrs['description']
                scene[name].attrs['units'] = chl.attrs['units']
                scene[name].attrs['area'] = swath_def

        return scene

    def _generate_products_satpy_scene(self) -> Scene:

        scene = self._generate_satpy_scene()
        swath_def= self._generate_swath_definition()

        attrs = {
                'file_type': None,
                'resolution': self.resolution,
                'name': None,
                'standard_name': None,
                'coordinates': ['latitude', 'longitude'],
                'units': None,
                'start_time': self.capture_datetime,
                'end_time': self.capture_datetime,
                'modifiers': (),
                'ancillary_variables': []
                }

        for key, product in self.products.items():

            if self._validate_array_dims(product, ndim=2):

                if self._is_xarray_dataarray(product):
                    data = product.to_numpy()
                else:
                    data = product

                scene[key] = xr.DataArray(data, dims=self.dim_names_2d)
                scene[key].attrs.update(attrs)

                scene[key].attrs['name'] = key
                scene[key].attrs['standard_name'] = key
                scene[key].attrs['area'] = swath_def

                try:
                    scene[key].attrs.update(product.attrs)
                except AttributeError:
                    pass


        return scene








    # Other functions

    def _create_2d_xarray(self, data: np.ndarray) -> xr.DataArray:

        return xr.DataArray(data, dims=self.dim_names_2d)

    def _create_3d_xarray(self, data: np.ndarray) -> xr.DataArray:

        return xr.DataArray(data, dims=self.dim_names_3d)


    




    # Replace with DataArrayValidator
    def _validate_array_dims(self, array: Union[np.ndarray, xr.DataArray], ndim: int = None) -> bool:

        # Check if the array is numpy or xarray format
        if not self._is_numpy_ndarray(array) and not self._is_xarray_dataarray(array):

            if self.VERBOSE:
                print('[WARNING] The array is not an xarray DataArray or NumPy ndarray object. Skipping.')
                
            return False
        
        # Check if number of dimensions in array is more than 2
        if array.ndim < 2:
            return False

        # Check if number of dimensions matches provided ndim
        if ndim:
            if not ndim == array.ndim:
                return False

        # Check if y and x dimensions of xarray match the capture spatial dimensions
        if not self._matches_spatial_dimensions(array):

            if self.VERBOSE:
                print('[WARNING] The array does not have dimensions ' + str(self.spatial_dimensions) + '. Skipping.')
                
            return False
        
        return True


    def _matches_spatial_dimensions(self, data: Union[np.ndarray, xr.DataArray]) -> bool:

        if data.shape[0:2] == self.spatial_dimensions:
            return True
        
        return False
    
    def _is_2d(self, data) -> bool:

        if data.ndim == 2:
            return True
        
        return False

    def _is_numpy_ndarray(self, data) -> bool:

        if type(data) is np.ndarray:
            return True
        
        return False
    
    def _is_xarray_dataarray(self, data) -> bool:
        
        if type(data) is xr.DataArray:
            return True
        
        return False

    def _get_flipped_cube(self, cube: np.ndarray) -> np.ndarray:

        if self.datacube_flipped is None:
            return cube
        else:
            if self.datacube_flipped:
                return cube[:, ::-1, :]

            else:
                return cube

        return cube
    




    # Public L1a methods

    def load_l1a_nc_file(self, path: str) -> None:

        self._load_l1a_nc_file(path=path)

        return None

    def get_l1a_cube(self) -> xr.DataArray:

        return self._get_l1a_cube()

    def get_l1a_spectrum(self, 
                        latitude=None, 
                        longitude=None,
                        x: int = None,
                        y: int = None
                        ) -> xr.DataArray:

        if self.l1a_cube is None:
            return None
        
        if latitude is not None and longitude is not None:
            idx = get_nearest_pixel(target_latitude=latitude, 
                                    target_longitude=longitude,
                                    latitudes=self.latitudes,
                                    longitudes=self.longitudes)

        elif x is not None and y is not None:
            idx = (x,y)

        else:
            return None

        spectrum = self.l1a_cube[idx[0], idx[1], :]

        return spectrum

    def plot_l1a_spectrum(self, 
                         latitude=None, 
                         longitude=None,
                         x: int = None,
                         y: int = None,
                         save: bool = False
                         ) -> None:
        
        if latitude is not None and longitude is not None:
            idx = get_nearest_pixel(target_latitude=latitude, 
                                    target_longitude=longitude,
                                    latitudes=self.latitudes,
                                    longitudes=self.longitudes)

        elif x is not None and y is not None:
            idx = (x,y)

        else:
            return None

        spectrum = self.l1a_cube[idx[0], idx[1], :]
        bands = range(0, len(spectrum))
        units = spectrum.attrs["units"]

        output_file = Path(self.nc_dir, self.capture_name + '_l1a_plot.png')

        plt.figure(figsize=(10, 5))
        plt.plot(bands, spectrum)
        plt.ylabel(units)
        plt.xlabel("Band number")
        plt.title(f"L1a (lat, lon) --> (X, Y) : ({latitude}, {longitude}) --> ({idx[0]}, {idx[1]})")
        plt.grid(True)

        if save:
            plt.imsave(output_file)
        else:
            plt.show()

        return None

    def write_l1a_nc_file(self, overwrite: bool = False) -> None:

        if Path(self.l1a_nc_file).is_file() and not overwrite:

            original_path = self.l1a_nc_file

            file_name = original_path.name
            modified_file_name = file_name.replace('-l1a', '-l1a-modified')
            modified_path = original_path.with_name(modified_file_name)

            dst_l1b_nc_file = modified_path

            if self.VERBOSE:
                print("[INFO] L1a NetCDF file has already been generated. Writing L1a data to: " + str(dst_l1b_nc_file))

        else:
            dst_l1b_nc_file = self.l1a_nc_file


        l1a_nc_writer(satobj=self, 
                      dst_l1b_nc_file=dst_l1b_nc_file, 
                      src_l1a_nc_file=self.l1a_nc_file)

        self.write_l1a_nc_file_has_run = True

        return None





    # Public L1b methods

    # TODO
    def load_l1b_nc_file(self, path: str) -> None:

        return None

    def generate_l1b_cube(self, **kwargs) -> None:

        self._run_calibration(**kwargs)

        return None

    def get_l1b_cube(self) -> xr.DataArray:

        return self._get_l1b_cube()

    def get_l1b_spectrum(self, 
                        latitude=None, 
                        longitude=None,
                        x: int = None,
                        y: int = None
                        ) -> tuple[np.ndarray, str]:

        if self.l1b_cube is None:
            return None
        
        if latitude is not None and longitude is not None:
            idx = get_nearest_pixel(target_latitude=latitude, 
                                    target_longitude=longitude,
                                    latitudes=self.latitudes,
                                    longitudes=self.longitudes)

        elif x is not None and y is not None:
            idx = (x,y)
            
        else:
            return None

        spectrum = self.l1b_cube[idx[0], idx[1], :]

        return spectrum

    def plot_l1b_spectrum(self, 
                        latitude=None, 
                        longitude=None,
                        x: int = None,
                        y: int = None,
                        save: bool = False
                        ) -> None:
        
        if latitude is not None and longitude is not None:
            idx = get_nearest_pixel(target_latitude=latitude, 
                                    target_longitude=longitude,
                                    latitudes=self.latitudes,
                                    longitudes=self.longitudes)

        elif x is not None and y is not None:
            idx = (x,y)

        else:
            return None

        spectrum = self.l1b_cube[idx[0], idx[1], :]
        bands = self.wavelengths
        units = spectrum.attrs["units"]

        output_file = Path(self.nc_dir, self.capture_name + '_l1a_plot.png')

        plt.figure(figsize=(10, 5))
        plt.plot(bands, spectrum)
        plt.ylabel(units)
        plt.xlabel("Wavelength (nm)")
        plt.title(f"L1b (lat, lon) --> (X, Y) : ({latitude}, {longitude}) --> ({idx[0]}, {idx[1]})")
        plt.grid(True)

        if save:
            plt.imsave(output_file)
        else:
            plt.show()

        return None

    def write_l1b_nc_file(self, overwrite: bool = False) -> None:

        if Path(self.l1b_nc_file).is_file() and not overwrite:

            if self.VERBOSE:
                print("[INFO] L1b NetCDF file has already been generated. Skipping.")

            return None

        l1b_nc_writer(satobj=self, 
                      dst_l1b_nc_file=self.l1b_nc_file, 
                      src_l1a_nc_file=self.l1a_nc_file)

        self.write_l1b_nc_file_has_run = True

        return None


    # Public L2a methods

    # TODO
    def load_l2a_cube(self, path: str, product_name: ATM_CORR_PRODUCTS = DEFAULT_ATM_CORR_PRODUCT) -> None:

        return None

    def generate_l2a_cube(self, product_name: ATM_CORR_PRODUCTS = DEFAULT_ATM_CORR_PRODUCT) -> None:

        self._run_atmospheric_correction(product_name=product_name)

        return None

    def get_l2a_cube(self) -> xr.DataArray:

        return self.l2a_cube

    def get_l2a_spectrum(self,
                        latitude=None, 
                        longitude=None,
                        x: int = None,
                        y: int = None
                        ) -> xr.DataArray:


        if latitude is not None and longitude is not None:
            idx = get_nearest_pixel(target_latitude=latitude, 
                                    target_longitude=longitude,
                                    latitudes=self.latitudes,
                                    longitudes=self.longitudes)

        elif x is not None and y is not None:
            idx = (x,y)
            
        else:
            return None

        try:
            spectrum = self.l2a_cube[idx[0], idx[1], :]
        except KeyError:
            return None

        return spectrum

    def plot_l2a_spectrum(self,
                         latitude=None, 
                         longitude=None,
                         x: int = None,
                         y: int = None,
                         save: bool = False
                         ) -> np.ndarray:
        
        if latitude is not None and longitude is not None:
            idx = get_nearest_pixel(target_latitude=latitude, 
                                    target_longitude=longitude,
                                    latitudes=self.latitudes,
                                    longitudes=self.longitudes)

        elif x is not None and y is not None:
            idx = (x,y)

        else:
            return None

        try:
            spectrum = self.l2a_cube[idx[0], idx[1], :]
        except KeyError:
            return None

        bands = self.wavelengths
        units = spectrum.attrs["units"]

        output_file = Path(self.nc_dir, self.capture_name + '_l2a_plot.png')

        plt.figure(figsize=(10, 5))
        plt.plot(bands, spectrum)
        plt.ylabel(units)
        plt.ylim([0, 1])
        plt.xlabel("Wavelength (nm)")
        plt.title(f"L2a (lat, lon) --> (X, Y) : ({latitude}, {longitude}) --> ({idx[0]}, {idx[1]})")
        plt.grid(True)

        if save:
            plt.imsave(output_file)
        else:
            plt.show()

    # TODO
    def write_l2a_nc_file(self, path: Union[str, Path] = None, product_name: str = None) -> None:

        self._write_l2a_nc_file(path=path, product_name=product_name)

        return None


    # Public georeferencing functions

    # TODO
    def load_georeferencing(self, path: str) -> None:

        return None
    
    # TODO
    def generate_georeferencing(self) -> None:

        return None
    
    # TODO
    def get_ground_control_points(self) -> None:

        return None

    # TODO
    def write_georeferencing(self, path: str) -> None:

        return None
    

    # Public geometry functions

    # TODO
    def load_geometry(self, path: str) -> None:

        return None

    def generate_geometry(self) -> None:

        self._run_geometry()

        return None

    # TODO
    def write_geometry(self, path: str) -> None:

        return None


    # Public land mask methods

    # TODO
    def load_land_mask(self, path: str) -> None:

        return None

    def generate_land_mask(self, land_mask_name: LAND_MASK_PRODUCTS = DEFAULT_LAND_MASK_PRODUCT, **kwargs) -> None:

        self._run_land_mask(land_mask_name=land_mask_name, **kwargs)

        return None

    def get_land_mask(self) -> xr.DataArray:

        return self.land_mask

    # TODO
    def write_land_mask(self, path: str) -> None:

        return None


    # Public cloud mask methods

    # TODO
    def load_cloud_mask(self, path: str) -> None:

        return None

    def generate_cloud_mask(self, cloud_mask_name: CLOUD_MASK_PRODUCTS = DEFAULT_CLOUD_MASK_PRODUCT, **kwargs):

        self._run_cloud_mask(cloud_mask_name=cloud_mask_name, **kwargs)

        return None

    def get_cloud_mask(self) -> xr.DataArray:

        return self.cloud_mask
    
    # TODO
    def write_cloud_mask(self, path: str) -> None:

        return None


    # Public unified mask methods

    def get_unified_mask(self) -> xr.DataArray:

        return self.unified_mask


    # Public chlorophyll methods

    # TODO
    def load_chlorophyll_estimates(self, path: str) -> None:

        return None

    def generate_chlorophyll_estimates(self, 
                                       product_name: CHL_EST_PRODUCTS = DEFAULT_CHL_EST_PRODUCT,
                                       model: Union[str, Path] = None,
                                       factor: float = 0.1
                                       ) -> None:

        self._run_chlorophyll_estimation(product_name=product_name, model=model, factor=factor)

    def get_chlorophyll_estimates(self, product_name: CHL_EST_PRODUCTS = DEFAULT_CHL_EST_PRODUCT,
                                 ) -> np.ndarray:

        key = product_name.lower()

        return self.chl[key]

    def get_chlorophyll_estimates_dict(self) -> dict:

        return self.chl

    # TODO
    def write_chlorophyll_estimates(self, path: str) -> None:

        return None
    


    # Public custom products methods
    
    # TODO
    def load_products(self, path: str) -> None:

        return None

    # TODO
    def write_products(self, path: str) -> None:

        return None



    # Public top of atmosphere (TOA) reflectance methods

    # TODO
    def load_toa_reflectance(self, path: str) -> None:
        
        return None

    def generate_toa_reflectance(self) -> None:

        self._run_toa_reflectance()

        return None

    def get_toa_reflectance(self) -> xr.DataArray:
        """
        Convert Top Of Atmosphere (TOA) Radiance to TOA Reflectance.

        :return: Array with TOA Reflectance.
        """

        return self.toa_reflectance


    # TODO
    def write_toa_reflectance(self, path: str) -> None:
        
        return None



    # TODO: organize

    def get_l1a_satpy_scene(self) -> Scene:

        return self._generate_l1a_satpy_scene()

    def get_l1b_satpy_scene(self) -> Scene:

        return self._generate_l1b_satpy_scene()

    def get_l2a_satpy_scene(self) -> Scene:

        return self._generate_l2a_satpy_scene()

    def get_chlorophyll_satpy_scene(self) -> Scene:

        return self._generate_chlorophyll_satpy_scene()

    def get_products_satpy_scene(self) -> Scene:

        return self._generate_products_satpy_scene()

    def get_bbox(self) -> tuple:
        
        return self.bbox
    
    def get_closest_wavelength_index(self, wavelength: Union[float, int]) -> int:

        wavelengths = np.array(self.wavelengths)
        differences = np.abs(wavelengths - wavelength)
        closest_index = np.argmin(differences)

        return closest_index





 

    # L2a file output

    # TODO
    def _write_l2a_nc_file(self, product_name: str = None, overwrite: bool = False) -> None:

        return None

    # TODO
    def _check_write_l2a_nc_file_has_run(self, product_name: str = None) -> bool:
        
        return self.write_l2a_nc_file_has_run

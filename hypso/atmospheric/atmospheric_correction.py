import matplotlib.pyplot as plt
import numpy as np
from .base import MeanDEM
from osgeo import gdal, osr
import os
import pandas as pd
import dateutil.parser
import Py6S
from tqdm import tqdm
import math
from scipy.interpolate import interp1d
from hypso.utils import find_file



def BasicParameters(wavelengths, hypercube_L1, hypso_info, lat_2d_array, lon_2d_array, time_capture):
    '''
    Get the parameters you need for 6s atmospheric correction
    '''

    # -------------------------------------------------
    #               Solar Parameters
    # -------------------------------------------------
    SixsParameters = dict()

    SixsParameters['time'] = time_capture

    SixsParameters['wavelengths'] = wavelengths
    SixsParameters['radiance_cube'] = hypercube_L1

    # Solar zenith angle, azimuth
    SixsParameters["SolarZenithAngle"] = hypso_info['solar_zenith_angle']
    SixsParameters["SolarAzimuthAngle"] = hypso_info['solar_azimuth_angle']

    # -------------------------------------------------
    #               Satellite Parameters
    # -------------------------------------------------
    # Satellite zenith angle, azimuth
    ViewZeniths = dict()
    ViewAzimuths = dict()
    # Make an 120 array with the average zenith and azimuth angle for every band
    # Ideally the average should be per band
    for i in range(120):
        # Each band has the average value
        ViewZeniths[i] = hypso_info['sat_zenith_angle']
        ViewAzimuths[i] = hypso_info['sat_azimuth_angle']

    SixsParameters["SatZenithAngles"] = ViewZeniths
    SixsParameters["SatAzimuthAngles"] = ViewAzimuths

    # -------------------------------------------------
    #                      Date
    # -------------------------------------------------
    # Date:Month, Day
    Date = dateutil.parser.isoparse(hypso_info['iso_time'])
    SixsParameters["ImgMonth"] = int(Date.month)
    SixsParameters["ImgDay"] = int(Date.day)

    # -------------------------------------------------
    #                Lat Lon Bounding Box
    # -------------------------------------------------
    lat = lat_2d_array
    lon = lon_2d_array

    # UPPER
    ULLat = lat[0, 0]
    # URLat = lat[0, -1]

    ULLon = lon[0, 0]
    # URLon = lon[0, -1]

    # BOTTOM
    # BLLat = lat[-1, 0]
    BRLat = lat[-1, -1]

    # BLLon = lon[-1, 0]
    BRLon = lon[-1, -1]

    min_lat = np.nanmin(lat)
    max_lat = np.nanmax(lat)

    min_lon = np.nanmin(lon)
    max_lon = np.nanmax(lon)

    print(f"ROI:\nMax Lat: {max_lat}  Min Lat: {min_lat}\nMax Lon: {max_lon}  Min Lon: {min_lon}")

    ImageCenterLon = (ULLon + BRLon) / 2
    ImageCenterLat = (ULLat + BRLat) / 2

    ImageCenterLon = np.mean([min_lon, max_lon])
    ImageCenterLat = np.mean([min_lat, max_lat])

    # Atmospheric mode type
    if -15 < ImageCenterLat <= 15:
        SixsParameters["AtmosphericProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.Tropical)

    elif (15 < ImageCenterLat <= 45) or (-45 <= ImageCenterLat < -15):
        if 4 < SixsParameters["ImgMonth"] <= 9:
            SixsParameters["AtmosphericProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.MidlatitudeSummer)
        else:
            SixsParameters["AtmosphericProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.MidlatitudeWinter)

    elif (45 < ImageCenterLat <= 60) or (-60 <= ImageCenterLat < -45):
        if 4 < SixsParameters["ImgMonth"] <= 9:
            SixsParameters["AtmosphericProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.SubarcticSummer)
        else:
            SixsParameters["AtmosphericProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.SubarcticWinter)

    rounded_lat = round(ImageCenterLat, -1)

    # Data from Table 2-2 in http://www.exelisvis.com/docs/FLAASH.html
    SAW = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.SubarcticWinter)
    SAS = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.SubarcticSummer)
    MLS = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.MidlatitudeSummer)
    MLW = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.MidlatitudeWinter)
    T = Py6S.AtmosProfile.PredefinedType(Py6S.AtmosProfile.Tropical)

    ap_JFMA = {
        80: SAW,
        70: SAW,
        60: MLW,
        50: MLW,
        40: SAS,
        30: MLS,
        20: T,
        10: T,
        0: T,
        -10: T,
        -20: T,
        -30: MLS,
        -40: SAS,
        -50: SAS,
        -60: MLW,
        -70: MLW,
        -80: MLW,
    }

    ap_MJ = {
        80: SAW,
        70: MLW,
        60: MLW,
        50: SAS,
        40: SAS,
        30: MLS,
        20: T,
        10: T,
        0: T,
        -10: T,
        -20: T,
        -30: MLS,
        -40: SAS,
        -50: SAS,
        -60: MLW,
        -70: MLW,
        -80: MLW,
    }

    ap_JA = {
        80: MLW,
        70: MLW,
        60: SAS,
        50: SAS,
        40: MLS,
        30: T,
        20: T,
        10: T,
        0: T,
        -10: T,
        -20: MLS,
        -30: MLS,
        -40: SAS,
        -50: MLW,
        -60: MLW,
        -70: MLW,
        -80: SAW,
    }

    ap_SO = {
        80: MLW,
        70: MLW,
        60: SAS,
        50: SAS,
        40: MLS,
        30: T,
        20: T,
        10: T,
        0: T,
        -10: T,
        -20: MLS,
        -30: MLS,
        -40: SAS,
        -50: MLW,
        -60: MLW,
        -70: MLW,
        -80: MLW,
    }

    ap_ND = {
        80: SAW,
        70: SAW,
        60: MLW,
        50: SAS,
        40: SAS,
        30: MLS,
        20: T,
        10: T,
        0: T,
        -10: T,
        -20: T,
        -30: MLS,
        -40: SAS,
        -50: SAS,
        -60: MLW,
        -70: MLW,
        -80: MLW,
    }

    ap_dict = {
        1: ap_JFMA,
        2: ap_JFMA,
        3: ap_JFMA,
        4: ap_JFMA,
        5: ap_MJ,
        6: ap_MJ,
        7: ap_JA,
        8: ap_JA,
        9: ap_SO,
        10: ap_SO,
        11: ap_ND,
        12: ap_ND,
    }

    SixsParameters["AtmosphericProfile"] = ap_dict[Date.month][rounded_lat]

    # Find the DEM height by studying the range of the area.
    pointUL = dict()
    pointDR = dict()
    pointUL["lat"] = ULLat
    pointUL["lon"] = ULLon
    pointDR["lat"] = BRLat
    pointDR["lon"] = BRLon

    # Modifications made due to HYPSO 2D Lat/Lon array not being squares, they may be skewed
    pointUL["lat"] = max_lat
    pointUL["lon"] = min_lon
    pointDR["lat"] = min_lat
    pointDR["lon"] = max_lon

    SixsParameters["meanDEM"] = (MeanDEM(pointUL, pointDR)) * 0.001

    # -------------------------------------------------
    #                Other Parameters
    # -------------------------------------------------
    # aerosol type continent
    SixsParameters["AeroProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AeroProfile.Maritime)
    # SixsParameters["AeroProfile"] = Py6S.AtmosProfile.PredefinedType(Py6S.AeroProfile.Continental)

    # Underlying surface type
    # SixsParameters["GroundReflectance"] = Py6S.GroundReflectance.HomogeneousLambertian(0.26)
    SixsParameters["GroundReflectance"] = Py6S.GroundReflectance.HomogeneousLambertian(Py6S.GroundReflectance.LakeWater)

    # 550nm aerosol optical thickness, obtained from MODIS based on date.
    # https://neo.gsfc.nasa.gov/analysis/index.php
    SixsParameters['aot550'] = 0.14497  # Constant value changed later if supplied
    SixsParameters['aot550'] = None

    # TOA Approach without SRF *****************************************************************

    # Non-homogeneous lower bedding surface, Lambertian
    # This should be used when the SRF is not known and it can be obtained from the TOA Reflectance
    # This is non tested example --------------------------------
    # Running for each wavelength with its respective reflectance value
    # for (i, j) in zip(wave, toa_reflec):
    #     s.wavelength = Wavelength(i)
    #     s.atmos_corr = Py6S.AtmosCorr.AtmosCorrLambertianFromReflectance(j)
    #     s.run()
    #     print "A18 wavelength: ", i, "Ref. : ", j, " ",
    #     boa_rec = s.outputs.atmos_corrected_reflectance_lambertian

    # s.atmos_corr = SixsInputParameter['AtmosCorrection']

    # Non-homogeneous lower bedding surface, Lambertian
    # SixsParameters['AtmosCorrection'] = Py6S.AtmosCorr.AtmosCorrLambertianFromReflectance(-0.1)
    # *****************************************************************

    return SixsParameters


def get_srf(fwhm, BandId, SixSParams):
    wavelengths = np.round(SixSParams['wavelengths'] / 1000, 4)  # to micrometers
    center_lambda = wavelengths[BandId]
    # From fwhm to sigma
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    start_lambda = np.round(center_lambda - (3 * sigma), 4)
    soft_end_lambda = np.round(center_lambda + (3 * sigma), 4)
    # Interval of 2.5nm from Py6S requirements
    delta = 2.5 / 1000  # nm to um
    lambda_py6s = [start_lambda]
    while np.max(lambda_py6s) < soft_end_lambda:
        lambda_py6s = np.append(lambda_py6s, np.max(lambda_py6s) + delta)

    # Delta based on Hypso Sampling (Wavelengths)
    gx = np.linspace(-3 * sigma, 3 * sigma, len(lambda_py6s))

    gaussian = np.exp(-(gx / sigma) ** 2 / 2)  # Not divided by the sum, because we want peak to 1.0

    # print("Gaussian sum: ", np.sum(gaussian))
    #
    # plt.figure(facecolor='white')
    # plt.plot(gx + center_lambda, gaussian)
    # plt.show()

    return lambda_py6s, gaussian


# 6s Atmospheric correction
def AtmosphericCorrection(BandId, SixsInputParameter, py6s_dict):
    '''
    Call the 6s model and assign values to the parameters to obtain the atmospheric correction parameters
    '''
    # 6S Models
    s = Py6S.SixS()

    # Enable Sensor type customization
    s.geometry = Py6S.Geometry.User()

    # Add Geometry Parameters
    s.geometry.solar_z = SixsInputParameter["SolarZenithAngle"]
    s.geometry.solar_a = SixsInputParameter["SolarAzimuthAngle"]
    s.geometry.view_z = SixsInputParameter["SatZenithAngles"][BandId]
    s.geometry.view_a = SixsInputParameter["SatAzimuthAngles"][BandId]

    # Date: Month, Day
    s.geometry.month = SixsInputParameter["ImgMonth"]
    s.geometry.day = SixsInputParameter["ImgDay"]

    # Type of atmospheric pattern
    s.atmos_profile = SixsInputParameter["AtmosphericProfile"]

    # Target Features
    s.ground_reflectance = SixsInputParameter["GroundReflectance"]

    # Aerosol Profile
    s.aero_profile = SixsInputParameter["AeroProfile"]  # Aerosol Type (Maritime Here)

    if py6s_dict is None:
        # Use Default Values
        s.aot550 = 0.14497  # Value checked from website for scene
    else:
        if 'aot550' in py6s_dict.keys():
            s.aot550 = py6s_dict['aot550']
            # Update AOT and Aero_profile if data provided
        elif 'aeronet' in py6s_dict.keys():
            s = Py6S.SixSHelpers.Aeronet.import_aeronet_data(s, py6s_dict['aeronet'], SixsInputParameter['time'])

    # Study area altitude, satellite sensor orbit altitude
    s.altitudes = Py6S.Altitudes()

    s.altitudes.set_target_custom_altitude(SixsInputParameter["meanDEM"])
    s.altitudes.set_sensor_satellite_level()

    # Correction band (according to band name)
    # SRF needs to be provided
    # for hypso a Gaussian of FWHM of 5nm is assumed

    fwhm = 5 / 1000  # micrometers need to be sent
    fwhm = 3.33 / 1000
    lambda_py6s, srf = get_srf(fwhm, BandId, SixsInputParameter)
    s.wavelength = Py6S.Wavelength(start_wavelength=lambda_py6s[0], end_wavelength=lambda_py6s[-1], filter=srf)

    # TOA Approach without SRF *****************************************************************

    # Non-homogeneous lower bedding surface, Lambertian
    # This should be used when the SRF is not known and it can be obtained from the TOA Reflectance
    # This is non tested example --------------------------------
    # Running for each wavelength with its respective reflectance value
    # for (i, j) in zip(wave, toa_reflec):
    #     s.wavelength = Wavelength(i)
    #     s.atmos_corr = Py6S.AtmosCorr.AtmosCorrLambertianFromReflectance(j)
    #     s.run()
    #     print "A18 wavelength: ", i, "Ref. : ", j, " ",
    #     boa_rec = s.outputs.atmos_corrected_reflectance_lambertian

    # s.atmos_corr = SixsInputParameter['AtmosCorrection']
    # *****************************************************************
    # Run 6s atmospheric model
    s.run()

    # extract 6S outputs
    Edir = s.outputs.direct_solar_irradiance  # direct solar irradiance
    Edif = s.outputs.diffuse_solar_irradiance  # diffuse solar irradiance
    Lp = s.outputs.atmospheric_intrinsic_radiance  # path radiance

    absorb = s.outputs.trans['global_gas'].upward  # absorption transmissivity
    scatter = s.outputs.trans['total_scattering'].upward  # scattering transmissivity
    tau = absorb * scatter  # total transmissivity

    # Coefficients
    # xa = s.outputs.coef_xa
    # xb = s.outputs.coef_xb
    # xc = s.outputs.coef_xc

    xa = None
    xb = None
    xc = None

    results_dict = {
        'direct_solar_irradiance': Edir,
        'diffuse_solar_irradiance': Edif,
        'path_radiance': Lp,
        'absorb_trans': absorb,
        'scatter_trans': scatter,
        'tau': tau,
        'coeff_a': xa,
        'coeff_b': xb,
        'coeff_c': xc,
    }

    return results_dict



def get_surface_reflectance(radiance_band, py6s_results):
    """
    Calculate surface reflectance from at-sensor radiance given waveband name
    """
    Lp = py6s_results['path_radiance']
    tau = py6s_results['tau']
    Edir = py6s_results['direct_solar_irradiance']
    Edif = py6s_results['diffuse_solar_irradiance']


    ref = np.subtract(radiance_band, Lp) * math.pi / (tau * (Edir + Edif))
    # ref = radiance_band.subtract(Lp).multiply(math.pi).divide(tau*(Edir+Edif))

    return ref


#
# def toa_reflectance_from_toa_radiance(wl, srf, radiance):
#     solar_df = pd.read_csv(r"Solar_irradiance_Thuillier_2002.csv")
#
#     solar_array = np.array(solar_df)
#     current_num = solar_array[0, 0]
#     delta = 0.01
#     new_x = [solar_array[0, 0]]
#     while current_num <= solar_array[-1, 0]:
#         current_num = current_num + delta
#         new_x.append(current_num)
#
#     new_y = np.interp(new_x, solar_array[:, 0], solar_array[:, 1])
#
#     solar_df = pd.DataFrame(np.column_stack((new_x, new_y)), columns=solar_df.columns)
#
#     resamp_srf = np.interp(new_x, wl, srf)
#     weights_srf = resamp_srf / np.sum(resamp_srf)
#     ESUN = np.sum(solar_df['mW/m2/nm'].values * weights_srf)
#
#     # Earth-Sun distance (from day of year)
#     doy = scene_date.timetuple().tm_yday  # julian date
#     distance_sun = 1 - 0.01672 * math.cos(0.9856 * (
#             doy - 4))  # http://physics.stackexchange.com/questions/177949/earth-sun-distance-on-a-given-day-of-the-year
#     TOA_ref = (np.pi * radiance * (distance_sun ** 2)) / (ESUN * np.cos(solar_zenith))
#     return TOA_ref


def get_corrected_radiance(radiance_band, py6s_results):
    coeff_a = py6s_results['coeff_a']
    coeff_b = py6s_results['coeff_b']
    coeff_c = py6s_results['coeff_c']

    temp_band = np.where(radiance_band != -9999, coeff_a * radiance_band - coeff_b, -9999)

    radiance_corr_band = np.where(temp_band != -9999, (temp_band / (1 + temp_band * coeff_c)) * 10000, -9999)

    return radiance_corr_band


def run_py6s(wavelengths, hypercube_L1, hypso_info, lat_2d_array, lon_2d_array, py6s_dict, time_capture):
    # Search for Full Geotiff
    potential_L2 = find_file(hypso_info["top_folder_name"],"-full_L2",".tif")
    if potential_L2 is not None:
        ds = gdal.Open(str(potential_L2))
        data = ds.ReadAsArray()
        return np.rot90(data.transpose((1, 2, 0)), k=2)
    print("\n-------  Py6S Atmospheric Correction  ----------")

    # Original units mW  (m^{-2} sr^{-1} nm^{-1})
    # hypercube_L1 = hypercube_L1 / 1000 # mW to W -> W  (m^{-2} sr^{-1} nm^{-1})
    #hypercube_L1 = hypercube_L1 / 0.001

    init_parameters = BasicParameters(wavelengths, hypercube_L1,
                                      hypso_info, lat_2d_array, lon_2d_array, time_capture)

    # Get Pre-Correction Cube
    radiance_hypercube = init_parameters['radiance_cube']
    atm_corr_ref_hypercube = np.empty_like(radiance_hypercube)

    for BandId in tqdm(range(120)):
        # Atmospheric correction for Zenith and Azimuth Band Values
        py6s_results = AtmosphericCorrection(BandId, init_parameters, py6s_dict)

        # Get Radiance Uncorrected Band
        uncorrected_radiance_band = radiance_hypercube[:, :, BandId]

        # Get Radiance-Corrected Band (We don't use it but it's an option)
        # I'm not really sure what the coefficients used here do
        #radiance_corr_band = get_corrected_radiance(uncorrected_radiance_band, py6s_results)

        # Get surface reflectance of Band (Method with SRF)
        reflectance_band = get_surface_reflectance(uncorrected_radiance_band, py6s_results)

        atm_corr_ref_hypercube[:, :, BandId] = reflectance_band

    # atm_corr_ref_hypercube = np.reshape(atm_corr_ref_hypercube,(120, -1))
    final_output = np.empty_like(atm_corr_ref_hypercube)

    for i in range(atm_corr_ref_hypercube.shape[0]):
        for j in range(atm_corr_ref_hypercube.shape[1]):
            spectra = atm_corr_ref_hypercube[i, j, :]
            wl = init_parameters['wavelengths']

            # Linear 1D Interp to Fill Values skipped due to AOT Variances
            nans = np.isnan(spectra)

            spectra[nans] = np.interp(wl[nans], wl[~nans], spectra[~nans])

            interp_spectra = spectra

            final_output[i, j, :] = interp_spectra



    return final_output

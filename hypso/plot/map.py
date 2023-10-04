# File Manipulation
import numpy as np
import pyproj
from matplotlib import colors
import warnings

import matplotlib.pyplot as plt
from matplotlib import ticker


# GIS
import cartopy.crs as ccrs
import cartopy.feature as cf
import geopandas as gpd
import rasterio as rio

from PIL import Image, ImageOps

PLOTZOOM = 1.0


def axis_extent(lat, lon):
    # Transform Current lon and lat limits to another
    extent_lon_min = np.nanmin(lon)
    extent_lon_max = np.nanmax(lon)

    extent_lat_min = np.nanmin(lat)
    extent_lat_max = np.nanmax(lat)

    return [extent_lon_min, extent_lon_max, extent_lat_min, extent_lat_max]


def image_extent(inproj_value, lat, lon):
    # Convert WKT projection information into a cartopy projection
    projcs = inproj_value.GetAuthorityCode("PROJCS")
    projection_img = ccrs.epsg(projcs)

    # Transform Current lon and lat limits to another
    new_max_lon = np.max(lon)
    new_max_lat = np.max(lat)
    new_min_lon = np.min(lon)
    new_min_lat = np.min(lat)

    # Convert lat and lon to the image CRS so we create Projection with Dataset CRS
    dataset_proj = pyproj.Proj(projection_img)  # your data crs

    # Transform Coordinates to Image CRS
    transformed_min_lon, transformed_min_lat = dataset_proj(
        new_min_lon, new_min_lat, inverse=False
    )
    transformed_max_lon, transformed_max_lat = dataset_proj(
        new_max_lon, new_max_lat, inverse=False
    )

    transformed_img_extent = (
        transformed_min_lon,
        transformed_max_lon,
        transformed_min_lat,
        transformed_max_lat,
    )

    return transformed_img_extent, projection_img


def tick_log_formatter(y, pos):
    # Find the number of decimal places required
    decimalplaces = int(np.maximum(-np.log10(y), 0))  # =0 for numbers >=1
    if decimalplaces == 0:
        # Insert that number into a format string
        formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
        # Return the formatted tick label
        return formatstring.format(y)
    else:
        formatstring = '{:2.1e}'.format(y)
        return formatstring

def get_cartopy_axis(satellite_obj,dpi_input):
    lat = satellite_obj.info["lat"]
    lon = satellite_obj.info["lon"]

    # Create Axis Transformation
    inproj = satellite_obj.projection_metadata["inproj"]
    extent_lon_min, extent_lon_max, extent_lat_min, extent_lat_max = axis_extent(
        lat, lon)
    transformed_img_extent, projection_img = image_extent(inproj, lat, lon)

    # crs is PlateCarree -> we are explicitly telling axes, that we are creating bounds that are in degrees
    projection = ccrs.Mercator()
    crs = ccrs.PlateCarree()

    # Now we will create axes object having specific projection
    fig = plt.figure(figsize=(10, 10), dpi=dpi_input, facecolor="white")
    fig.patch.set_alpha(1)
    ax = fig.add_subplot(projection=projection, frameon=True)

    ax.set_extent(
        [
            extent_lon_min / PLOTZOOM,
            extent_lon_max * PLOTZOOM,
            extent_lat_min / PLOTZOOM,
            extent_lat_max * PLOTZOOM,
        ],
        crs=crs,
    )

    # Draw gridlines in degrees over Mercator map
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.6, color="gray", alpha=0.5, linestyle="--"
    )
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    # To plot borders and coastlines, we can use cartopy feature
    ax.add_feature(cf.COASTLINE.with_scale("10m"), lw=0.5)
    ax.add_feature(cf.BORDERS.with_scale("10m"), lw=0.3)
    ax.add_feature(cf.LAKES.with_scale('10m'), lw=0.3, alpha=0.2, zorder=100)
    ax.add_feature(cf.OCEAN.with_scale('10m'), lw=0.3, alpha=0.2, zorder=100)
    # ax.add_feature(cf.RIVERS.with_scale('10m'), lw=0.3, alpha=0.5)
    # ax.add_feature(cf.LAND, zorder=100, edgecolor='k')  # Covers Data in land


    warnings.resetwarnings()

    return ax,transformed_img_extent,projection_img,lat,lon

def show_rgb_map(satellite_obj, plotTitle="RGB Image", dpi_input=450):

    # TODO: Warnings are disabled as a rounding error with shapely causes an "no intersection warning". New version of GDAL might solve it
    with np.errstate(all="ignore"): 
        # Get Axis for Map
        ax,transformed_img_extent,projection_img,lat,lon= get_cartopy_axis(satellite_obj,dpi_input)

        # Extract Image
        rgb_array = satellite_obj.projection_metadata["rgba_data"][:3, :, :]
        plot_rgb = np.ma.masked_where(rgb_array == 0, rgb_array)

        ax.imshow(
            np.rot90(plot_rgb.transpose((1, 2, 0)), k=2),
            # np.rot90(plot_rgb, k=2),
            # plot_rgb,
            origin="upper",
            extent=transformed_img_extent,
            transform=projection_img,
            zorder=1,
        )

        plt.title(plotTitle)
        plt.show()



def plot_array_overlay(satellite_obj, plot_array, plotTitle="2D Array",cbar_title=" Chlorophyll Concentration [mg m^-3]",dpi_input=450,min_value=0.01,max_value=100):
    MAXARRAY = max_value
    MINARRAY = min_value


    # TODO: Warnings are disabled as a rounding error with shapely causes an "no intersection warning". New version of GDAL might solve it
    with np.errstate(all="ignore"):

        # Get Axis for Map
        ax,transformed_img_extent,projection_img,lat,lon= get_cartopy_axis(satellite_obj,dpi_input)

        # Set Color
        min_actual_val = np.nanmin(plot_array)
        lower_limit = MINARRAY if min_actual_val < MINARRAY else min_actual_val

        max_actual_val = np.nanmax(plot_array)
        upper_limit = MAXARRAY if max_actual_val > MAXARRAY else max_actual_val

        # Set Range to next full log
        for i in range(-2, 3):
            full_log = 10**i
            if full_log < lower_limit:
                lower_limit = full_log
        for i in range(2, -3, -1):
            full_log = 10**i
            if full_log > upper_limit:
                upper_limit = full_log

        array_range = [lower_limit, upper_limit]

        print("2D Array Plot Range: ", array_range)

        # Log Normalize Color Bar colors
        norm = colors.LogNorm(array_range[0], array_range[1])
        im = ax.pcolormesh(
            lon,
            lat,
            plot_array,
            cmap=plt.cm.jet,
            transform=ccrs.PlateCarree(),
            norm=norm,
            zorder=0,
        )

        # Colourmap with axes to match figure size
        cbar = plt.colorbar(im, location="bottom", shrink=1, ax=ax, pad=0.05)

        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(tick_log_formatter))
        cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(tick_log_formatter))

        cbar.set_label(cbar_title)
        plt.title(plotTitle)

        plt.show()



def auto_adjust_img(img: np.ndarray) -> np.ndarray:
    """ Adjust image contrast using histogram equalization.

    Args:
        img: Image to adjust.

    Returns:
        Adjusted image.
    """

    img = Image.fromarray(np.uint8(img * 255 / np.max(img)))

    # Convert image to grayscale
    gray_img = ImageOps.grayscale(img)

    # Compute histogram
    hist = gray_img.histogram()

    # Compute cumulative distribution function (CDF)
    cdf = [sum(hist[:i + 1]) for i in range(len(hist))]

    # Normalize CDF
    cdf_normalized = [
        int((x - min(cdf)) * 255 / (max(cdf) - min(cdf))) for x in cdf]

    # Create lookup table
    lookup_table = dict(zip(range(256), cdf_normalized))

    # Apply lookup table to each channel
    channels = img.split()
    adjusted_channels = []
    for channel in channels:
        adjusted_channels.append(channel.point(lookup_table))

    # Merge channels and return result
    pil_image = Image.merge(img.mode, tuple(adjusted_channels))

    return np.array(pil_image)/255.0


def get_rgb(sat_obj,
               R_wl: float = 650,
               G_wl: float = 550,
               B_wl: float = 450) -> Image:
    """
    Write the RGB image.

    Args:
        path_to_save (str): The path to save the RGB image.
        R_wl (float, optional): The wavelength for the red channel. Defaults to 650.
        G_wl (float, optional): The wavelength for the green channel. Defaults to 550.
        B_wl (float, optional): The wavelength for the blue channel. Defaults to 450.
    """

    # R = np.argmin(abs(sat_obj.spectral_coefficients - R_wl))
    # G = np.argmin(abs(sat_obj.spectral_coefficients - G_wl))
    # B = np.argmin(abs(sat_obj.spectral_coefficients - B_wl))

    # get the rgb image
    # rgb = sat_obj.l1b_cube[:, :, [R, G, B]]
    # rgb_img = auto_adjust_img(rgb)

    rgb_array = sat_obj.projection_metadata["rgba_data"][:3, :, :]
    # plot_rgb = create_rgb(sat_obj=satellite_obj)

    plot_rgb = np.ma.masked_where(rgb_array == 0, rgb_array)

    rgb_img = np.rot90(plot_rgb.transpose((1, 2, 0)), k=2)

    return rgb_img


def write_rgb_to_png(
        sat_obj,
        path_to_save: str,
) -> Image:

    rgb_img = get_rgb(sat_obj)

    # check if file ends with .jpg
    if not path_to_save.endswith('.png'):
        path_to_save = path_to_save + '.png'


    fig = plt.figure(dpi=350, facecolor="white")
    plt.imshow(rgb_img, vmin=0, vmax=1.0)
    plt.savefig(path_to_save)

import flopy
import numpy as np
import pandas as pd
from pyproj import CRS
import rasterio
import matplotlib.pyplot as plt

def save_tif(tif, array, xmin, ymin, dx, dy=None, NODATA_value=0, dtype=None, compress="LZW", zlevel=9, crs="epsg:4326", rotation=0):
    """
    Save a NumPy array as a GeoTIFF file.

    Parameters:
    tif (str): Output file path for the GeoTIFF.
    array (numpy.ndarray): 2D or 3D array of raster data to save. If 3D, the first dimension represents bands.
    dx (float): Pixel width (resolution in x-direction).
    dy (float): Pixel height (resolution in y-direction, typically negative for north-up).
    xmin (float): X-coordinate of lower left corner.
    ymin (float): Y-coordinate of lower left corner.
    NODATA_value (int or float, optional): Value representing no data in the raster. Default is 0.
    dtype (data-type, optional): Data type of the output raster. If None, inferred from array.
    compress (str, optional): Compression method for GeoTIFF (e.g., "LZW", "DEFLATE"). Default is "LZW".
    zlevel (int, optional): Compression level (for DEFLATE or LZW). Default is 9.
    epsg (int, optional): EPSG code for the coordinate reference system. Default is 26913 (UTM Zone 13N).

    Returns:
    None: Saves the array as a GeoTIFF file.
    """

    array = np.flip(array, axis=-2) # flip rows so we can rotate with respective to origin (lower left)

    # Determine the number of bands
    if array.ndim == 2:
        array = array[np.newaxis, ...]  # Convert 2D to 3D (single-band)
    bands, ny, nx = array.shape
    if dy is None:
        dy = dx
    if "+proj" in crs.lower():
        crs = CRS.from_proj4(crs)
    else:
        crs = CRS.from_string(crs)

    rad_rotation = rotation*np.pi/180

    metadata = {
        'crs': crs,
        'transform': (
            np.cos(rad_rotation)*dx, -np.sin(rad_rotation)*dy, xmin, # note xmin is the lower left corner
            np.sin(rad_rotation)*dx, +np.cos(rad_rotation)*dy, ymin  # note ymin is the lower left corner
        ),
        'height': ny,
        'width': nx,
        'count': bands,
        'dtype': dtype or array.dtype,
        'driver': 'GTiff',
        'compress': compress,
        'zlevel': zlevel,
        'nodata': NODATA_value,
    }

    # Write the data to a new GeoTIFF file
    with rasterio.open(tif, 'w', **metadata) as dst:
        dst.write(array)

nlay     = 10       #
nrow     = 489      #
ncol     = 180      #
nper     = 474      #
xll      = 6453400  # feet
yll      = 1769800  # feet
dx       = 500      # feet
dy       = 500      # feet
rotation = 20       # degree
epsg     = 2226     # epsg

mf = flopy.modflow.Modflow.load("NVIHM.nam", model_ws=".", load_only=["rch"], version="mfnwt", check=False)

rch = mf.rch.rech.array * 12 * 365
#%%

save_tif("head_198404_flood.tif", np.maximum(0, head[ 0,0]-top), xll, yll, dx, dy, NODATA_value=0, crs=f"epsg:{epsg}", rotation=rotation)
save_tif("head_202309_flood.tif", np.maximum(0, head[-1,0]-top), xll, yll, dx, dy, NODATA_value=0, crs=f"epsg:{epsg}", rotation=rotation)
save_tif("head_highest_flood.tif", np.maximum(0, head[ :,0].max(axis=0)-top), xll, yll, dx, dy, NODATA_value=0, crs=f"epsg:{epsg}", rotation=rotation)
save_tif("head_lowest_flood.tif", np.maximum(0, head[ :,0].min(axis=0)-top), xll, yll, dx, dy, NODATA_value=0, crs=f"epsg:{epsg}", rotation=rotation)

save_tif("head_top_highest.tif", head[ :,0].max(axis=0), xll, yll, dx, dy, NODATA_value=-99999, crs=f"epsg:{epsg}", rotation=rotation)
save_tif("head_top_lowest.tif",  head[ :,0].min(axis=0), xll, yll, dx, dy, NODATA_value=-99999, crs=f"epsg:{epsg}", rotation=rotation)

save_tif("head_bedrock_highest.tif", head[ :,-1].max(axis=0), xll, yll, dx, dy, NODATA_value=-99999, crs=f"epsg:{epsg}", rotation=rotation)
save_tif("head_bedrock_lowest.tif",  head[ :,-1].min(axis=0), xll, yll, dx, dy, NODATA_value=-99999, crs=f"epsg:{epsg}", rotation=rotation)

#%% read obs
plt.style.use("seaborn-v0_8-whitegrid")
obs = pd.read_csv("../Postproc/Data/obsgwl.csv")
obs.obsname = obs.obsname.str.upper()
obs.time    = pd.to_datetime(obs.time)
sim = pd.read_csv("../Output_HOB.out", sep="\\s+")

sim = pd.merge(sim, obs, left_on="OBSERVATION NAME", right_on="obsname")

wells = pd.read_csv("../headerr.csv").set_index("wellname")
# wells["png"] = [rf"file:///simhead\{w}.png" for w in wells.wellname]
# wells["png"] = [rf"simhead\{w}.png" for w in wells.wellname]
# wells.to_csv("wells.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 6))
for w, df in sim.groupby("wellname"):
    if w not in wells.index: continue
    ax.clear()
    df.set_index("time").iloc[:,:2].plot(ax=ax, color=["r", "k"], )
    ax.legend()
    ax.set(title=f"{w}; Layer: {wells.loc[w, 'layertop']} - {wells.loc[w, 'layerbot']}", xlim=["1984", "2023-11"])
    fig.savefig(f"simhead/{w}.png", dpi=150, bbox_inches="tight")


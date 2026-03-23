import geopandas as gpd
import pandas as pd

wells = pd.read_csv(r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\04_GIS\well_logs.tsv", sep="\t")

wells = gpd.GeoDataFrame(wells, geometry=gpd.points_from_xy(wells.Longitude_NAD83, wells.Latitude_NAD83), crs="nad83")
wells = wells.to_crs("epsg:2226").set_index("LSCE_ID")


df = pd.read_csv("welllog.tsv", sep="\t")
df["X"] = wells.loc[df.LSCE_ID.values, "geometry"].x.values
df["Y"] = wells.loc[df.LSCE_ID.values, "geometry"].y.values
df.to_csv("welllog.csv", index=False)

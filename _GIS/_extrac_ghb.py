import flopy
import numpy as np
import pandas as pd
from pyproj import CRS
import rasterio
import matplotlib.pyplot as plt

with open("../InputFiles/GHB/GHB.ghb") as f:
    l = f.readline()
    l = f.readline()
    nghb = int(l.split()[0])

    ghb = pd.read_csv(f, header=None, nrows=nghb, sep="\\s+")
    ghb.columns = "layer row column head cond".split()

grid = pd.read_csv("gridxy.csv")

pd.merge(ghb, grid, on=["row", "column"]).to_csv("ghb_south.csv")

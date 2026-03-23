import flopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import sys
if r"c:\Cloud\Dropbox\PythonScripts\pyMEUK" not in sys.path:
    sys.path.insert(0, r"c:\Cloud\Dropbox\PythonScripts\pyMEUK")

from pyMEUK.variogram import raw_vgm, plot_vgm, avg_vgm, fit_vgm, plot_vgm_map, estimate_aniso_angle, plot_vgm_anisotropy3d, filter_vgm
#%%
models=["exponential"]
sigma_col=("variogram", "std")
x_col=("distance", "mean")
y_col=("variogram", "mean")
p0 = (0.03, 12000, 0.08)
maxfev=10000

pc = pd.read_csv("napa_COARSE_layavg.csv")
#%%
for l in range(1, 4):

    z = pc[f"Layer{l}"]
    mask = z>=0
    obsloc = pc.loc[mask, ["X", "Y"]].values
    zz = z[mask].values

    rv = raw_vgm(obsloc, zz)
    av = avg_vgm(rv, h_width=300, cutoff=20000)
    # ax = plot_vgm(av, models=models, plot_model=False, annotate=False)
    # ax.set(xlim=(0,30000))

    av.loc[0  , sigma_col] = 1e4
    av.loc[1:20, sigma_col] = 1
    av.loc[21: , sigma_col] = 10

    p, cov, ax = fit_vgm(av, x_col, y_col, sigma_col, models, p0, True, maxfev, xlabel="Horizontal distance (m)")

    ax = plot_vgm(av, models=models, plot_model=True, annotate=True, parameters=[0.075, 4000, 0.04])
    ax.figure.savefig(f"plot_horizontal_variogram{l}.png", dpi=300, bbox_inches="tight")

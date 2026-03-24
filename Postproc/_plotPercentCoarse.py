import flopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages

plt.style.use("ggplot")
logo = plt.imread("Postproc/data/ss-papadopulos-and-associates-logo.png")

mf = flopy.modflow.Modflow.load("NVIHM.nam", model_ws=".", load_only=["upw", "bas6"], version="mfnwt", check=False)
ib = mf.bas6.ibound.array[0]
ib = np.where(ib!=0, 1, np.nan)

pc = pd.read_csv(r"Texture2Par\napa_COARSE.csv")
with PdfPages('plotPercentCoarse.pdf') as pdf:
    
    for i in range(1, 7):
        zz = np.where(ib==1, np.loadtxt(rf"Data_ModelArrays\ZoneArrays\Lay{i}Zones.txt"), np.nan)
        vv = np.where(zz==1, np.loadtxt(rf"Data_ModelArrays\PctCoarse\Lay{i}_PctCoarse.txt"), np.nan)[:330]
        v0 = np.where(zz==1, pc[rf"Layer{i}"].values.reshape([-1, 180]), np.nan)[:330]
        fig, axs = plt.subplots(1, 2, figsize=(10, 7), tight_layout=True)
        axs[0].imshow(ib, cmap="gray", zorder=1, vmin=-5, vmax=3)
        axs[1].imshow(ib, cmap="gray", zorder=1, vmin=-5, vmax=3)
        im0 = axs[0].imshow(vv, cmap="RdYlBu_r", zorder=2, vmin=0, vmax=1)
        im1 = axs[1].imshow(v0, cmap="RdYlBu_r", zorder=2, vmin=0, vmax=1)
        axs[0].text(185, 5, "NVIHM", ha="right", va="top")
        axs[1].text(185, 5, "Texture2Par", ha="right", va="top")
        axs[0].axis("off"); axs[1].axis("off")
        fig.colorbar(im0, ax=axs[0], orientation="horizontal", pad=0.01, shrink=0.8, fraction=0.03, label="Percent of coarse grained sediment")
        fig.colorbar(im1, ax=axs[1], orientation="horizontal", pad=0.01, shrink=0.8, fraction=0.03, label="Percent of coarse grained sediment")
        #fig.savefig(pp+".png", dpi=100, bbox_inches="tight")
        # fig.figimage(logo, alpha=0.8)
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        del fig

import flopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages

plt.style.use("ggplot")

mf0 = flopy.modflow.Modflow.load("NVIHM.nam", model_ws=r"p:\1869-Modeling Services for SWRCB\03_Napa\Model\NVIHM_SWRCB_NWT", load_only=["upw", "bas6"], version="mfnwt", check=False)
mf = flopy.modflow.Modflow.load("NVIHM.nam", model_ws=".", load_only=["upw", "bas6"], version="mfnwt", check=False)
ib = mf.bas6.ibound.array
names = {
    "hk":"Horizontal hydraulic conductivity (feet/day)",
    "vka":"Vertical hydraulic conductivity (feet/day)",
    "ani":"Horizontal/Vertical anissotropy",
    "ss":"Specific storage (1/feet)",
    "sy":"Specific yield"}
logo = plt.imread("Postproc/data/ss-papadopulos-and-associates-logo.png")
#%%
with PdfPages('plotParameterECDF_Compare.pdf') as pdf:
    for pp in ["hk", "vka", "ani", "ss", "sy"]:
        if pp == "ani":
            par = np.where(ib==0, np.nan, getattr(mf.upw, "hk").array / getattr(mf.upw, "vka").array)
            par0= np.where(ib==0, np.nan, getattr(mf0.upw, "hk").array / getattr(mf0.upw, "vka").array)
        else:
            par = np.where(ib==0, np.nan, getattr(mf.upw, pp).array)
            par0= np.where(ib==0, np.nan, getattr(mf0.upw, pp).array)
        for i in range(mf.nlay):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.clear()
            ax.ecdf(par0[i][~np.isnan(par0[i])], orientation='horizontal', label="NVIHM")
            ax.ecdf(par[i][~np.isnan(par[i])], orientation='horizontal', label="Calibrated")
            ax.set(xlabel="Cumulative probability", ylabel="Parameter values", title=f"{names[pp]} Layer {i+1}", xlim=(-0.03, 1.03))
            if pp != "sy":
                ax.set_yscale("log")
                ax.grid(visible=True, which="both")
                ax.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
                ax.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs=(.1, .2, .3, .4, .5, .6, .7, .8, .9)))
                ax.grid(True, "minor", "y", alpha=0.5, linewidth=0.5)
            ax.text(-0.13, 0.5, "1", transform=ax.transAxes, color="none")
            ax.legend()
            fig.figimage(logo, alpha=0.8)
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            del fig

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.clear()
        ax.ecdf(par0[~np.isnan(par0)], orientation='horizontal', label="NVIHM")
        ax.ecdf(par[~np.isnan(par)], orientation='horizontal', label="Calibrated")
        ax.set(xlabel="Cumulative probability", ylabel="Parameter values", title=f"{names[pp]} All Layers", xlim=(-0.03, 1.03))
        ax.text(-0.13, 0.5, "1", transform=ax.transAxes, color="none")
        if pp != "sy":
            ax.set_yscale("log")
            ax.grid(visible=True, which="both")
            ax.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
            ax.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs=(.1, .2, .3, .4, .5, .6, .7, .8, .9)))
            ax.grid(True, "minor", "y", alpha=0.5, linewidth=0.5)
        ax.legend()
        fig.figimage(logo, alpha=0.8)
        pdf.savefig(fig, bbox_inches='tight', dpi=300)

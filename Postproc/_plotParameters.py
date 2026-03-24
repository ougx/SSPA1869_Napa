import flopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages

plt.style.use("ggplot")

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
with PdfPages('plotParameterECDF.pdf') as pdf:
    for pp in ["hk", "vka", "ani", "ss", "sy"]:
        if pp == "ani":
            par = np.where(ib==0, np.nan, getattr(mf.upw, "hk").array / getattr(mf.upw, "vka").array)
        else:
            par = np.where(ib==0, np.nan, getattr(mf.upw, pp).array)
        for i in range(mf.nlay):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.clear()
            ax.ecdf(par[i][~np.isnan(par[i])], orientation='horizontal')
            ax.set(xlabel="Cumulative probability", ylabel="Parameter values", title=f"{names[pp]} Layer {i+1}", xlim=(-0.03, 1.03))
            if pp != "sy":
                ax.set_yscale("log")
                ax.grid(visible=True, which="both")
                ax.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
                ax.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs=(.1, .2, .3, .4, .5, .6, .7, .8, .9)))
                ax.grid(True, "minor", "y", alpha=0.5, linewidth=0.5)
            ax.text(-0.13, 0.5, "1", transform=ax.transAxes, color="none")
            fig.figimage(logo, alpha=0.8)
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            del fig

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.clear()
        ax.ecdf(par[~np.isnan(par)], orientation='horizontal')
        ax.set(xlabel="Cumulative probability", ylabel="Parameter values", title=f"{names[pp]} All Layers", xlim=(-0.03, 1.03))
        ax.text(-0.13, 0.5, "1", transform=ax.transAxes, color="none")
        if pp != "sy":
            ax.set_yscale("log")
            ax.grid(visible=True, which="both")
            ax.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
            ax.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs=(.1, .2, .3, .4, .5, .6, .7, .8, .9)))
            ax.grid(True, "minor", "y", alpha=0.5, linewidth=0.5)
        fig.figimage(logo, alpha=0.8)
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
#%%
vmin = {"hk":0.2, "vka":5e-3, "ani":2, "ss":1e-6, "sy":0.0}
vmax = {"hk":200, "vka":5, "ani":100, "ss":1e-4, "sy":0.13}
with PdfPages('plotParameterMap.pdf') as pdf:
    for pp in ["hk", "vka", "ani", "ss", "sy"]:
        if pp == "ani":
            par = np.where(ib==0, np.nan, getattr(mf.upw, "hk").array / getattr(mf.upw, "vka").array)[:,:324]
        else:
            par = np.where(ib==0, np.nan, getattr(mf.upw, pp).array)[:,:324]
        for i in range(mf.nlay):
            fig1, ax1 = plt.subplots(figsize=(7, 10))
            pv = np.where(par[i]>0, par[i], np.nan)
            # if np.isnan(pv).all():
            #     continue
            if pp != "sy":
                im = ax1.imshow(pv, cmap="jet", norm=LogNorm(vmin[pp], vmax[pp]))
            else:
                im = ax1.imshow(pv, cmap="jet", vmin=vmin[pp], vmax=vmax[pp])
            ax1.grid(False)
            ax1.set_title(f"{names[pp]} Layer {i+1}", fontsize="medium")
            fig1.colorbar(im, pad=0.01, fraction=0.04, shrink=0.9, label="Parameter values")
            ax1.text(238, 150, "1",  color="none")
            fig1.figimage(logo, alpha=0.8)
            # fig1.savefig('plotParameterMap.png', bbox_inches='tight', dpi=300)
            pdf.savefig(fig1, bbox_inches='tight', dpi=300)
            del fig1

import flopy
import numpy as np
import matplotlib.pyplot as plt

#%%
parameters = {}
with open("parameter_orig.txt", "r") as f:
    while True:
        head = f.readline().strip()
        print(head)
        if head:
            v = np.loadtxt(f, max_rows=8802).reshape([489, -1])
            head = head.replace("HYD. COND. ALONG ROWS FOR LAYER", "hk")
            head = head.replace("VERTICAL HYD. COND. FOR LAYER", "vka")
            head = head.replace("SPECIFIC STORAGE FOR LAYER", "ss")
            head = head.replace("SPECIFIC YIELD FOR LAYER", "sy")
            head = head.replace(" ", "")
            parameters[head] = v.astype(float)
        else:
            break
#%%
mf = flopy.modflow.Modflow.load("napa.nam")


for par in ["hk", "vka", "sy", "ss"]:
    for i in range(10):
        pp = f"{par}{i+1}"
        vv = np.where(parameters[pp]==0, np.nan, parameters[pp])[:274]

        v0 = getattr(mf.upw, par).array[i][:274]
        v0 = np.where(v0<=0, np.nan, v0)


        fig, axs = plt.subplots(1, 2, figsize=(10, 7), tight_layout=True)
        im0 = axs[0].imshow(vv, cmap="jet")
        im1 = axs[1].imshow(v0, cmap="jet")
        axs[0].axis("off"); axs[1].axis("off")
        fig.colorbar(im0, ax=axs[0], orientation="horizontal", pad=0.01, shrink=0.8, fraction=0.03)
        fig.colorbar(im1, ax=axs[1], orientation="horizontal", pad=0.01, shrink=0.8, fraction=0.03)
        fig.savefig(pp+".png", dpi=100, bbox_inches="tight")

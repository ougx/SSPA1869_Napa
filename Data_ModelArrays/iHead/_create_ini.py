import numpy as np

ihead1 = np.loadtxt("Lay1_iHead.txt")
ibound = np.loadtxt("../iBound/ibound_top.txt")

top  = np.loadtxt("../LayerElevations/LandSurface.txt")
botm = np.loadtxt("../LayerElevations/Lay1Bottom.txt")

thick = top - botm

new = np.where(ihead1<-9990,  botm+0.9*np.minimum(20,thick), ihead1)

np.savetxt("iHead_new.dat", new, fmt="%.2f")

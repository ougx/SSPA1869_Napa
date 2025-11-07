import numpy as np
import pandas as pd

ib0 = np.loadtxt(r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\05_Model\00_model_meta\iBound_LSPC.txt") .reshape([489,-1]).astype(int)
ib1 = np.loadtxt(r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\05_Model\00_model_meta\ibound_nvihm.txt").reshape([489,-1]).astype(int)

zz = pd.read_csv(r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\05_Model\00_model_meta\Zones_nvihm.csv")

for i in range(10):
    iz = i + 1
    z = zz[f"Lay{iz}Zones"].values.reshape([489, -1])
    np.savetxt(f"Lay{iz}Zones.txt", np.where((ib0==1)&(ib1==0), 3, z), fmt="%s")

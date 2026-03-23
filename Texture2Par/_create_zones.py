import geopandas as gpd
import pandas as pd
import numpy as np

zones = pd.read_csv(r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\05_Model\00_model_meta\Zones_nvihm.csv")
ib0 = np.loadtxt(r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\05_Model\00_model_meta\ibound_nvihm.txt").flatten()
ib1 = np.loadtxt(r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\05_Model\00_model_meta\iBound_LSPC.txt").flatten()


for i in range(10):
    c = f"Lay{i+1}Zones"
    z = zones[c]
    zones.loc[(ib0==0)&(ib1==1), c] = 3
    zones.loc[ib1==0, c] = 0
    print(i+1, zones[c].unique())
# zones.loc[:, "Lay1Zones":] = zones.loc[:, "Lay1Zones":].replace({5:2, 3:2})
zones.to_csv(r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\05_Model\00_model_meta\Zones_LSPC.csv", index=False)

with open("zones.dat", "w") as f:
    f.write("1 2 3 4 5 6 7 8 9 9 \n")

    for i in range(9):
        c = f"Lay{i+1}Zones"
        z = zones[c].values.reshape([489, -1]).astype(int)
        np.savetxt(f, z, fmt="%s")
    f.write("constant 2\n")

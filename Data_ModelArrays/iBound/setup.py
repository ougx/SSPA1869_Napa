import numpy as np

ib0 = np.loadtxt(r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\05_Model\00_model_meta\iBound_LSPC.txt") .reshape([489,-1]).astype(int)
ib1 = np.loadtxt(r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\05_Model\00_model_meta\ibound_nvihm.txt").reshape([489,-1]).astype(int)

#np.savetxt("ibound_top.txt", ib0, fmt="%s")

np.savetxt("ibound_bedrock.txt", ib0*ib1, fmt="%s")
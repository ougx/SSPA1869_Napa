import pandas as pd
import numpy as np
import sys
if r"C:\Cloud\Dropbox\PythonScripts\a0_util" not in sys.path:
    sys.path.insert(0, r"C:\Cloud\Dropbox\PythonScripts\a0_util")
import MODFLOW

ibound = np.loadtxt("../../Data_ModelArrays/iBound/ibound_top.txt")
mnw = MODFLOW.mf_mnw(r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\05_Model\NVIHM_SWRCB_NWT\InputFiles\MNW2\MNW2.mnw", 487)
rc = pd.DataFrame([mnw.ROW, mnw.COL]).T
rc["ibound"] = [ibound[irc[0]-1, irc[1]-1] for irc in rc.sum(axis=1)]


mnw.WELLID = [w for w in mnw.WELLID if rc.loc[w, "ibound"]!=0]

mnw.write_package("MNW2.mnw")

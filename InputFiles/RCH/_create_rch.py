import numpy as np
import flopy
import pandas as pd
import sys

gwi = sys.argv[1]

nlay     = 10       #
nrow     = 489      #
ncol     = 180      #
nper     = 474      #
xll      = 6453400  # feet
yll      = 1769800  # feet
dx       = 500      # feet
dy       = 500      # feet
rotation = 20       # degree
CRS      = 2226     # epsg
rch_lspc = np.loadtxt(gwi) / 0.3048
rch_lspcs = rch_lspc.reshape([-1, nrow, ncol])[3:] # skip 1984-01 to 1984-03

times = pd.date_range("1984-04-01", periods=999, freq="MS")
scale = 1.0
with open("RCH.rch", "w") as f:
    f.write("3         50\n")
    for ii in range(nper):
        fname = 'RCH_' + times[ii].strftime("%Y%m") + '.rch'
        f.write(f"1         1 sp {ii+1}\n")
        f.write('OPEN/CLOSE InputFiles/RCH/' + fname + ' ' + str(scale) + ' (FREE) -1\n' )
        np.savetxt(fname, rch_lspcs[ii], fmt="%s")

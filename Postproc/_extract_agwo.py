import pandas as pd
import numpy as np
from io import StringIO


def readsfr2df(sfrfile):
    _header = 'period step layer row column segment reach qin qaq qout overland precip et stage depth width condutance gradient'.split()
    with open(sfrfile) as f:
        lines = f.readlines()

    ii = []
    for i, l in enumerate(lines):
        if 'PERIOD' in l:
            ii.append(i)

    dfs = []
    for i1,i2 in zip(ii, ii[1:]+[len(lines)]):
        sp = lines[i1][lines[i1].find('PERIOD')+6:].replace('STEP', '').strip() + ' '
        dfs.append(pd.read_csv(StringIO(''.join([sp + l for l in lines[i1+5:i2] if l != '\n'])), header=None,
                   sep=' ', skipinitialspace=True, index_col=False, names=_header))
    return pd.concat(dfs).reset_index(drop=True)

sfrfile = "Output_SFR.out"
sfr = readsfr2df(sfrfile)
sfr = sfr.groupby(["period", "segment", "reach"], as_index=False).qaq.mean()
sfr["GW2SW"] = np.minimum(0, sfr.qaq)
sfr["SW2GW"] = np.maximum(0, sfr.qaq)

rch = pd.read_csv("_GIS/sfr.csv")
rch = rch[rch.segment<=157]
rch["SWSID"] = rch["SWSID"].astype(int)


sfrout = pd.merge(sfr, rch, on=["segment", "reach"]).groupby(["period", "SWSID"])[["GW2SW", "SW2GW"]].sum()
sfrout["Net flow to GW"] = sfrout.GW2SW + sfrout.SW2GW
#(sfrout/86400).to_csv("AGWO_MODFLOW_cfs.csv")


tdis = pd.read_csv("Postproc/times _sp.csv")
tdis["dt"] = tdis.time.diff()
tdis.loc[0, "dt"] = 30

(sfrout * tdis.loc[sfrout.reset_index()["period"]-1, ["dt"]].values*(.3048**3)).to_csv("AGWO_MODFLOW_m3.csv")

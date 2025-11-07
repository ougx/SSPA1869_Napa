import pandas as pd
import numpy as np
import sys
if r"C:\Cloud\Dropbox\PythonScripts\a0_util" not in sys.path:
    sys.path.insert(0, r"C:\Cloud\Dropbox\PythonScripts\a0_util")
import MODFLOW

#%%
route = pd.read_csv(r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\05_Model\00_model_meta\sfr_transfer_matrix_inflow2.csv", index_col=0)
runoff = pd.read_csv(r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\05_Model\00_model_meta\sfr_transfer_matrix_runoff2.csv", index_col=0)

sfr = MODFLOW.mf_sfr(r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\05_Model\NVIHM_SWRCB_NWT\InputFiles\SFR\SFR.sfr", 1)
sfr.TABFILES = None
sfr.NSS = 157
sfr.REACH = sfr.REACH[sfr.REACH.ISEG.astype(int)<=sfr.NSS]
sfr.NSTRM = len(sfr.REACH)


sw = pd.read_csv(r"d:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\02_Incoming\20251105_LSPCoutput\Streams_WaterBalanceParams_20251105-175622_m3.csv")
sw["date"] = pd.to_datetime(sw.DTTM)
dv = pd.read_csv(r"d:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\02_Incoming\20251105_LSPCoutput\Actual_PointSource_Withdrawals_CubicFeetperMonth.csv")
dv["date"] = pd.to_datetime(dv.date)
dv["SWSID"] = dv.rchid
sw = pd.merge(sw, dv, on=["SWSID", "date"])

sw_route = sw[sw.SWSID.isin(route.index)].copy()
sw_route["q"] = ((sw_route.AGWO + sw_route.SURO + sw_route.IFWO) / (.3048**3)  - sw_route.PointSource_Withdrawls_cubicFeetperMonth) / sw_route.date.dt.day
sw_route = sw_route.pivot(index="date", columns="SWSID", values="q")
sw_route = sw_route.sort_index().loc["1984-04":].T.sort_index().T @ route


sw_runoff = sw[sw.SWSID.isin(runoff.index)].copy()
sw_runoff["q"] = ((sw_runoff.SURO + sw_runoff.IFWO) / .3048**3  - sw_runoff.PointSource_Withdrawls_cubicFeetperMonth) / sw_runoff.date.dt.day
sw_runoff = sw_runoff.pivot(index="date", columns="SWSID", values="q")
sw_runoff = sw_runoff.sort_index().loc["1984-04":].T.sort_index().T @ runoff

sw_route.columns = sw_route.columns.astype(int)
sw_runoff.columns = sw_runoff.columns.astype(int)
for iseg in range(sfr.NSS):
    sseg = (iseg +1)
    if sseg not in sw_route.columns:
        sw_route.loc[:, sseg] = 0.0
    if sseg not in sw_runoff.columns:
        sw_runoff.loc[:, sseg] = 0.0

sw_route.sort_index(axis=1, inplace=True)
sw_runoff.sort_index(axis=1, inplace=True)

nper = 474
sps = {}
for isp in range(nper):
    sp = sfr.sp[1].copy()[:sfr.NSS]
    for iseg in range(sfr.NSS):
        sp[iseg] = sfr.sp[1][iseg].copy()
        sp[iseg]["FLOW"  ] = sw_route .iloc[isp, iseg]    # A real number that is the streamflow (in units of volume per time)
        sp[iseg]["RUNOFF"] = sw_runoff.iloc[isp, iseg]    # A real number that is the volumetric rate of the diffuse overland runoff that enters the stream segment (in units of volume per time)

    sps[isp + 1] = sp

sfr.ITMP   = [sfr.NSS,] * nper
sfr.IRDFLG = [1,] * nper
sfr.IPTFLG = [0,] * nper
sfr.NP     = [0,] * nper
sfr.sp = sps
sfr.write_package("SFR.sfr")

sw_route .to_csv(r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\03_Analyse\20251105_LSPCoutput_QA\modflow_inflows_cfd.csv")
sw_runoff.to_csv(r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\03_Analyse\20251105_LSPCoutput_QA\modflow_runoff_cfd.csv")

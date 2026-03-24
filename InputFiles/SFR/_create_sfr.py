import pandas as pd
import numpy as np
import os
import sys
if r"C:\Cloud\Dropbox\PythonScripts\a0_util" not in sys.path:
    sys.path.insert(0, r"C:\Cloud\Dropbox\PythonScripts\a0_util")
import MODFLOW

lspc_ver = sys.argv[1]
lspc_stream = sys.argv[2]
lspc_withdraw = sys.argv[3]
lspc_lake = sys.argv[4].strip()

# lspc_ver      = "20260306_LSPCoutput"
# lspc_stream   = r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\02_Incoming\20260306_LSPCoutput\Streams_WaterBalanceParams_20260306-142227_m3.csv"
# lspc_withdraw = r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\02_Incoming\20260306_LSPCoutput\Actual_PointSource_Withdrawals_CubicFeetperMonth.csv"

os.chdir(os.path.join(os.environ["ONEDRIVE"], "1869-SWRCB_Napa"))
# lspc_ver = "20251105_LSPCoutput"
#%%
sw = pd.read_csv(lspc_stream)
dv = pd.read_csv(lspc_withdraw)
lk = pd.read_csv(lspc_lake) if lspc_lake.endswith("csv") else pd.read_excel(lspc_lake)
route = pd.read_csv(r"05_Model\00_model_meta\sfr_transfer_matrix_inflow2.csv", index_col=0)
runoff = pd.read_csv(r"05_Model\00_model_meta\sfr_transfer_matrix_runoff2.csv", index_col=0)
sfrorder = pd.read_csv(r"05_Model\00_model_meta\sfr_segment_order.csv", index_col=0)

sfr = MODFLOW.mf_sfr(r"05_Model\NVIHM_SWRCB_NWT\InputFiles\SFR\SFR.sfr", 1)
sfr.TABFILES = None
sfr.NSS = 157
sfr.REACH = sfr.REACH[sfr.REACH.ISEG.astype(int)<=sfr.NSS]
sfr.NSTRM = len(sfr.REACH)

sfrorder = sfrorder.loc[:sfr.NSS]
sfrorder.loc[sfrorder.OUTSEG>sfr.NSS, "OUTSEG"] = 0

sw["date"] = pd.to_datetime(sw.DTTM)
dv["date"] = pd.to_datetime(dv.date)
lk["Date"] = pd.to_datetime(lk.Date)
dv["SWSID"] = dv.rchid
sw = pd.merge(sw, dv, on=["SWSID", "date"], how="outer")

sw_route = sw[sw.SWSID.isin(route.index)].copy()
dv1 = sw_route.pivot(index="date", columns="SWSID", values="PointSource_Withdrawls_cubicFeetperMonth")
sw_route["q"] = ((sw_route.SURO + sw_route.IFWO) / (.3048**3)) / sw_route.date.dt.day
sw_route = sw_route.pivot(index="date", columns="SWSID", values="q")
sw_route = sw_route.sort_index().loc["1984-04":].T.sort_index().T @ route

dv1 = dv1 / dv1.index.day.values[:, None]
route1 = route.copy()
route1.columns = [str(sfrorder.loc[int(c), "OUTSEG"]) if sfrorder.loc[int(c), "OUTSEG"]>0 else c for c in route1.columns]
sw_dv1 = dv1.sort_index().loc["1984-04":].T.sort_index().T @ route1
sw_dv1 = sw_dv1.T.groupby(level=0).sum().T

sw_runoff = sw[sw.SWSID.isin(runoff.index)].copy()
dv2 = sw_runoff.pivot(index="date", columns="SWSID", values="PointSource_Withdrawls_cubicFeetperMonth")
sw_runoff["q"] = ((sw_runoff.SURO + sw_runoff.IFWO) / (.3048**3)) / sw_runoff.date.dt.day
sw_runoff = sw_runoff.pivot(index="date", columns="SWSID", values="q")
sw_runoff = sw_runoff.sort_index().loc["1984-04":].T.sort_index().T @ runoff

dv2 = dv2 / dv2.index.day.values[:, None]
runoff1 = runoff.copy()
runoff1.columns = [str(sfrorder.loc[int(c), "OUTSEG"]) if sfrorder.loc[int(c), "OUTSEG"]>0 and int(c)<sfr.NSS else c for c in runoff1.columns]
sw_dv2 = dv2.sort_index().loc["1984-04":].T.sort_index().T @ runoff1
sw_dv2 = sw_dv2.T.groupby(level=0).sum().T


sw_route.columns = sw_route.columns.astype(int)
sw_runoff.columns = sw_runoff.columns.astype(int)
sw_dv1.columns = sw_dv1.columns.astype(int)
sw_dv2.columns = sw_dv2.columns.astype(int)

zeros = pd.DataFrame(0, index=sw_route.index, columns=range(1, sfr.NSS+1))
sw_route  = sw_route .add(zeros, fill_value=0)
sw_runoff = sw_runoff.add(zeros, fill_value=0)
sw_dv1    = sw_dv1   .add(zeros, fill_value=0)
sw_dv2    = sw_dv2   .add(zeros, fill_value=0)

sw_route = sw_route - sw_dv1 - sw_dv2

sw_route[:]  = np.where(sw_route .abs()<1e-5, 0.0, sw_route )
sw_runoff[:] = np.where(sw_runoff.abs()<1e-5, 0.0, sw_runoff)

lk = lk.set_index("Date", ).loc["1984-04":]
sw_route.loc["1984-04":, 106] += (lk["PCP"]-lk["EVAP"]-lk["Change in Storage"])

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
sfr.comments = f"# Streamflow Routing (SFR) Package Input File\n#  Created using {lspc_ver}\n"
sfr.write_package(r"05_Model\NVIHM_SWRCB_NWT_LSPC\InputFiles\SFR\SFR.sfr")

os.makedirs(rf"03_Analyse\{lspc_ver}_QA", exist_ok=True)
sw_route .to_csv(rf"03_Analyse\{lspc_ver}_QA\modflow_inflows_cfd.csv")
sw_runoff.to_csv(rf"03_Analyse\{lspc_ver}_QA\modflow_runoff_cfd.csv")

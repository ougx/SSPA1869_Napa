import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.style.use("ggplot")
# logo = plt.imread("Postproc/data/ss-papadopulos-and-associates-logo.png")

pw = pd.read_csv(r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\02_Incoming\20260210_LSPCoutput\Actual_PointSource_Withdrawals_CubicFeetperMonth.csv")
pw["date"] = pd.to_datetime(pw["date"])
pw = pw[pw.date>="2004-10-01"]

lspc0 = pd.read_csv(r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\02_Incoming\20260210_LSPCoutput\Streams_WaterBalanceParams_20260211-145317_m3.csv", parse_dates=True)
lspc0.DTTM = pd.to_datetime(lspc0.DTTM)
lspc0 = lspc0[lspc0.DTTM>="2004-10-01"]

lspc = lspc0[lspc0.SWSID.isin([10334, 10143, 10315])]
lspc["LSPC"] = lspc["RO"] / (.3048**3) / 43560
lspc.set_index(["SWSID", "DTTM"], inplace=True)

sws_order = pd.read_csv(r"Postproc\Data\sws_routing.csv", index_col=0).squeeze()
lspc0 = pd.merge(lspc0, pw, left_on=["SWSID", "DTTM"], right_on=["rchid", "date"])
lspc0["LSPC_Composite"] = (lspc0.AGWO + lspc0.SURO + lspc0.IFWO - lspc0.PointSource_Withdrawls_cubicFeetperMonth*.3048**3) / (.3048**3) / 43560
l1 = lspc0[lspc0.SWSID.isin(sws_order.loc[10334].values)].groupby("DTTM", as_index=False).LSPC_Composite.sum()
l1["SWSID"] = 10334
l2 = lspc0[lspc0.SWSID.isin(sws_order.loc[10143].values)].groupby("DTTM", as_index=False).LSPC_Composite.sum()
l2["SWSID"] = 10143
l3 = lspc0.groupby("DTTM", as_index=False).LSPC_Composite.sum()
l3["SWSID"] = 10315
lspcc = pd.concat([l1, l2, l3]).set_index(["SWSID", "DTTM"],)

obsflow = pd.read_csv(r"Postproc\Data\obsflow.csv", parse_dates=True)
obsflow.time = pd.to_datetime(obsflow.time)
obsflow = obsflow[(obsflow.time >= "2004-10-01")&(obsflow.time < "2023-10-01")]
obsflow["Observed"] = obsflow["obscfs"] * obsflow.time.dt.day * 86400 / 43560
obsflow.set_index(["MFName", "time"], inplace=True)

mf = pd.read_csv("Output_flow_monthly.csv")
mf["date"] = np.tile(pd.date_range("1984-04-01", "2023-09-30", freq="ME"), 4)
mf = mf[mf.date >= "2004-10-01"]
mf["MODFLOW"] = mf["OUTFLOW"] * mf["date"].dt.day / 43560
mf.set_index(["MFName", "date"], inplace=True)


with PdfPages('plotRunoff.pdf') as pdf:

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), )
    obsflow.loc["NapaR_Helena"].Observed.plot(ax=ax, color="k", linewidth=2)
    lspc.loc[10334].LSPC.plot(ax=ax, color="b", linewidth=1, label="LSPC_RO")
    mf.loc["NapaR_Helena"].MODFLOW.plot(ax=ax, color="r", linewidth=1.5, linestyle="--")
    ax.legend()
    ax.set(ylabel="Napa River Discharge at Helena (acrefeet)")
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), )
    obsflow.loc["NapaR_Helena"].Observed.plot(ax=ax, color="k", linewidth=2)
    lspcc.loc[10334].LSPC_Composite.plot(ax=ax, color="b", linewidth=1, label="LSPC_SURO+IFWO+AGWO-Withdrawal")
    mf.loc["NapaR_Helena"].MODFLOW.plot(ax=ax, color="r", linewidth=1.5, linestyle="--")
    ax.legend()
    ax.set(ylabel="Napa River Discharge at Helena (acrefeet)")
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), )
    obsflow.loc["NapaR_Helena"].Observed.plot(ax=ax, color="k", linewidth=2)
    lspc.loc[10334].LSPC.plot(ax=ax, color="g", linewidth=1, label="LSPC_RO")
    lspcc.loc[10334].LSPC_Composite.plot(ax=ax, color="b", linewidth=1, label="LSPC_SURO+IFWO+AGWO-Withdrawal")
    mf.loc["NapaR_Helena"].MODFLOW.plot(ax=ax, color="r", linewidth=1.5, linestyle="--")
    ax.legend()
    ax.set(ylabel="Napa River Discharge at Helena (acrefeet)")
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), )
    obsflow.loc["NapaR_Napa"].Observed.plot(ax=ax, color="k", linewidth=2)
    lspc.loc[10143].LSPC.plot(ax=ax, color="b", linewidth=1, label="LSPC_RO")
    mf.loc["NapaR_Napa"].MODFLOW.plot(ax=ax, color="r", linewidth=1.5, linestyle="--")
    ax.legend()
    ax.set(ylabel="Napa River Discharge at Napa (acrefeet)")
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), )
    obsflow.loc["NapaR_Napa"].Observed.plot(ax=ax, color="k", linewidth=2)
    lspcc.loc[10143].LSPC_Composite.plot(ax=ax, color="b", linewidth=1, label="LSPC_SURO+IFWO+AGWO-Withdrawal")
    mf.loc["NapaR_Napa"].MODFLOW.plot(ax=ax, color="r", linewidth=1.5, linestyle="--")
    ax.legend()
    ax.set(ylabel="Napa River Discharge at Napa (acrefeet)")
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), )
    obsflow.loc["NapaR_Napa"].Observed.plot(ax=ax, color="k", linewidth=2)
    lspc.loc[10143].LSPC.plot(ax=ax, color="g", linewidth=1, label="LSPC_RO")
    lspcc.loc[10143].LSPC_Composite.plot(ax=ax, color="b", linewidth=1, label="LSPC_SURO+IFWO+AGWO-Withdrawal")
    mf.loc["NapaR_Napa"].MODFLOW.plot(ax=ax, color="r", linewidth=1.5, linestyle="--")
    ax.legend()
    ax.set(ylabel="Napa River Discharge at Napa (acrefeet)")
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), )
    lspc.loc[10315].LSPC.plot(ax=ax, color="g", linewidth=1, label="LSPC_RO")
    lspcc.loc[10315].LSPC_Composite.plot(ax=ax, color="b", linewidth=1, label="LSPC_SURO+IFWO+AGWO-Withdrawal")
    mf.loc["NapaR_Outlet"].MODFLOW.plot(ax=ax, color="r", linewidth=1.5, linestyle="--")
    ax.legend()
    ax.set(ylabel="Napa River Discharge at Outlet (acrefeet)")
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), )
    obsflow.loc["NapaR_Helena"].Observed.cumsum().plot(ax=ax, color="k", linewidth=2)
    lspc.loc[10334].LSPC.cumsum().plot(ax=ax, color="b", linewidth=1, label="LSPC_RO")
    mf.loc["NapaR_Helena"].MODFLOW.cumsum().plot(ax=ax, color="r", linewidth=1.5, linestyle="--")
    ax.legend()
    ax.set(ylabel="Napa River Cumulative Discharge at Helena (acrefeet)")
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), )
    obsflow.loc["NapaR_Helena"].Observed.cumsum().plot(ax=ax, color="k", linewidth=2)
    lspcc.loc[10334].LSPC_Composite.cumsum().plot(ax=ax, color="b", linewidth=1, label="LSPC_SURO+IFWO+AGWO-Withdrawal")
    mf.loc["NapaR_Helena"].MODFLOW.cumsum().plot(ax=ax, color="r", linewidth=1.5, linestyle="--")
    ax.legend()
    ax.set(ylabel="Napa River Cumulative Discharge at Helena (acrefeet)")
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), )
    obsflow.loc["NapaR_Helena"].Observed.cumsum().plot(ax=ax, color="k", linewidth=2)
    lspc.loc[10334].LSPC.cumsum().plot(ax=ax, color="g", linewidth=1, label="LSPC_RO")
    lspcc.loc[10334].LSPC_Composite.cumsum().plot(ax=ax, color="b", linewidth=1, label="LSPC_SURO+IFWO+AGWO-Withdrawal")
    mf.loc["NapaR_Helena"].MODFLOW.cumsum().plot(ax=ax, color="r", linewidth=1.5, linestyle="--")
    ax.legend()
    ax.set(ylabel="Napa River Cumulative Discharge at Helena (acrefeet)")
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), )
    obsflow.loc["NapaR_Napa"].Observed.cumsum().plot(ax=ax, color="k", linewidth=2)
    lspc.loc[10143].LSPC.cumsum().plot(ax=ax, color="b", linewidth=1, label="LSPC_RO")
    mf.loc["NapaR_Napa"].MODFLOW.cumsum().plot(ax=ax, color="r", linewidth=1.5, linestyle="--")
    ax.legend()
    ax.set(ylabel="Napa River Cumulative Discharge at Napa (acrefeet)")
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), )
    obsflow.loc["NapaR_Napa"].Observed.cumsum().plot(ax=ax, color="k", linewidth=2)
    lspcc.loc[10143].LSPC_Composite.cumsum().plot(ax=ax, color="b", linewidth=1, label="LSPC_SURO+IFWO+AGWO-Withdrawal")
    mf.loc["NapaR_Napa"].MODFLOW.cumsum().plot(ax=ax, color="r", linewidth=1.5, linestyle="--")
    ax.legend()
    ax.set(ylabel="Napa River Cumulative Discharge at Napa (acrefeet)")
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), )
    obsflow.loc["NapaR_Napa"].Observed.cumsum().plot(ax=ax, color="k", linewidth=2)
    lspc.loc[10143].LSPC.cumsum().plot(ax=ax, color="g", linewidth=1, label="LSPC_RO")
    lspcc.loc[10143].LSPC_Composite.cumsum().plot(ax=ax, color="b", linewidth=1, label="LSPC_SURO+IFWO+AGWO-Withdrawal")
    mf.loc["NapaR_Napa"].MODFLOW.cumsum().plot(ax=ax, color="r", linewidth=1.5, linestyle="--")
    ax.legend()
    ax.set(ylabel="Napa River Cumulative Discharge at Napa (acrefeet)")
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), )
    lspc.loc[10315].LSPC.cumsum().plot(ax=ax, color="g", linewidth=1, label="LSPC_RO")
    lspcc.loc[10315].LSPC_Composite.cumsum().plot(ax=ax, color="b", linewidth=1, label="LSPC_SURO+IFWO+AGWO-Withdrawal")
    mf.loc["NapaR_Outlet"].MODFLOW.cumsum().plot(ax=ax, color="r", linewidth=1.5, linestyle="--")
    ax.legend()
    ax.set(ylabel="Napa River Cumulative Discharge at Outlet (acrefeet)")
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

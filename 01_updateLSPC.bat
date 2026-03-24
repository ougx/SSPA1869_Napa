
cd InputFiles\RCH
python _create_rch.py "c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\02_Incoming\20260306_LSPCoutput\GWI_20260306-121949.rch"

cd ..\SFR
python _create_sfr.py 20260306_LSPCoutput ^
  "c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\02_Incoming\20260306_LSPCoutput\Streams_WaterBalanceParams_20260306-142227_m3.csv" ^
  "c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\02_Incoming\20260306_LSPCoutput\Actual_PointSource_Withdrawals_CubicFeetperMonth.csv" ^
  "c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\02_Incoming\20260306_LSPCoutput\LakeBudget_ForSSPA.xlsx"

cd ..\..

pause
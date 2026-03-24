@echo off

echo.
set lst=Output.lst

rem preprocessing
cd Texture2Par
call _runT2P.bat

cd ..\InputFiles\GHB
call _create_ghb.bat

cd ..\SFR
call _create_sfr.bat

cd ..\..

rem run model
bin\MODFLOW-NWT_64.exe NVIHM.nam


rem flows
findstr ^
  /C:"   1  139   81    82     1" ^
  /C:"   1  267  127   127     5" ^
  /C:"   1  318  120   150    18" ^
  /C:"   1  318  120   157    20" ^
  Output_SFR.out > dump

echo MFName                 INFLOW     LEAKAGE      OUTFLOW   OVERLAND     PRECIP         ET        STAGE      DEPTH      WIDTH   CONDCTNC   GRADIENT> dump.out
bin\sed -E -e "s/   1  139   81    82     1/NapaR_Helena   /g" ^
           -e "s/   1  267  127   127     5/NapaR_Napa     /g" ^
           -e "s/   1  318  120   150    18/NapaR_Outlet   /g" ^
           -e "s/   1  318  120   157    20/BrownsVC_Outlet/g" ^
           dump >> dump.out

bin\paste -d "  " Postproc\times.dat dump.out > Output_flow.dat

findstr /C:"     2 " Output_flow.dat > dump1
findstr /C:"NapaR_Napa"   dump1 > dump2
findstr /C:"NapaR_Helena" dump1 >>dump2
bin\ArrayMath -d 948 1 -sc 6 -a dump2 - 1.1574074e-05 > Output_flow_cfs.dat
rem del dump*
goto :eof


rem export budget
findstr /R ^
  /C:"VOLUMETRIC BUDGET FOR ENTIRE MODEL AT END OF TIME STEP" ^
  /C:"=.* =.*[0-9]$" ^
  %lst% > Output_budget.lst

echo LAYER,ROW,COL,SEG,RCH,INFLOW,LEAKAGE,OUTFLOW,OVERLAND,PRECIP,ET,STAGE,DEPTH,WIDTH,CONDCTNC,GRADIENT> dump.out
findstr /R ^
  /C:"   1  318  120   157    20" ^
  dump | bin\sed -E -e "s/^\s+//g" -e "s/\s+/,/g"  >> dump.out
bin\paste -d "," Postproc\outlet_brownsvalley.csv Postproc\times.csv dump.out > Output_BrownsValleyCreek.csv

echo LAYER,ROW,COL,SEG,RCH,INFLOW,LEAKAGE,OUTFLOW,OVERLAND,PRECIP,ET,STAGE,DEPTH,WIDTH,CONDCTNC,GRADIENT> dump.out
findstr /R ^
  /C:"   1  318  120   150    18" ^
  dump | bin\sed -E -e "s/^\s+//g" -e "s/\s+/,/g"  >> dump.out
bin\paste -d "," Postproc\outlet_napa.csv Postproc\times.csv dump.out > Output_NapaRiver.csv

rem flows
echo LAYER,ROW,COL,SEG,RCH,INFLOW,LEAKAGE,OUTFLOW,OVERLAND,PRECIP,ET,STAGE,DEPTH,WIDTH,CONDCTNC,GRADIENT> dump.out
findstr /R ^
  /C:"   1  139   81    82     1" ^
  dump | bin\sed -E -e "s/^\s+//g" -e "s/\s+/,/g" >> dump.out
bin\paste -d "," Postproc\gage_helena.csv Postproc\times.csv dump.out> Output_gage_1145600_2.csv


rem flows
rem findstr /R ^
rem   /C:"   1  157   93    86     2" ^
rem   dump | sed -E -e "s/^\s+//g" -e "s/\s+/,/g" > dump.out
rem bin\paste -d "," Postproc\gage_helena.csv Postproc\times.csv dump.out > Output_gage_1145600_1.csv

del dump*

copy /Y /B Output_gage_1145800.csv Output_gage.csv
bin\tail -n +2 Output_gage_1145600_2.csv >> Output_gage.csv

bin\ArrayMath -d 948 11 -cn -rn -a Output_gage_1145800.csv - 1.0 --groupby sp mean | bin\sed -E -e "s/^\s+//g" -e "s/\s+$//g" -e "s/\s+/,/g" > Output_gage_1145800_monthly.csv
bin\ArrayMath -d 948 11 -cn -rn -a Output_gage_1145600_2.csv - 1.0 --groupby sp mean | bin\sed -E -e "s/^\s+//g" -e "s/\s+$//g" -e "s/\s+/,/g" > Output_gage_1145600_monthly.csv
bin\ArrayMath -d 948 11 -cn -rn -a Output_BrownsValleyCreek.csv - 1.0 --groupby sp mean | bin\sed -E -e "s/^\s+//g" -e "s/\s+$//g" -e "s/\s+/,/g" > Output_BrownsValleyCreek_monthly.csv
bin\ArrayMath -d 948 11 -cn -rn -a Output_NapaRiver.csv - 1.0 --groupby sp mean | bin\sed -E -e "s/^\s+//g" -e "s/\s+$//g" -e "s/\s+/,/g" > Output_NapaRiver_monthly.csv

copy /Y /B Output_gage_1145800_monthly.csv Output_flow_monthly.csv
bin\tail -n +2 Output_gage_1145600_monthly.csv >> Output_flow_monthly.csv
bin\tail -n +2 Output_BrownsValleyCreek_monthly.csv >> Output_flow_monthly.csv
bin\tail -n +2 Output_NapaRiver_monthly.csv >> Output_flow_monthly.csv

rem convert CFD to CFS
bin\ArrayMath -d 1896 1 -sr 1 -sc 11 -a Output_flow_monthly.csv - 1.1574074e-05 > Output_flow_cfs.dat

set lst=Output.lst
bin\MODFLOW-NWT_64.exe NVIHM.nam

rem export budget
findstr /R ^
  /C:"VOLUMETRIC BUDGET FOR ENTIRE MODEL AT END OF TIME STEP" ^
  /C:"=[0-9 ]*\.[0-9][0-9][0-9][0-9]   *[A-Z].*=" ^
  %lst% > Output_budget.lst


rem flows
findstr /R ^
  /C:"   1  267  127   127     5" ^
  /C:"   1  139   81    82     1" ^
  /C:"   1  157   93    86     2" ^
  Output_SFR.out > dump

echo LAYER,ROW,COL,SEG,RCH,INFLOW,LEAKAGE,OUTFLOW,OVERLAND,PRECIP,ET,STAGE,DEPTH,WIDTH,CONDCTNC,GRADIENT> dump.out
findstr /R ^
  /C:"   1  267  127   127     5" ^
  dump | sed -E -e "s/^\s+//g" -e "s/\s+/,/g"  >> dump.out
bin\paste -d "," Postproc\gage_napa.csv Postproc\times.csv dump.out > Output_gage_1145800.csv

rem flows
echo LAYER,ROW,COL,SEG,RCH,INFLOW,LEAKAGE,OUTFLOW,OVERLAND,PRECIP,ET,STAGE,DEPTH,WIDTH,CONDCTNC,GRADIENT> dump.out
findstr /R ^
  /C:"   1  139   81    82     1" ^
  dump | sed -E -e "s/^\s+//g" -e "s/\s+/,/g" >> dump.out
bin\paste -d "," Postproc\gage_helena.csv Postproc\times.csv dump.out> Output_gage_1145600_2.csv


rem flows
rem findstr /R ^
rem   /C:"   1  157   93    86     2" ^
rem   dump | sed -E -e "s/^\s+//g" -e "s/\s+/,/g" > dump.out
rem bin\paste -d "," Postproc\gage_helena.csv Postproc\times.csv dump.out > Output_gage_1145600_1.csv

del dump*

copy /Y /B Output_gage_1145800.csv Output_gage.csv
bin\tail -n +2 Output_gage_1145600_2.csv >> Output_gage.csv

bin\ArrayMath -d 948 11 -cn -rn -a Output_gage_1145800.csv - 1.0 --groupby sp mean | sed -E -e "s/^\s+//g" -e "s/\s+$//g" -e "s/\s+/,/g" > Output_gage_1145800_monthly.csv
bin\ArrayMath -d 948 11 -cn -rn -a Output_gage_1145600_2.csv - 1.0 --groupby sp mean | sed -E -e "s/^\s+//g" -e "s/\s+$//g" -e "s/\s+/,/g" > Output_gage_1145600_monthly.csv

copy /Y /B Output_gage_1145800_monthly.csv Output_gage_monthly.csv
bin\tail -n +2 Output_gage_1145600_monthly.csv >> Output_gage_monthly.csv
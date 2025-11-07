set lst=Output.lst

findstr /R ^
  /C:"VOLUMETRIC BUDGET FOR ENTIRE MODEL AT END OF TIME STEP" ^
  /C:"=[0-9 ]*\.[0-9][0-9][0-9][0-9]   *[A-Z].*=" ^
  %lst% > Output_budget.lst
 
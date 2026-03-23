@echo off
set "py=makePst.py"


:: check if python is available
where /q python
IF ERRORLEVEL 1 (
    ECHO The application is missing. Ensure it is installed and placed in your PATH.
	if exist "p:\MichaelOu\Miniconda3" set "conda=p:\MichaelOu\Miniconda3"	
	if exist "c:\Miniconda3"           set "conda=c:\Miniconda3"
	echo."%PATH%" | findstr /C:"%conda%"  >nul && (
		echo Found Python in %conda%
		python --version
	) || (
		echo Set up Python Env
		set "PATH=%conda%;%conda%\Library\mingw-w64\bin;%conda%\Library\usr\bin;%conda%\Library\bin;%conda%\Scripts;%conda%\bin;%PATH%"
		python --version
	)
) ELSE (
    ECHO Found Python. Let's go!
)


set xls=Calib2026_Napa.xlsm
set pst=Napa2026_00

copy /y %xls% %pst%.xlsm

echo creating "%pst%.pst" in "%%~nxI"
python "%py%" "%pst%".pst estimation ^
	--set_ctl_xls   "%pst%.xlsm,CONTROL" ^
	--add_pargp_xls "%pst%.xlsm,PARGP" ^
	--add_par_xls   "%pst%.xlsm,PAR_T2P"      ^
	--add_par_xls   "%pst%.xlsm,PAR_HK"       ^
	--add_par_xls   "%pst%.xlsm,PAR_VK"       ^
	--add_par_xls   "%pst%.xlsm,PAR_SS"       ^
	--add_par_xls   "%pst%.xlsm,PAR_SY"       ^
	--add_par_xls   "%pst%.xlsm,PAR_SFR"      ^
	--add_par_xls   "%pst%.xlsm,PAR_GHB"      ^
	--add_obs_xls   "%pst%.xlsm,OBS_HEAD"     ^
	--add_obs_xls   "%pst%.xlsm,OBS_FLOW"     ^
	--add_io_xls    "%pst%.xlsm,IO"           ^
	--add_pp_xls    "%pst%.xlsm,PPcntl"       ^
	--add_pp_xls    "%pst%.xlsm,PPies"        ^
	--add_comment   "Napa Model Calibration Ver 2026_00"   

pause


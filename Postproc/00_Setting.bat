@echo off
set conda=p:\MichaelOu\Miniconda3
if exist C:\Miniconda3 set conda=C:\Miniconda3
REM set up python
echo."%PATH%" | findstr /C:"%conda%"  >nul && (
    python --version
) || (
    echo Set up Python Env using "%conda%"
    set "PATH=%conda%;%conda%\Library\mingw-w64\bin;%conda%\Library\usr\bin;%conda%\Library\bin;%conda%\Scripts;%conda%\bin;%PATH%"
    python --version
)
::set chump=p:\MichaelOu\sspaPlot\sspa_20240110\dist\chumpCmd\chumpCmd.exe
set "chump=python ..\Postproc\chump\chumpCmd.py"

rem %chump% extract toml\et_emgt.toml
rem %chump% extract toml\et_frst.toml
rem %chump% extract toml\et_shrb.toml
rem 
rem %chump% stat    toml\pstres.toml
rem 
rem %chump% plot    toml\plotErrMap.toml
rem %chump% plot    toml\plotHydrograph.toml
rem %chump% plot    toml\plotStreamflow.toml
rem 
rem %chump% plot    toml\plotHeadContour_Flood.toml
rem %chump% plot    toml\plotScatterplot.toml
rem 
rem %chump% plot    toml\plotScatterplot.toml
rem 
rem %chump% plot    toml\plotET.toml
rem %chump% plot    toml\plotETMap.toml
rem 
rem python 90-WaterBudget.py
rem python 91-PlotOtherTargets.py
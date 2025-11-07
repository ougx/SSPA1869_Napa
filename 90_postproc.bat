@echo off

python Postproc\chump\chumpCmd.py stat Postproc\TOML\simhead.toml
python Postproc\chump\chumpCmd.py plot Postproc\TOML\plotHydrograph.toml
python Postproc\chump\chumpCmd.py plot Postproc\TOML\plotScatterplot.toml
python Postproc\chump\chumpCmd.py plot Postproc\TOML\plotHeadErrMap.toml
python Postproc\chump\chumpCmd.py plot Postproc\TOML\plotStreamflow.toml
python Postproc\chump\chumpCmd.py plot Postproc\TOML\plotEcdf.toml
pause
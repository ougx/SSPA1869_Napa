@echo off
echo .bat    >  exclude
echo .exe    >> exclude
echo .txt    >> exclude
echo .vscode >> exclude
echo .git    >> exclude
echo autotest>> exclude
echo docs    >> exclude
echo .spyproject >> exclude
echo __pycache__ >> exclude
echo 00_build_template >> exclude

xcopy /I /Y /S /EXCLUDE:exclude c:\Cloud\Dropbox\PythonScripts\sspaPlot\* .\
@echo off
setlocal

echo Creating conda env "pysr" with Python 3.11...
conda create -n pysr python=3.11 -y
if errorlevel 1 (
  echo Failed to create conda environment.
  exit /b 1
)

echo Activating environment...
call conda activate pysr
if errorlevel 1 (
  echo Failed to activate conda environment.
  exit /b 1
)

echo Installing PySR...
pip install pysr
if errorlevel 1 (
  echo Failed to install pysr.
  exit /b 1
)

echo Done. To run:
echo   conda activate pysr
echo   python momentum\Cai_exhaust.py

endlocal

@echo off
setlocal

:: Strategy 5: Use detected Conda path
:: From 'conda info --envs', carla env is at: C:\ProgramData\anaconda3\envs\carla

set "PYTHON_EXE=C:\ProgramData\anaconda3\envs\carla\python.exe"

if not exist "%PYTHON_EXE%" (
    echo "CRITICAL ERROR: Python not found at %PYTHON_EXE%"
    echo "Please check the path manually."
    exit /b 1
)

echo Found Python at: %PYTHON_EXE%

:: Reset PATH to minimal to fix "Environment variable too long"
set PATH=C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\

:: Execute
:: If the first argument ends with .py, run it as a script
:: Otherwise, run main_collection.py with all arguments
set FIRST_ARG=%1
if "%FIRST_ARG:~-3%"==".py" (
    "%PYTHON_EXE%" %*
) else (
    "%PYTHON_EXE%" d:\code\carla\occnetv3_data_generator\main_collection.py %*
)

endlocal

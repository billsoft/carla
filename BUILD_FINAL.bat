@echo off
setlocal EnableDelayedExpansion

echo ========================================================
echo CARLA FINAL BUILD SCRIPT (VS 2026 + /wd4723 FIX)
echo ========================================================

rem -- 1. SETUP ENVIRONMENT --
echo [1/4] Setting up environment for VS 2026...

rem Force specify VS 2026 (v18) path
set "VS_DEV_CMD=C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat"

if not exist "%VS_DEV_CMD%" (
    echo FATAL ERROR: VS 2026 environment script not found at:
    echo %VS_DEV_CMD%
    echo Please verify your VS 2026 installation.
    exit /b 1
)

echo Using VS Toolchain: %VS_DEV_CMD%
call "%VS_DEV_CMD%"

rem Verify cl.exe
where cl.exe >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: cl.exe not found! VS environment setup failed.
    exit /b 1
)

echo [2/4] Activating Conda environment...
call conda activate carla

rem Verify Python
for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable)"') do set PYTHON_EXE=%%i
echo Using Python: %PYTHON_EXE%
set PYTHON_ROOT=C:\Users\bills\.conda\envs\carla

cd /d D:\code\carla

rem -- 2. CONFIGURE CMAKE --
echo [3/4] Configuring CMake...
rem Clean old cache
if exist Build\CMakeCache.txt del /f Build\CMakeCache.txt

rem KEY FIX: Added /wd4723 to suppress "potential divide by 0" error in UE5 ChaosVehicles
set CXX_FLAGS=/W1 /wd4267 /wd4244 /wd4305 /wd4456 /wd4459 /wd4702 /wd4710 /wd4711 /wd4514 /wd5045 /wd4365 /wd4457 /wd4723 /D_CRT_SECURE_NO_WARNINGS
set C_FLAGS=/W1 /D_CRT_SECURE_NO_WARNINGS

cmake -G Ninja -S . -B Build ^
    --toolchain=CMake/Toolchain.cmake ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCARLA_UNREAL_ENGINE_PATH="D:\code\UnrealEngine5_carla" ^
    -DPython_ROOT_DIR="%PYTHON_ROOT%" ^
    -DPython3_ROOT_DIR="%PYTHON_ROOT%" ^
    -DPython_FIND_REGISTRY=NEVER ^
    -DPython_FIND_STRATEGY=LOCATION ^
    -DCMAKE_CXX_FLAGS="%CXX_FLAGS%" ^
    -DCMAKE_C_FLAGS="%C_FLAGS%" ^
    -DBUILD_SHARED_LIBS=OFF

if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed!
    exit /b %ERRORLEVEL%
)

rem -- 3. BUILD UNREAL EDITOR PLUGINS --
echo [4/4] Building CARLA Unreal Editor Plugins (target: carla-unreal-editor)...
cmake --build Build --target carla-unreal-editor

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    exit /b %ERRORLEVEL%
)

echo ========================================================
echo BUILD COMPLETED SUCCESSFULLY!
echo ========================================================
echo.
echo Verifying plugin files...
if exist "Unreal\CarlaUnreal\Plugins\Carla\Binaries\Win64" (
    echo SUCCESS: Plugin binaries found:
    dir /b "Unreal\CarlaUnreal\Plugins\Carla\Binaries\Win64\*.dll" 2>nul
) else (
    echo WARNING: Plugin binaries directory not found
)
echo.
echo ========================================================
echo NEXT STEPS:
echo ========================================================
echo.
echo 1. Start server:
echo    start_carla_server.bat
echo.
echo 2. Click Play button in Unreal Editor
echo.
echo 3. Test connection:
echo    conda activate carla
echo    python test_connection.py
echo.
pause

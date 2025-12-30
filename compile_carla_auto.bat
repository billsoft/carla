@echo off
setlocal

echo [INFO] Initializing Visual Studio 2022 Professional x64 environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"

echo [INFO] Setting up PATH for Ninja and CMake...
set "PATH=C:\Program Files\CMake\bin;C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"

echo [INFO] Activating Conda environment 'carla'...
call conda activate carla

echo [INFO] Configuring CMake...
cd /d D:\code\carla
cmake -G Ninja -S . -B Build ^
    --toolchain=%CD%\CMake\Toolchain.cmake ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DPython_ROOT_DIR="C:\ProgramData\anaconda3\envs\carla" ^
    -DPython3_ROOT_DIR="C:\ProgramData\anaconda3\envs\carla"

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CMake configuration failed!
    exit /b %ERRORLEVEL%
)

echo [INFO] Building CARLA Python API...
cmake --build Build --target carla-python-api-install

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build failed!
    exit /b %ERRORLEVEL%
)

echo [INFO] Verifying installation...
python -c "import carla; print('SUCCESS: CARLA version', carla.__version__)"

echo [INFO] Compilation finished successfully!
endlocal

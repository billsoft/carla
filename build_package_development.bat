@echo off
setlocal
REM ============================================================
REM  Package CARLA Development build (includes local fisheye RAW
REM  and narrow-FOV fixes) for remote GPU server deployment.
REM  Output: Build\Package\Carla-0.10.0-Win64-Development(.zip)
REM  NOTE: Reset PATH first - inheriting the shell's long PATH and
REM  then calling vcvars64 exceeds cmd's 8192-char line limit.
REM ============================================================
set "PATH=C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0;C:\Program Files\CMake\bin;C:\Program Files\Microsoft Visual Studio\18\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"

call "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat"
cd /d D:\code\carla

echo ==== [1/2] CMake configure (compression ON, zip output) ====
cmake -G Ninja -S . -B Build -DCMAKE_BUILD_TYPE=Release -DCARLA_UNREAL_PACKAGE_NO_COMPRESSION=OFF "-DCARLA_UNREAL_ENGINE_PATH=D:\code\UnrealEngine5_carla" "-DPython_ROOT_DIR=C:\ProgramData\anaconda3\envs\carla" "-DPython3_ROOT_DIR=C:\ProgramData\anaconda3\envs\carla" -DPython_FIND_REGISTRY=NEVER -DPython_FIND_STRATEGY=LOCATION
if errorlevel 1 (
  echo CONFIGURE_FAILED
  exit /b 1
)

echo ==== [2/2] Package carla-unreal-package-development ====
cmake --build Build --target carla-unreal-package-development
if errorlevel 1 (
  echo PACKAGE_FAILED
  exit /b 1
)

echo PACKAGE_SUCCESS
dir "D:\code\carla\Build\Package"

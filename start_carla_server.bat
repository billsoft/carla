@echo off
setlocal

echo ========================================================
echo STARTING CARLA SIMULATION SERVER
echo ========================================================

set CARLA_ROOT=D:\code\carla
set UE5_ROOT=D:\code\UnrealEngine5_carla

if not exist "%UE5_ROOT%\Engine\Binaries\Win64\UnrealEditor.exe" (
    echo ERROR: UnrealEditor.exe not found at %UE5_ROOT%\Engine\Binaries\Win64\UnrealEditor.exe
    echo Please make sure Unreal Engine 5 is built successfully.
    pause
    exit /b 1
)

echo Launching Unreal Editor with CARLA project...
echo Project: %CARLA_ROOT%\Unreal\CarlaUnreal\CarlaUnreal.uproject

"%UE5_ROOT%\Engine\Binaries\Win64\UnrealEditor.exe" "%CARLA_ROOT%\Unreal\CarlaUnreal\CarlaUnreal.uproject"

echo.
echo Editor launched. 
echo 1. Wait for shaders to compile (may take a while first time).
echo 2. Press the green 'Play' button in the editor to start the server.
echo 3. Once the server is running, you can run your Python scripts.
pause

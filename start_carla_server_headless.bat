@echo off
setlocal

echo ========================================================
echo STARTING CARLA SIMULATION SERVER (HEADLESS MODE)
echo ========================================================

set CARLA_ROOT=D:\code\carla
set UE5_ROOT=D:\code\UnrealEngine5_carla

if not exist "%UE5_ROOT%\Engine\Binaries\Win64\UnrealEditor.exe" (
    echo ERROR: UnrealEditor.exe not found at %UE5_ROOT%\Engine\Binaries\Win64\UnrealEditor.exe
    echo Please make sure Unreal Engine 5 is built successfully.
    pause
    exit /b 1
)

echo Launching CARLA Server in Headless Mode...
echo Project: %CARLA_ROOT%\Unreal\CarlaUnreal\CarlaUnreal.uproject
echo Quality: Epic (High)

rem The command below starts the server directly. No GUI will appear.
rem -carla-server: Starts the server role.
rem -quality-level=Epic: Sets the highest graphical quality for powerful GPUs.

"%UE5_ROOT%\Engine\Binaries\Win64\UnrealEditor.exe" "%CARLA_ROOT%\Unreal\CarlaUnreal\CarlaUnreal.uproject" -carla-server -quality-level=Epic

echo.
echo CARLA server is running in the background.
echo You can now run your Python scripts to connect to it.
echo Press Ctrl+C in this window to stop the server.

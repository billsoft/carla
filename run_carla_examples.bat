@echo off
setlocal enabledelayedexpansion
echo ========================================================
echo CARLA EXAMPLE SCRIPTS LAUNCHER
echo ========================================================
echo.
echo IMPORTANT: Make sure before running any example:
echo 1. Unreal Editor is running
echo 2. You clicked the Play button (green triangle)
echo 3. Scene is loaded and running (you see FPS in top-left)
echo.
pause

call conda activate carla
cd /d d:\code\carla\PythonAPI\examples

echo.
echo ========================================================
echo Available Examples:
echo ========================================================
echo.
echo 1. Manual Control - Drive with keyboard (RECOMMENDED)
echo 2. Generate Traffic - Spawn 30 vehicles + 10 pedestrians
echo 3. Generate Heavy Traffic - Spawn 80 vehicles + 30 pedestrians
echo 4. Automatic Control - Watch AI drive
echo 5. Sensor Visualization - Camera/Lidar display
echo 6. Vehicle Gallery - Browse all vehicles
echo 7. Test Connection - Simple connection test
echo.
set /p choice="Select example (1-7): "

if "%choice%"=="1" (
    echo.
    echo ========================================================
    echo MANUAL CONTROL
    echo ========================================================
    echo Controls:
    echo   WASD - Drive
    echo   Space - Brake
    echo   Q - Reverse
    echo   P - Toggle Autopilot
    echo   Tab - Change camera
    echo   ESC - Quit
    echo.
    python manual_control.py
) else if "%choice%"=="2" (
    echo.
    echo ========================================================
    echo GENERATING TRAFFIC
    echo ========================================================
    echo Spawning 30 vehicles and 10 pedestrians...
    echo.
    python generate_traffic.py -n 30 -w 10
) else if "%choice%"=="3" (
    echo.
    echo ========================================================
    echo GENERATING HEAVY TRAFFIC
    echo ========================================================
    echo Spawning 80 vehicles and 30 pedestrians...
    echo This may take a while and affect performance!
    echo.
    python generate_traffic.py -n 80 -w 30
) else if "%choice%"=="4" (
    echo.
    echo ========================================================
    echo AUTOMATIC CONTROL
    echo ========================================================
    echo Watch the AI drive a vehicle
    echo Press Ctrl+C to stop
    echo.
    python automatic_control.py
) else if "%choice%"=="5" (
    echo.
    echo ========================================================
    echo SENSOR VISUALIZATION
    echo ========================================================
    echo Visualizing camera and sensor data
    echo.
    python sensor_synchronization.py
) else if "%choice%"=="6" (
    echo.
    echo ========================================================
    echo VEHICLE GALLERY
    echo ========================================================
    echo Browse all available vehicles
    echo Use arrow keys to navigate
    echo.
    python vehicle_gallery.py
) else if "%choice%"=="7" (
    echo.
    echo ========================================================
    echo CONNECTION TEST
    echo ========================================================
    cd /d d:\code\carla
    python test_connection.py
) else (
    echo Invalid choice!
    goto :end
)

:end
echo.
echo ========================================================
echo Script finished
echo ========================================================
echo.
echo Press Ctrl+C to terminate background processes if needed
pause

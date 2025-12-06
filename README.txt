========================================================
CARLA BUILD GUIDE (VS 2026)
========================================================

This guide describes how to build CARLA using Visual Studio 2026.
We use a specific compiler flag (/wd4723) to fix compatibility issues with UE5 ChaosVehicles.

PREREQUISITES
-------------
1. Visual Studio 2026 (v18) installed.
2. Anaconda environment 'carla' created.
3. Unreal Engine 5 source code compiled.

FILES
-----
- BUILD_FINAL.bat         : The ONLY script you need to run to build everything.
- start_carla_server.bat  : Launch the editor (server) after build.
- test_connection.py      : Test if Python client can connect to server.

INSTRUCTIONS
------------

STEP 1: Open VS Command Prompt
   - Press Windows Key
   - Search for "x64 Native Tools Command Prompt for VS 18" (or VS 2026)
   - Open it.

STEP 2: Run Build Script
   - In the command prompt, navigate to D:\code\carla
   - Run:
     BUILD_FINAL.bat

   This script will:
   1. Load VS 2026 environment.
   2. Activate 'carla' conda env.
   3. Configure CMake with /wd4723 flag.
   4. Build the editor plugins.
   5. Launch Unreal Editor automatically upon success.

STEP 3: Start Server
   - When Unreal Editor opens, wait for shaders to compile.
   - Click the green "Play" button (top toolbar).

STEP 4: Test Connection
   - Open a NEW command prompt (cmd or powershell).
   - Run:
     conda activate carla
     python test_connection.py

   - If you see "Connection Successful!", you are done!

TROUBLESHOOTING
---------------
- If build fails with "cl.exe not found", make sure you have VS 2026 installed at:
  C:\Program Files\Microsoft Visual Studio\18\Professional
- If build fails with "divide by zero" (C4723), ensure BUILD_FINAL.bat has /wd4723 in CXX_FLAGS.

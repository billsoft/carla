I have verified the system environment and configuration files as per `D:\code\carla\BUILD_FINAL.md`.
- **CMake**: 4.2.0 (Verified)
- **Python**: 3.10.19 (Verified)
- **VS2022**: `vcvarsall.bat` found at expected location.
- **DefaultGame.ini**: Configuration matches the requirements (Maps, Blueprints, UFS settings).

I am ready to proceed with the build process.

## 🚀 Execution Plan

### Phase 1: Environment Setup & Configuration
1.  Initialize Visual Studio 2022 x64 environment variables.
2.  Configure CMake with `Development` build type (to include logs for debugging).
    - Command: `cmake -G Ninja -S . -B Build --toolchain=%CD%\CMake\Toolchain.cmake -DCMAKE_BUILD_TYPE=Development`

### Phase 2: Compilation & Packaging
1.  Execute the packaging command.
    - Command: `cmake --build Build --target package-development`
    - *Note: This process may take 30-60 minutes.*

### Phase 3: Error Handling (Adaptive)
-   **If `UE_USE_VERSE_PATHS` error occurs**: I will pull the latest `ue5` branch changes as documented.
-   **If Weather blueprints are missing**: I will clean the cooked cache and re-run the build.
-   **Other errors**: I will investigate logs and apply fixes based on documentation and web search.

### Phase 4: Deployment & Verification
1.  Unzip the generated package to `D:\code\carla\server_new`.
2.  Backup the existing server directory.
3.  Deploy the new server files.
4.  Verify critical files (Executables, Paks, Scripts) exist.

Please confirm to start the build process.
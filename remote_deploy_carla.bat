@echo off
REM Upload local Development zip to server, replace, restart.
REM Zip source: D:\code\carla\Build\Package\Carla-0.10.0-Win64-Development.zip
REM Override with: remote_deploy_carla.bat C:\path\to\other.zip
setlocal
set "ZIPARG="
if not "%~1"=="" set "ZIPARG=-ZipPath %~1"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0remote\carla_remote.ps1" -Action deploy %ZIPARG%
pause

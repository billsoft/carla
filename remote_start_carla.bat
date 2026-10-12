@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0remote\carla_remote.ps1" -Action start
pause

@echo off
rem Temporary transfer script: stream local stage dir to 81 via SSH tar pipe
tar -cf - -C "D:\code\carla\Build\Package\Carla-0.10.0-Win64-Development" . | ssh -o BatchMode=yes Administrator@192.168.100.81 "tar -xf - -C D:/carla-server"
echo TRANSFER_DONE exit=%ERRORLEVEL%

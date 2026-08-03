@echo off
cd /d "%~dp0"
py obd_log_analyzer.py
if errorlevel 1 pause

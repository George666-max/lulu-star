@echo off
cd /d "%~dp0"
E:\miniconda3\python.exe catch_the_stars.py
if errorlevel 1 pause

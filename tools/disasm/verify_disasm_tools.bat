@echo off
chcp 65001
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0verify_disasm_tools.ps1"
pause

@echo off
chcp 65001
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_disasm_tools.ps1"
pause

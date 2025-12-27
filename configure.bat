@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if "%BEP_BUILDSYSTEM_PATH%"=="" (
  set "BUILDSYSTEM_DIR=%SCRIPT_DIR%third_party\BuildSystem"
) else (
  set "BUILDSYSTEM_DIR=%BEP_BUILDSYSTEM_PATH%"
)
set "INTERFACE=%BUILDSYSTEM_DIR%\scripts\interface.bat"

if not exist "%INTERFACE%" (
  echo [BEP][ERROR] BuildSystem interface not found: %INTERFACE%
  exit /b 1
)

call "%INTERFACE%" "%SCRIPT_DIR%buildsystem.xml" "%SCRIPT_DIR%"

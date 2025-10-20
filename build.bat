@echo off
REM Build script for CELM (Windows)
REM Usage: build.bat [target]
REM Targets: build (default), test, install, clean, distclean, uninstall

setlocal

set "BUILD_DIR=build"
set "TARGET=%~1"

if "%TARGET%"=="" set "TARGET=build"

REM Check if build directory exists
if not exist "%BUILD_DIR%" (
    if "%TARGET%"=="distclean" (
        echo Build directory does not exist, nothing to clean.
        exit /b 0
    )
    echo Error: Build directory not found. Run configure.bat first.
    exit /b 1
)

REM Execute the requested target
if /i "%TARGET%"=="build" goto :build
if /i "%TARGET%"=="test" goto :test
if /i "%TARGET%"=="install" goto :install
if /i "%TARGET%"=="clean" goto :clean
if /i "%TARGET%"=="distclean" goto :distclean
if /i "%TARGET%"=="uninstall" goto :uninstall
if /i "%TARGET%"=="help" goto :help

echo Unknown target: %TARGET%
echo Use "build.bat help" for usage information
exit /b 1

:build
echo Building CELM...
cmake --build "%BUILD_DIR%" --config Release -j
exit /b %errorlevel%

:test
echo Running tests...
ctest --test-dir "%BUILD_DIR%" --output-on-failure -C Release
exit /b %errorlevel%

:install
echo Installing CELM...
call :build
if %errorlevel% neq 0 exit /b %errorlevel%
cmake --install "%BUILD_DIR%" --config Release
exit /b %errorlevel%

:clean
echo Cleaning build artifacts...
cmake --build "%BUILD_DIR%" --target clean
exit /b %errorlevel%

:distclean
echo Removing build directory...
if exist "%BUILD_DIR%" (
    rmdir /s /q "%BUILD_DIR%"
    echo Build directory removed.
) else (
    echo Build directory does not exist.
)
exit /b 0

:uninstall
if not exist "%BUILD_DIR%\install_manifest.txt" (
    echo Error: install_manifest.txt not found. Cannot uninstall.
    exit /b 1
)
echo Uninstalling CELM...
for /f "delims=" %%F in (%BUILD_DIR%\install_manifest.txt) do (
    if exist "%%F" (
        echo Removing: %%F
        del /f "%%F"
    )
)
echo Uninstall complete.
exit /b 0

:help
echo CELM Build System (Windows^)
echo.
echo Usage:
echo   build.bat [target]
echo.
echo Targets:
echo   build       Build the project (default^)
echo   test        Run tests
echo   install     Install the binary
echo   clean       Clean build artifacts
echo   distclean   Remove build directory completely
echo   uninstall   Uninstall the binary
echo   help        Display this help message
echo.
echo Before running build.bat, run configure.bat to set up the build.
exit /b 0

endlocal

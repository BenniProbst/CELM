@echo off
REM Configure script for CELM (Windows)
REM Usage: configure.bat [--prefix=PREFIX] [--build-type=TYPE]

setlocal enabledelayedexpansion

REM Default values
set "PREFIX=C:\Program Files\CELM"
set "BUILD_TYPE=Release"
set "BUILD_DIR=build"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :end_parse
if "%~1"=="--help" goto :show_help

echo %~1 | findstr /C:"--prefix=" >nul
if !errorlevel! equ 0 (
    set "PREFIX=%~1"
    set "PREFIX=!PREFIX:--prefix=!"
    shift
    goto :parse_args
)

echo %~1 | findstr /C:"--build-type=" >nul
if !errorlevel! equ 0 (
    set "BUILD_TYPE=%~1"
    set "BUILD_TYPE=!BUILD_TYPE:--build-type=!"
    shift
    goto :parse_args
)

echo Unknown option: %~1
echo Use --help for usage information
exit /b 1

:show_help
echo Usage: configure.bat [OPTIONS]
echo.
echo Options:
echo   --prefix=PREFIX       Installation prefix (default: C:\Program Files\CELM)
echo   --build-type=TYPE     Build type: Release, Debug, RelWithDebInfo (default: Release)
echo   --help                Display this help message
exit /b 0

:end_parse

echo === CELM Configuration ===
echo Prefix:     %PREFIX%
echo Build type: %BUILD_TYPE%
echo.

REM Check for CMake
where cmake >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: CMake is not installed or not in PATH
    echo Please install CMake 3.20 or higher from https://cmake.org/download/
    exit /b 1
)

for /f "tokens=3" %%i in ('cmake --version ^| findstr /C:"cmake version"') do set CMAKE_VERSION=%%i
echo Found CMake: %CMAKE_VERSION%

REM Check for C++ compiler (Visual Studio or other)
where cl.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo Found C++ compiler: MSVC (cl.exe^)
    set "GENERATOR=Visual Studio 17 2022"
    goto :compiler_found
)

where g++.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo Found C++ compiler: g++.exe
    set "GENERATOR=MinGW Makefiles"
    goto :compiler_found
)

where clang++.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo Found C++ compiler: clang++.exe
    set "GENERATOR=Ninja"
    goto :compiler_found
)

echo ERROR: No C++ compiler found
echo Please install Visual Studio, MinGW, or Clang
exit /b 1

:compiler_found

REM Create build directory
echo.
echo Creating build directory: %BUILD_DIR%
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

REM Run CMake configuration
echo.
echo Running CMake configuration...
cmake -S . -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_INSTALL_PREFIX="%PREFIX%"

if %errorlevel% equ 0 (
    echo.
    echo === Configuration successful ===
    echo.
    echo Next steps:
    echo   build.bat          # Build the project
    echo   build.bat test     # Run tests
    echo   build.bat install  # Install to %PREFIX%
    echo.
) else (
    echo.
    echo ERROR: CMake configuration failed
    exit /b 1
)

endlocal

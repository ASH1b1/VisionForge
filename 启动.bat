@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "KMP_DUPLICATE_LIB_OK=TRUE"
set "LOG=%~dp0launch.log"

> "%LOG%" echo ========================================
>>"%LOG%" echo   VisionForge
>>"%LOG%" echo   dir: %cd%
>>"%LOG%" echo   time: %DATE% %TIME%
>>"%LOG%" echo ========================================

echo ========================================
echo   VisionForge
echo ========================================

if not exist "src\main.py" (
    echo [ERROR] src\main.py not found
    >>"%LOG%" echo [ERROR] src\main.py not found
    pause
    exit /b 1
)

rem --- prevent duplicate: relative src\main.py + visionforge python ---
set "ALREADY="
for /f "delims=" %%P in ('powershell -NoProfile -Command "Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'python' -and $_.CommandLine -and ($_.CommandLine -match 'src[\\/]+main\.py') -and ($_.CommandLine -match 'visionforge') } | Select-Object -First 1 -ExpandProperty ProcessId"') do set "ALREADY=%%P"
if defined ALREADY (
    echo [INFO] already running PID=%ALREADY%, skip.
    >>"%LOG%" echo [INFO] already running PID=%ALREADY%
    exit /b 0
)

set "PY="
if defined CONDA_PREFIX if not "%CONDA_PREFIX%"=="" (
    if exist "%CONDA_PREFIX%\python.exe" (
        for %%I in ("%CONDA_PREFIX%") do if /I "%%~nxI"=="visionforge" set "PY=%CONDA_PREFIX%\python.exe"
    )
    if not defined PY if exist "%CONDA_PREFIX%\envs\visionforge\python.exe" set "PY=%CONDA_PREFIX%\envs\visionforge\python.exe"
)
if not defined PY if exist "%USERPROFILE%\anaconda3\envs\visionforge\python.exe" set "PY=%USERPROFILE%\anaconda3\envs\visionforge\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\envs\visionforge\python.exe" set "PY=%USERPROFILE%\miniconda3\envs\visionforge\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\anaconda3\envs\visionforge\python.exe" set "PY=%LOCALAPPDATA%\anaconda3\envs\visionforge\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\miniconda3\envs\visionforge\python.exe" set "PY=%LOCALAPPDATA%\miniconda3\envs\visionforge\python.exe"
if not defined PY if exist "C:\ProgramData\anaconda3\envs\visionforge\python.exe" set "PY=C:\ProgramData\anaconda3\envs\visionforge\python.exe"
if not defined PY (
    echo [ERROR] visionforge python.exe not found
    >>"%LOG%" echo [ERROR] python not found
    pause
    exit /b 1
)

echo [INFO] Python: %PY%
>>"%LOG%" echo [INFO] Python: %PY%
echo [INFO] launching...
>>"%LOG%" echo [INFO] launching

start "VisionForge" /D "%~dp0" "%PY%" src\main.py
>>"%LOG%" echo [INFO] started

endlocal
exit /b 0

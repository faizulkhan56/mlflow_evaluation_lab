@echo off
REM Setup script for Kaggle API credentials (Windows)
REM Usage: setup_kaggle.bat

echo Kaggle API Credentials Setup
echo ============================
echo.
echo This script will help you set up Kaggle API credentials.
echo You can get your API token from: https://www.kaggle.com/settings
echo.

REM Check if credentials are already set
if defined KAGGLE_USERNAME (
    if defined KAGGLE_KEY (
        echo Credentials are already set.
        echo Username: %KAGGLE_USERNAME%
        echo.
        set /p update="Do you want to update them? (y/n): "
        if /i not "%update%"=="y" (
            echo Keeping existing credentials.
            exit /b 0
        )
    )
)

REM Prompt for credentials
set /p username="Enter your Kaggle username: "
set /p key="Enter your Kaggle API key: "

REM Validate inputs
if "%username%"=="" (
    echo Error: Username cannot be empty
    exit /b 1
)
if "%key%"=="" (
    echo Error: API key cannot be empty
    exit /b 1
)

REM Set credentials
set KAGGLE_USERNAME=%username%
set KAGGLE_KEY=%key%

echo.
echo Credentials set successfully!
echo.
echo To make these permanent, add them to your system environment variables
echo or create a .env file with:
echo   KAGGLE_USERNAME=%username%
echo   KAGGLE_KEY=%key%
echo.
echo You can now run the lab scripts:
echo   python lab6_wine_quality_autologging.py
echo   python lab7_wine_quality_manual.py
echo.

pause


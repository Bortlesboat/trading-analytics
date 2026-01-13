@echo off
echo ========================================
echo    WEEKLY TRADING REVIEW
echo ========================================
echo.

REM Update this path to your project directory
cd /d "%~dp0"

echo Step 1: Running Weekly Review...
echo.
python weekly_review.py

echo.
echo ========================================
echo Review complete! Press any key to exit.
pause > nul

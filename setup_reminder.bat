@echo off
echo ========================================
echo Setting up Friday Trading Reminder
echo ========================================
echo.
echo This will create a Windows Task Scheduler task that
echo reminds you every Friday at 4:30 PM to do your weekly review.
echo.
echo NOTE: Update the path below to match your installation!
echo.
pause

REM Update this path to your project directory
set PROJECT_DIR=%~dp0

REM Create the scheduled task
schtasks /create /tn "Trading Weekly Review Reminder" /tr "python %PROJECT_DIR%friday_reminder.py" /sc weekly /d FRI /st 16:30 /f

if %errorlevel% == 0 (
    echo.
    echo ========================================
    echo SUCCESS! Reminder scheduled for:
    echo   Every Friday at 4:30 PM
    echo.
    echo To modify: Open Task Scheduler and find
    echo "Trading Weekly Review Reminder"
    echo ========================================
) else (
    echo.
    echo ERROR: Could not create task.
    echo Try running this as Administrator.
)

echo.
pause

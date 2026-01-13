"""
FRIDAY TRADING REVIEW REMINDER
==============================
This script sends a Windows desktop notification reminding you to:
1. Download your weekly trades from Fidelity
2. Run your weekly review

Set this up in Windows Task Scheduler to run every Friday at 5:00 PM.
"""

import subprocess
import sys
from datetime import datetime

def show_notification(title, message):
    """Show a Windows toast notification using PowerShell."""
    # PowerShell command to show a toast notification
    ps_script = f'''
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
    [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null

    $template = @"
    <toast duration="long">
        <visual>
            <binding template="ToastText02">
                <text id="1">{title}</text>
                <text id="2">{message}</text>
            </binding>
        </visual>
        <audio src="ms-winsoundevent:Notification.Default"/>
    </toast>
"@

    $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
    $xml.LoadXml($template)
    $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Trading Review").Show($toast)
    '''

    try:
        subprocess.run(["powershell", "-Command", ps_script], capture_output=True)
        return True
    except:
        return False


def show_simple_popup(title, message):
    """Fallback: Show a simple Windows message box."""
    ps_script = f'''
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show("{message}", "{title}", [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Information)
    '''
    try:
        subprocess.run(["powershell", "-Command", ps_script], capture_output=True)
        return True
    except:
        return False


if __name__ == "__main__":
    today = datetime.now()

    title = "Weekly Trading Review"
    message = f"""Time for your weekly review!

1. Go to Fidelity > Activity & Orders > History
2. Download this week's trades (CSV)
3. Save to Documents\\Trading folder
4. Run: python weekly_review.py

Your patterns: Monday PUTs good, Friday CALLs bad.
Next week is {"MID" if 11 <= (today.day + 3) <= 20 else "EARLY/LATE"} month."""

    # Try toast notification first, fall back to popup
    if not show_notification(title, message):
        show_simple_popup(title, message)

    print(f"Reminder sent at {today.strftime('%Y-%m-%d %H:%M')}")

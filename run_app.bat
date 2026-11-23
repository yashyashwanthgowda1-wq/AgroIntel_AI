@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: The project environment was not found.
    echo Expected: %~dp0.venv\Scripts\python.exe
    pause
    exit /b 1
)

echo Starting AgroIntel AI...
echo Keep this terminal open while using the application.
echo On a phone connected to the same Wi-Fi, open:
echo http://172.17.216.4:8501
echo.
".venv\Scripts\python.exe" -m streamlit run "app.py" --server.address 0.0.0.0 --server.port 8501

if errorlevel 1 (
    echo.
    echo AgroIntel AI stopped with an error.
    pause
)

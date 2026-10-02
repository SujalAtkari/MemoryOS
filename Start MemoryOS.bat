@echo off
setlocal

set "ROOT=%~dp0"

if not exist "%ROOT%backend\.venv\Scripts\python.exe" (
    echo MemoryOS backend environment was not found.
    echo Run the backend setup instructions in README.md first.
    pause
    exit /b 1
)

if not exist "%ROOT%frontend\node_modules\vite\bin\vite.js" (
    echo Frontend dependencies were not found.
    echo Open a terminal in the frontend folder and run: npm install
    pause
    exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
    echo Node.js and npm were not found. Install Node.js, then try again.
    pause
    exit /b 1
)

curl.exe -fsS http://127.0.0.1:8000/health >nul 2>&1
if errorlevel 1 (
    start "MemoryOS Backend" /D "%ROOT%backend" cmd /k "set PYTHONPATH=. && .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
)

curl.exe -fsS http://localhost:5173/ >nul 2>&1
if errorlevel 1 (
    start "MemoryOS Frontend" /D "%ROOT%frontend" cmd /k "npm run dev -- --host localhost --port 5173"
)

ping -n 4 127.0.0.1 >nul
start "" "http://localhost:5173"

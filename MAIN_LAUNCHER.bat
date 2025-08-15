@echo off
title YouTube AI Policy Detection System
color 0A

echo.
echo ================================================================================
echo                    YOUTUBE AI POLICY DETECTION SYSTEM
echo ================================================================================
echo.
echo [1] Start Backend Server
echo [2] Start Full System (Backend + Frontend)
echo [3] Train AI Model
echo [4] Test System
echo [5] Install Dependencies
echo [6] Exit
echo.
set /p choice="Choose an option (1-6): "

if "%choice%"=="1" goto start_backend
if "%choice%"=="2" goto start_full
if "%choice%"=="3" goto train_model
if "%choice%"=="4" goto test_system
if "%choice%"=="5" goto install_deps
if "%choice%"=="6" goto exit

:start_backend
echo.
echo Starting Backend Server...
cd backend
python main.py
goto end

:start_full
echo.
echo Starting Full System...
echo Starting Backend...
start /b cmd /c "cd backend && python main.py"
timeout /t 3
echo Starting Frontend...
cd frontend
npm start
goto end

:train_model
echo.
echo Starting AI Model Training...
cd backend
python comprehensive_realistic_training.py
goto end

:test_system
echo.
echo Running System Tests...
python test_simple_api.py
goto end

:install_deps
echo.
echo Installing Dependencies...
pip install -r backend/requirements.txt
cd frontend
npm install
goto end

:exit
echo Goodbye!
exit

:end
echo.
echo Process completed.
pause

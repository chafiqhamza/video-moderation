@echo off
title Système d'Analyse Vidéo YouTube IA
echo.
echo [94m====================================================[0m
echo [94m=  DEMARRAGE DU SYSTEME IA D'ANALYSE VIDEO YOUTUBE  =[0m
echo [94m====================================================[0m
echo.

REM Vérifier si le port 8000 est déjà utilisé
netstat -ano | findstr :8000 > nul
if %ERRORLEVEL% EQU 0 (
    echo [91mATTENTION: Le port 8000 est déjà utilisé. Arrêt de l'application existante...[0m
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do taskkill /F /PID %%a 2>nul
    timeout /t 2 /nobreak > nul
)

REM Vérifier si le port 3000 est déjà utilisé
netstat -ano | findstr :3000 > nul
if %ERRORLEVEL% EQU 0 (
    echo [91mATTENTION: Le port 3000 est déjà utilisé. Arrêt de l'application existante...[0m
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3000') do taskkill /F /PID %%a 2>nul
    timeout /t 2 /nobreak > nul
)

REM Vérifier les dépendances IA
echo [92mVérification des dépendances d'IA avancées...[0m
python -c "import cv2, numpy, librosa" >nul 2>&1
if %errorlevel% neq 0 (
    echo [93mInstallation des dépendances Python pour l'IA...[0m
    cd backend
    pip install -r requirements.txt
    echo [92mDépendances IA installées avec succès[0m
    cd ..
)

echo [92mVérification des dépendances Node...[0m
cd frontend
if not exist "node_modules" (
    echo [93mInstallation des dépendances Node.js...[0m
    npm install --silent
)
cd ..

REM Démarrer le backend
echo.
echo [94mDémarrage du backend IA...[0m
start "Backend IA YouTube" cmd /c "cd backend && python main.py"

REM Attendre que le backend démarre
echo [93mAttente du démarrage du backend...[0m
timeout /t 5 /nobreak > nul

REM Démarrer le frontend
echo.
echo [94mDémarrage du frontend...[0m
start "Frontend React" cmd /c "cd frontend && npm start"

REM Ouvrir automatiquement le navigateur après quelques secondes
echo [93mPréparation du navigateur...[0m
timeout /t 5 /nobreak > nul
start http://localhost:3000

echo.
echo [92m====================================================[0m
echo [92m=  SYSTEME IA D'ANALYSE VIDEO DEMARRÉ AVEC SUCCÈS   =[0m
echo [92m====================================================[0m
echo.
echo [96mAccès à l'application:[0m
echo  - Interface utilisateur: [97mhttp://localhost:3000[0m
echo  - API backend: [97mhttp://localhost:8000/docs[0m
echo.
echo [93mPour analyser une vidéo:[0m
echo  1. Accédez à l'interface utilisateur
echo  2. Cliquez sur "Analyser une vidéo"
echo  3. Téléversez votre fichier vidéo
echo  4. Attendez l'analyse complète (texte, image, audio)
echo  5. Consultez le rapport de conformité YouTube
echo.
echo [91mPour arrêter l'application complète:[0m
echo [91m 1. Fermez cette fenêtre principale (la fermeture arrêtera tous les processus)[0m
echo [91m 2. Ou appuyez sur une touche pour tout arrêter proprement[0m
echo.

REM Créer un raccourci sur le bureau si demandé
set /p CREATE_SHORTCUT="Voulez-vous créer un raccourci sur le bureau pour la prochaine fois? (O/N): "
if /i "%CREATE_SHORTCUT%"=="O" (
    echo [92mCréation du raccourci sur le bureau...[0m
    powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\Analyse IA Video YouTube.lnk'); $Shortcut.TargetPath = 'cmd.exe'; $Shortcut.Arguments = '/c \"\"' + '%~f0' + '\"\"'; $Shortcut.IconLocation = 'shell32.dll,46'; $Shortcut.WorkingDirectory = '%~dp0'; $Shortcut.Description = 'Système d''analyse vidéo IA pour YouTube'; $Shortcut.Save()"
    echo [92mRaccourci créé avec succès ![0m
)

echo [95mAppuyez sur une touche pour arrêter l'application...[0m
pause > nul

REM Arrêt des processus au moment de la fermeture
echo [93mArrêt de tous les processus...[0m
taskkill /F /FI "WINDOWTITLE eq Backend IA YouTube*" > nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Frontend React*" > nul 2>&1
timeout /t 1 /nobreak > nul
echo [92mTous les processus ont été arrêtés avec succès.[0m
timeout /t 2 /nobreak > nul

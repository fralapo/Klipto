@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║             ✂️  K L I P T O                                   ║
echo ║         AI YouTube Shorts Generator                          ║
echo ║                                                              ║
echo ║              Installazione Automatica                        ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

:: Verifica se eseguito come amministratore
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ⚠️  Alcuni componenti potrebbero richiedere privilegi di amministratore.
    echo    Se l'installazione fallisce, esegui come Amministratore.
    echo.
)

:: Variabili
set "VENV_DIR=venv"
set "PYTHON_MIN_VERSION=3.10"

:: ═══════════════════════════════════════════════════════════════
:: STEP 1: Verifica Python
:: ═══════════════════════════════════════════════════════════════
echo [1/7] 🐍 Verifica Python...

where python >nul 2>&1
if %errorLevel% neq 0 (
    echo.
    echo ❌ Python non trovato!
    echo.
    echo    Scarica Python da: https://www.python.org/downloads/
    echo    Durante l'installazione, seleziona "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

:: Verifica versione Python
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo    Trovato Python %PYTHON_VERSION%

:: Estrai major e minor version
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

if %PYTHON_MAJOR% LSS 3 (
    echo ❌ Richiesto Python 3.10 o superiore
    pause
    exit /b 1
)
if %PYTHON_MAJOR% EQU 3 if %PYTHON_MINOR% LSS 10 (
    echo ❌ Richiesto Python 3.10 o superiore
    pause
    exit /b 1
)

echo ✓ Python %PYTHON_VERSION% OK

:: ═══════════════════════════════════════════════════════════════
:: STEP 2: Verifica/Installa FFmpeg
:: ═══════════════════════════════════════════════════════════════
echo.
echo [2/7] 🎥 Verifica FFmpeg...

where ffmpeg >nul 2>&1
if %errorLevel% neq 0 (
    echo    FFmpeg non trovato. Tentativo di installazione...
    
    :: Prova con winget
    where winget >nul 2>&1
    if %errorLevel% equ 0 (
        echo    Installazione con winget...
        winget install --id=Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
        goto :check_ffmpeg
    )
    
    :: Prova con choco
    where choco >nul 2>&1
    if %errorLevel% equ 0 (
        echo    Installazione con Chocolatey...
        choco install ffmpeg -y
        goto :check_ffmpeg
    )
    
    :: Download manuale
    echo.
    echo ⚠️  Impossibile installare FFmpeg automaticamente.
    echo.
    echo    Opzioni:
    echo    1. Installa winget ^(Windows Package Manager^) da Microsoft Store
    echo    2. Installa Chocolatey: https://chocolatey.org/install
    echo    3. Scarica manualmente: https://www.gyan.dev/ffmpeg/builds/
    echo       - Scarica "ffmpeg-release-essentials.zip"
    echo       - Estrai in C:\ffmpeg
    echo       - Aggiungi C:\ffmpeg\bin al PATH di sistema
    echo.
    set /p CONTINUE="Vuoi continuare comunque? (s/n): "
    if /i "!CONTINUE!" neq "s" exit /b 1
    goto :skip_ffmpeg
)

:check_ffmpeg
where ffmpeg >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=3" %%v in ('ffmpeg -version 2^>^&1 ^| findstr /i "version"') do set FFMPEG_VERSION=%%v
    echo ✓ FFmpeg !FFMPEG_VERSION! OK
) else (
    echo ⚠️  FFmpeg installato ma potrebbe richiedere riavvio del terminale
)

:skip_ffmpeg

:: ═══════════════════════════════════════════════════════════════
:: STEP 3: Crea Virtual Environment
:: ═══════════════════════════════════════════════════════════════
echo.
echo [3/7] 📦 Creazione ambiente virtuale...

if exist "%VENV_DIR%" (
    echo    Ambiente virtuale esistente trovato.
    set /p RECREATE="Vuoi ricrearlo? (s/n): "
    if /i "!RECREATE!" equ "s" (
        echo    Rimozione ambiente esistente...
        rmdir /s /q "%VENV_DIR%"
    ) else (
        goto :activate_venv
    )
)

python -m venv "%VENV_DIR%"
if %errorLevel% neq 0 (
    echo ❌ Errore nella creazione del virtual environment
    pause
    exit /b 1
)
echo ✓ Ambiente virtuale creato

:activate_venv
:: ═══════════════════════════════════════════════════════════════
:: STEP 4: Attiva Virtual Environment
:: ═══════════════════════════════════════════════════════════════
echo.
echo [4/7] 🔌 Attivazione ambiente virtuale...

call "%VENV_DIR%\Scripts\activate.bat"
if %errorLevel% neq 0 (
    echo ❌ Errore nell'attivazione del virtual environment
    pause
    exit /b 1
)
echo ✓ Ambiente virtuale attivato

:: ═══════════════════════════════════════════════════════════════
:: STEP 5: Aggiorna pip
:: ═══════════════════════════════════════════════════════════════
echo.
echo [5/7] ⬆️  Aggiornamento pip...

python -m pip install --upgrade pip --quiet
echo ✓ pip aggiornato

:: ═══════════════════════════════════════════════════════════════
:: STEP 6: Installa dipendenze
:: ═══════════════════════════════════════════════════════════════
echo.
echo [6/7] 📚 Installazione dipendenze...

:: Crea requirements.txt se non esiste
if not exist "requirements.txt" (
    echo    Creazione requirements.txt...
    (
        echo # Klipto - Core dependencies
        echo yt-dlp^>=2024.1.0
        echo httpx^>=0.25.0
        echo python-dotenv^>=1.0.0
        echo openai^>=1.0.0
        echo.
        echo # Audio analysis ^(optional, for precision cutting^)
        echo librosa^>=0.10.0
        echo scipy^>=1.11.0
        echo webrtcvad^>=2.0.10
        echo numpy^>=1.24.0
    ) > requirements.txt
)

echo    Installazione dipendenze base...
pip install yt-dlp httpx python-dotenv openai --quiet
if %errorLevel% neq 0 (
    echo ❌ Errore nell'installazione delle dipendenze base
    pause
    exit /b 1
)

echo    Installazione dipendenze audio (precision mode)...
pip install numpy scipy --quiet
pip install librosa --quiet 2>nul
if %errorLevel% neq 0 (
    echo ⚠️  librosa non installato - precision mode limitato
)

pip install webrtcvad --quiet 2>nul
if %errorLevel% neq 0 (
    echo ⚠️  webrtcvad non installato - VAD non disponibile
)

echo ✓ Dipendenze installate

:: ═══════════════════════════════════════════════════════════════
:: STEP 7: Configurazione
:: ═══════════════════════════════════════════════════════════════
echo.
echo [7/7] ⚙️  Configurazione...

:: Crea cartelle
if not exist "data\downloads" mkdir "data\downloads"
if not exist "data\transcripts" mkdir "data\transcripts"
if not exist "data\analysis" mkdir "data\analysis"
if not exist "data\clips" mkdir "data\clips"
echo ✓ Cartelle create

:: Crea .env se non esiste
if not exist ".env" (
    echo.
    echo ╔══════════════════════════════════════════════════════════════╗
    echo ║  📝 CONFIGURAZIONE API KEYS                                  ║
    echo ╚══════════════════════════════════════════════════════════════╝
    echo.
    echo    Klipto necessita di due API keys per funzionare:
    echo.
    echo    🎙️  Deepgram ^(trascrizione audio^)
    echo       Registrati su: https://deepgram.com
    echo       Costo: ~$0.0043/minuto
    echo.
    echo    🤖 OpenRouter ^(analisi AI^)
    echo       Registrati su: https://openrouter.ai
    echo       Costo: ~$0.001/analisi
    echo.
    
    set /p DEEPGRAM_KEY="Inserisci Deepgram API Key (o premi Invio per dopo): "
    set /p OPENROUTER_KEY="Inserisci OpenRouter API Key (o premi Invio per dopo): "
    
    (
        echo # ═══════════════════════════════════════════════════════════════
        echo # KLIPTO - Configurazione API
        echo # ═══════════════════════════════════════════════════════════════
        echo.
        echo # Deepgram - Trascrizione audio
        echo # Ottieni la chiave da: https://console.deepgram.com/
        echo DEEPGRAM_API_KEY=!DEEPGRAM_KEY!
        echo.
        echo # OpenRouter - Analisi AI
        echo # Ottieni la chiave da: https://openrouter.ai/keys
        echo OPENROUTER_API_KEY=!OPENROUTER_KEY!
    ) > .env
    
    echo ✓ File .env creato
    
    if "!DEEPGRAM_KEY!" equ "" (
        echo.
        echo ⚠️  Ricorda di aggiungere le API keys nel file .env prima di usare Klipto
    )
) else (
    echo ✓ File .env esistente
)

:: ═══════════════════════════════════════════════════════════════
:: COMPLETATO
:: ═══════════════════════════════════════════════════════════════
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║          ✅ KLIPTO INSTALLATO CON SUCCESSO!                  ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo 📋 Prossimi passi:
echo.
echo    1. Configura le API keys in .env (se non fatto)
echo       notepad .env
echo.
echo    2. Avvia Klipto con:
echo       run.bat
echo.
echo    3. Oppure usa direttamente:
echo       %VENV_DIR%\Scripts\activate
echo       python src/main.py --help
echo.

:: Verifica installazione
echo.
echo 🔍 Verifica installazione:
python -c "import yt_dlp; print('   ✓ yt-dlp')" 2>nul || echo    ✗ yt-dlp
python -c "import httpx; print('   ✓ httpx')" 2>nul || echo    ✗ httpx
python -c "import dotenv; print('   ✓ python-dotenv')" 2>nul || echo    ✗ python-dotenv
python -c "import librosa; print('   ✓ librosa (precision mode)')" 2>nul || echo    ⚠ librosa (opzionale)
python -c "import webrtcvad; print('   ✓ webrtcvad (VAD)')" 2>nul || echo    ⚠ webrtcvad (opzionale)
where ffmpeg >nul 2>&1 && echo    ✓ ffmpeg || echo    ✗ ffmpeg

echo.
pause

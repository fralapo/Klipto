@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

title Klipto - AI YouTube Shorts Generator (v4.0)

:: Verifica installazione
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo ╔══════════════════════════════════════════════════════════════╗
    echo ║  ❌ KLIPTO NON INSTALLATO                                    ║
    echo ╚══════════════════════════════════════════════════════════════╝
    echo.
    echo    Esegui prima install.bat per installare Klipto
    echo.
    pause
    exit /b 1
)

:: Attiva venv
call venv\Scripts\activate.bat

:: Verifica .env
if not exist ".env" (
    echo.
    echo ⚠️  File .env non trovato!
    echo    Crea il file .env con le tue API keys.
    echo.
)

:: Menu principale
:menu
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║             ✂️  K L I P T O   v4.0                          ║
echo ║         AI YouTube Shorts Generator                          ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo  Cosa vuoi fare?
echo.
echo  ┌─── PRODUCTION PIPELINE (V4.0) ──────────────────────────────┐
echo  │ [1] 🎯 Full Process          Dl+Transcribe+Analyze+Cut      │
echo  │ [2] 🧪 Dry Run               Anteprima interattiva          │
echo  │ [3] 🏭 Batch Processing      Elabora cartella trascrizioni  │
echo  ├─── TOOLS ───────────────────────────────────────────────────┤
echo  │ [4] 📝 Solo Trascrizione     Estrai testo da URL/File       │
echo  │ [5] 🧠 Solo Analisi          Trova momenti virali           │
echo  │ [6] ✂️  Taglia Clips          Da analisi esistente           │
echo  ├─── SETTINGS ────────────────────────────────────────────────┤
echo  │ [7] 📊 Modelli LLM           Gestisci modelli AI            │
echo  │ [8] 🎙️  Modelli Deepgram      Gestisci modelli trascrizione  │
echo  │ [9] 💾 Cache                 Gestisci file temporanei       │
echo  │ [10]⚙️  Configurazione        Modifica API keys              │
echo  │ [11]📖 Aiuto                 Guida comandi                  │
echo  └─────────────────────────────────────────────────────────────┘
echo  │ [0] 🚪 Esci                                                  │
echo.
set /p CHOICE="Seleziona opzione (0-11): "

if "%CHOICE%"=="1" goto :process
if "%CHOICE%"=="2" goto :dryrun
if "%CHOICE%"=="3" goto :batch
if "%CHOICE%"=="4" goto :transcribe
if "%CHOICE%"=="5" goto :analyze
if "%CHOICE%"=="6" goto :cut
if "%CHOICE%"=="7" goto :models
if "%CHOICE%"=="8" goto :deepgram
if "%CHOICE%"=="9" goto :cache
if "%CHOICE%"=="10" goto :config
if "%CHOICE%"=="11" goto :help
if "%CHOICE%"=="0" goto :exit

echo Opzione non valida
timeout /t 2 >nul
goto :menu

:: ═══════════════════════════════════════════════════════════════
:: PROCESS - Full Pipeline (V4.0 Integration)
:: ═══════════════════════════════════════════════════════════════
:process
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║  🎯 FULL PIPELINE (Download -> Analyze -> Cut)               ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo  Sorgenti supportate:
echo  • YouTube: https://youtube.com/watch?v=VIDEO_ID
echo  • File locale: C:\Videos\mio_video.mp4
echo.
set /p SOURCE="🔗 URL o percorso file (o 'b' per tornare): "

if "%SOURCE%"=="" goto :menu
if /i "%SOURCE%"=="b" goto :menu

echo.
set /p MAX_CLIPS="📊 Max clips [default: 5]: "
if "%MAX_CLIPS%"=="" set MAX_CLIPS=5

echo.
echo 🤖 Modalità Analyzer:
echo    [1] Enhanced Multi-Agent (TOON v3.0)  *Consigliato*
echo    [2] Standard (Veloce)
set /p A_CHOICE="   Seleziona [default: 1]: "

set ANALYZER=multiagent
if "%A_CHOICE%"=="2" set ANALYZER=standard

echo.
echo 🚀 Avvio pipeline...
echo.
:: Usa main.py che ora integra la logica v4.0
python src/main.py process "%SOURCE%" --max-clips %MAX_CLIPS% --analyzer %ANALYZER% --debug

echo.
pause
goto :menu

:: ═══════════════════════════════════════════════════════════════
:: DRY RUN - Interactive Preview
:: ═══════════════════════════════════════════════════════════════
:dryrun
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║  🧪 DRY RUN - ANTEPRIMA INTERATTIVA                          ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo  Richiede un file di trascrizione esistente (.json).
echo  Per generarlo, usa l'opzione [4] Solo Trascrizione.
echo.
echo  📁 Trascrizioni disponibili:
echo  ─────────────────────────────────────────────────────────────────
dir /b data\transcripts\*_transcript.json 2>nul || echo     (nessuna trovata)
echo  ─────────────────────────────────────────────────────────────────
echo.
set /p TRANSCRIPT="📄 File trascrizione (o 'b' per tornare): "
if "%TRANSCRIPT%"=="" goto :menu
if /i "%TRANSCRIPT%"=="b" goto :menu

:: Verifica path completo
if not exist "%TRANSCRIPT%" (
    if exist "data\transcripts\%TRANSCRIPT%" (
        set "TRANSCRIPT=data\transcripts\%TRANSCRIPT%"
    )
)

echo.
python src/pipeline.py --transcript "%TRANSCRIPT%" --dry-run

echo.
pause
goto :menu

:: ═══════════════════════════════════════════════════════════════
:: BATCH - Processing multiplo
:: ═══════════════════════════════════════════════════════════════
:batch
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║  🏭 BATCH PROCESSING                                         ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo  Elabora automaticamente tutti i file .json in una cartella.
echo.
set /p PATTERN="📂 Pattern (es. data/transcripts/*.json): "
if "%PATTERN%"=="" set PATTERN=data/transcripts/*.json

echo.
echo 🚀 Avvio batch processing...
echo.
python src/pipeline.py --batch "%PATTERN%"

echo.
pause
goto :menu

:: ═══════════════════════════════════════════════════════════════
:: TRANSCRIBE
:: ═══════════════════════════════════════════════════════════════
:transcribe
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║  📝 TRASCRIZIONE                                             ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
set /p SOURCE="🔗 URL o percorso video: "
if "%SOURCE%"=="" goto :menu

set /p MODEL_CHOICE="🎙️ Modello (1=nova-2, 2=nova-3, 3=whisper): "
set MODEL=nova-2
if "%MODEL_CHOICE%"=="2" set MODEL=nova-3
if "%MODEL_CHOICE%"=="3" set MODEL=whisper-large

echo.
python src/main.py transcribe "%SOURCE%" --model %MODEL%
echo.
pause
goto :menu

:: ═══════════════════════════════════════════════════════════════
:: ANALYZE (Quick)
:: ═══════════════════════════════════════════════════════════════
:analyze
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║  🧠 ANALISI RAPIDA                                           ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
set /p SOURCE="🔗 URL o percorso video: "
if "%SOURCE%"=="" goto :menu
python src/main.py analyze "%SOURCE%"
echo.
pause
goto :menu

:: ═══════════════════════════════════════════════════════════════
:: CUT
:: ═══════════════════════════════════════════════════════════════
:cut
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║  ✂️  TAGLIO CLIPS                                             ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Usare src/main.py cut ...
set /p VIDEO="🎬 Percorso video: "
set /p ANALYSIS="📄 Percorso analisi: "
python src/main.py cut "%VIDEO%" --analysis "%ANALYSIS%"
echo.
pause
goto :menu

:: ═══════════════════════════════════════════════════════════════
:: MODELS
:: ═══════════════════════════════════════════════════════════════
:models
cls
python src/main.py models --interactive
pause
goto :menu

:: ═══════════════════════════════════════════════════════════════
:: DEEPGRAM MODELS
:: ═══════════════════════════════════════════════════════════════
:deepgram
cls
python src/main.py transcriber --interactive
pause
goto :menu


:: ═══════════════════════════════════════════════════════════════
:: CACHE
:: ═══════════════════════════════════════════════════════════════
:cache
cls
echo [1] Info  [2] Pulisci tutto
set /p C_ACT="Scelta: "
if "%C_ACT%"=="1" python src/main.py cache --info
if "%C_ACT%"=="2" python src/main.py cache --clear
pause
goto :menu

:: ═══════════════════════════════════════════════════════════════
:: CONFIG
:: ═══════════════════════════════════════════════════════════════
:config
cls
notepad .env
goto :menu

:: ═══════════════════════════════════════════════════════════════
:: HELP
:: ═══════════════════════════════════════════════════════════════
:help
cls
python src/main.py --help
pause
goto :menu

:: ═══════════════════════════════════════════════════════════════
:: EXIT
:: ═══════════════════════════════════════════════════════════════
:exit
echo.
echo 👋 Bye!
exit /b 0

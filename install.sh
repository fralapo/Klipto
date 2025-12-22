#!/bin/bash

# ═══════════════════════════════════════════════════════════════
# KLIPTO - Script di Installazione per Linux
# ═══════════════════════════════════════════════════════════════

# Colori
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Banner
clear
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                              ║${NC}"
echo -e "${CYAN}║             ${MAGENTA}✂️  K L I P T O${CYAN}                                   ║${NC}"
echo -e "${CYAN}║         ${NC}AI YouTube Shorts Generator${CYAN}                          ║${NC}"
echo -e "${CYAN}║                                                              ║${NC}"
echo -e "${CYAN}║              ${NC}Installazione Automatica${CYAN}                        ║${NC}"
echo -e "${CYAN}║                                                              ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Variabili
VENV_DIR="venv"
PYTHON_CMD=""

# Funzioni di utilità
print_status() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }
print_step() { echo -e "${CYAN}[$1]${NC} $2"; }

# ═══════════════════════════════════════════════════════════════
# STEP 1: Verifica Python
# ═══════════════════════════════════════════════════════════════
print_step "1/7" "🐍 Verifica Python..."

for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v $cmd &> /dev/null; then
        VERSION=$($cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        MAJOR=$(echo $VERSION | cut -d. -f1)
        MINOR=$(echo $VERSION | cut -d. -f2)
        
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    print_error "Python 3.10+ non trovato!"
    echo ""
    echo "   Installa Python con:"
    echo ""
    
    if [ -f /etc/debian_version ]; then
        echo -e "   ${CYAN}sudo apt update && sudo apt install python3.11 python3.11-venv python3-pip${NC}"
    elif [ -f /etc/redhat-release ]; then
        echo -e "   ${CYAN}sudo dnf install python3.11${NC}"
    elif [ -f /etc/arch-release ]; then
        echo -e "   ${CYAN}sudo pacman -S python${NC}"
    else
        echo "   Visita https://www.python.org/downloads/"
    fi
    exit 1
fi

print_status "Python $($PYTHON_CMD --version | cut -d' ' -f2) OK"

# ═══════════════════════════════════════════════════════════════
# STEP 2: Verifica/Installa FFmpeg
# ═══════════════════════════════════════════════════════════════
echo ""
print_step "2/7" "🎥 Verifica FFmpeg..."

if ! command -v ffmpeg &> /dev/null; then
    print_warning "FFmpeg non trovato. Installazione..."
    
    if command -v apt &> /dev/null; then
        sudo apt update && sudo apt install -y ffmpeg
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y ffmpeg
    elif command -v yum &> /dev/null; then
        sudo yum install -y ffmpeg
    elif command -v pacman &> /dev/null; then
        sudo pacman -S --noconfirm ffmpeg
    elif command -v zypper &> /dev/null; then
        sudo zypper install -y ffmpeg
    else
        print_error "Impossibile installare FFmpeg automaticamente"
        echo "   Installa manualmente FFmpeg per il tuo sistema"
        read -p "   Vuoi continuare comunque? (s/n): " CONTINUE
        if [ "$CONTINUE" != "s" ]; then
            exit 1
        fi
    fi
fi

if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version | head -n1 | cut -d' ' -f3)
    print_status "FFmpeg $FFMPEG_VERSION OK"
fi

# ═══════════════════════════════════════════════════════════════
# STEP 3: Installa dipendenze di sistema
# ═══════════════════════════════════════════════════════════════
echo ""
print_step "3/7" "🔧 Dipendenze di sistema..."

if command -v apt &> /dev/null; then
    sudo apt install -y libsndfile1 python3-dev 2>/dev/null
elif command -v dnf &> /dev/null; then
    sudo dnf install -y libsndfile python3-devel 2>/dev/null
fi

print_status "Dipendenze sistema OK"

# ═══════════════════════════════════════════════════════════════
# STEP 4: Crea Virtual Environment
# ═══════════════════════════════════════════════════════════════
echo ""
print_step "4/7" "📦 Creazione ambiente virtuale..."

if [ -d "$VENV_DIR" ]; then
    print_info "Ambiente virtuale esistente trovato."
    read -p "   Vuoi ricrearlo? (s/n): " RECREATE
    if [ "$RECREATE" = "s" ]; then
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_CMD -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        print_error "Errore nella creazione del virtual environment"
        echo "   Prova: sudo apt install python3-venv"
        exit 1
    fi
fi

print_status "Ambiente virtuale creato"

# ═══════════════════════════════════════════════════════════════
# STEP 5: Attiva e aggiorna pip
# ═══════════════════════════════════════════════════════════════
echo ""
print_step "5/7" "🔌 Attivazione ambiente virtuale..."

source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

print_status "Ambiente attivato e pip aggiornato"

# ═══════════════════════════════════════════════════════════════
# STEP 6: Installa dipendenze Python
# ═══════════════════════════════════════════════════════════════
echo ""
print_step "6/7" "📚 Installazione dipendenze Python..."

# Crea requirements.txt
cat > requirements.txt << 'EOF'
# Klipto - Core dependencies
yt-dlp>=2024.1.0
httpx>=0.25.0
python-dotenv>=1.0.0
openai>=1.0.0
colorama>=0.4.6

# Audio analysis (optional, for precision cutting)
librosa>=0.10.0
scipy>=1.11.0
webrtcvad>=2.0.10
numpy>=1.24.0
EOF

echo "   Installazione dipendenze base..."
pip install yt-dlp httpx python-dotenv openai --quiet

echo "   Installazione dipendenze audio (precision mode)..."
pip install numpy scipy --quiet

pip install librosa --quiet 2>/dev/null
if [ $? -ne 0 ]; then
    print_warning "librosa non installato completamente"
fi

pip install webrtcvad --quiet 2>/dev/null
if [ $? -ne 0 ]; then
    print_warning "webrtcvad non installato"
fi

print_status "Dipendenze installate"

# ═══════════════════════════════════════════════════════════════
# STEP 7: Configurazione
# ═══════════════════════════════════════════════════════════════
echo ""
print_step "7/7" "⚙️  Configurazione..."

# Crea cartelle
mkdir -p data/{downloads,transcripts,analysis,clips}
print_status "Cartelle create"

# Crea .env se non esiste
if [ ! -f ".env" ]; then
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  📝 CONFIGURAZIONE API KEYS                                  ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "   Klipto necessita di due API keys per funzionare:"
    echo ""
    echo -e "   ${MAGENTA}🎙️  Deepgram${NC} (trascrizione audio)"
    echo "       Registrati su: https://deepgram.com"
    echo ""
    echo -e "   ${MAGENTA}🤖 OpenRouter${NC} (analisi AI)"
    echo "       Registrati su: https://openrouter.ai"
    echo ""
    
    read -p "Inserisci Deepgram API Key (o premi Invio per dopo): " DEEPGRAM_KEY
    read -p "Inserisci OpenRouter API Key (o premi Invio per dopo): " OPENROUTER_KEY
    
    cat > .env << EOF
# ═══════════════════════════════════════════════════════════════
# KLIPTO - Configurazione API
# ═══════════════════════════════════════════════════════════════

# Deepgram - Trascrizione audio
# Ottieni la chiave da: https://console.deepgram.com/
DEEPGRAM_API_KEY=$DEEPGRAM_KEY

# OpenRouter - Analisi AI
# Ottieni la chiave da: https://openrouter.ai/keys
OPENROUTER_API_KEY=$OPENROUTER_KEY
EOF
    
    chmod 600 .env
    print_status "File .env creato"
    
    if [ -z "$DEEPGRAM_KEY" ]; then
        echo ""
        print_warning "Ricorda di aggiungere le API keys in .env"
    fi
else
    print_status "File .env esistente"
fi

# Rendi eseguibili gli script
chmod +x install.sh run.sh 2>/dev/null

# ═══════════════════════════════════════════════════════════════
# COMPLETATO
# ═══════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                              ║${NC}"
echo -e "${CYAN}║          ${GREEN}✅ KLIPTO INSTALLATO CON SUCCESSO!${CYAN}                  ║${NC}"
echo -e "${CYAN}║                                                              ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}📋 Prossimi passi:${NC}"
echo ""
echo "   1. Configura le API keys (se non fatto)"
echo -e "      ${CYAN}nano .env${NC}"
echo ""
echo "   2. Avvia Klipto"
echo -e "      ${CYAN}./run.sh${NC}"
echo ""
echo "   3. Oppure usa direttamente"
echo -e "      ${CYAN}source $VENV_DIR/bin/activate${NC}"
echo -e "      ${CYAN}python src/main.py --help${NC}"
echo ""

# Verifica installazione
echo -e "${BOLD}🔍 Verifica installazione:${NC}"
python -c "import yt_dlp; print('   ✓ yt-dlp')" 2>/dev/null || echo "   ✗ yt-dlp"
python -c "import httpx; print('   ✓ httpx')" 2>/dev/null || echo "   ✗ httpx"
python -c "import dotenv; print('   ✓ python-dotenv')" 2>/dev/null || echo "   ✗ python-dotenv"
python -c "import librosa; print('   ✓ librosa (precision)')" 2>/dev/null || echo "   ⚠ librosa (opzionale)"
python -c "import webrtcvad; print('   ✓ webrtcvad (VAD)')" 2>/dev/null || echo "   ⚠ webrtcvad (opzionale)"
command -v ffmpeg &> /dev/null && echo "   ✓ ffmpeg" || echo "   ✗ ffmpeg"
echo ""

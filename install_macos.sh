#!/bin/bash

# ═══════════════════════════════════════════════════════════════
# KLIPTO - Script di Installazione per macOS
# ═══════════════════════════════════════════════════════════════

# Colori
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'
BOLD='\033[1m'

clear
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                              ║${NC}"
echo -e "${CYAN}║             ${MAGENTA}✂️  K L I P T O${CYAN}                                   ║${NC}"
echo -e "${CYAN}║         ${NC}AI YouTube Shorts Generator${CYAN}                          ║${NC}"
echo -e "${CYAN}║                                                              ║${NC}"
echo -e "${CYAN}║              ${NC}Installazione macOS${CYAN}                             ║${NC}"
echo -e "${CYAN}║                                                              ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

VENV_DIR="venv"
PYTHON_CMD=""

print_status() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_step() { echo -e "${CYAN}[$1]${NC} $2"; }

# ═══════════════════════════════════════════════════════════════
# STEP 1: Verifica/Installa Homebrew
# ═══════════════════════════════════════════════════════════════
print_step "1/8" "🍺 Verifica Homebrew..."

if ! command -v brew &> /dev/null; then
    echo "   Homebrew non trovato. Installazione..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Aggiungi Homebrew al PATH per Apple Silicon
    if [ -f "/opt/homebrew/bin/brew" ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    fi
fi

if command -v brew &> /dev/null; then
    print_status "Homebrew OK"
else
    print_error "Homebrew non installato"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════
# STEP 2: Verifica/Installa Python
# ═══════════════════════════════════════════════════════════════
echo ""
print_step "2/8" "🐍 Verifica Python..."

for cmd in python3.12 python3.11 python3.10 python3; do
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
    echo "   Python 3.10+ non trovato. Installazione..."
    brew install python@3.11
    PYTHON_CMD="python3.11"
fi

print_status "Python $($PYTHON_CMD --version | cut -d' ' -f2) OK"

# ═══════════════════════════════════════════════════════════════
# STEP 3: Verifica/Installa FFmpeg
# ═══════════════════════════════════════════════════════════════
echo ""
print_step "3/8" "🎥 Verifica FFmpeg..."

if ! command -v ffmpeg &> /dev/null; then
    echo "   FFmpeg non trovato. Installazione..."
    brew install ffmpeg
fi

if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version | head -n1 | cut -d' ' -f3)
    print_status "FFmpeg $FFMPEG_VERSION OK"
fi

# ═══════════════════════════════════════════════════════════════
# STEP 4: Installa dipendenze di sistema
# ═══════════════════════════════════════════════════════════════
echo ""
print_step "4/8" "🔧 Dipendenze di sistema..."

brew list libsndfile &>/dev/null || brew install libsndfile
print_status "Dipendenze sistema OK"

# ═══════════════════════════════════════════════════════════════
# STEP 5: Crea Virtual Environment
# ═══════════════════════════════════════════════════════════════
echo ""
print_step "5/8" "📦 Creazione ambiente virtuale..."

if [ -d "$VENV_DIR" ]; then
    read -p "   Ambiente esistente. Ricreare? (s/n): " RECREATE
    if [ "$RECREATE" = "s" ]; then
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

print_status "Ambiente virtuale creato"

# ═══════════════════════════════════════════════════════════════
# STEP 6: Attiva e aggiorna pip
# ═══════════════════════════════════════════════════════════════
echo ""
print_step "6/8" "🔌 Attivazione ambiente..."

source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

print_status "Ambiente attivato"

# ═══════════════════════════════════════════════════════════════
# STEP 7: Installa dipendenze Python
# ═══════════════════════════════════════════════════════════════
echo ""
print_step "7/8" "📚 Installazione dipendenze..."

cat > requirements.txt << 'EOF'
# Klipto - Core dependencies
yt-dlp>=2024.1.0
httpx>=0.25.0
python-dotenv>=1.0.0
openai>=1.0.0
colorama>=0.4.6

# Audio analysis
librosa>=0.10.0
scipy>=1.11.0
numpy>=1.24.0
EOF

pip install yt-dlp httpx python-dotenv openai --quiet
pip install numpy scipy librosa --quiet

pip install webrtcvad --quiet 2>/dev/null
if [ $? -ne 0 ]; then
    print_warning "webrtcvad non installato (opzionale su macOS)"
fi

print_status "Dipendenze installate"

# ═══════════════════════════════════════════════════════════════
# STEP 8: Configurazione
# ═══════════════════════════════════════════════════════════════
echo ""
print_step "8/8" "⚙️  Configurazione..."

mkdir -p data/{downloads,transcripts,analysis,clips}

if [ ! -f ".env" ]; then
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  📝 CONFIGURAZIONE API KEYS                                  ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    read -p "Deepgram API Key (Invio per dopo): " DEEPGRAM_KEY
    read -p "OpenRouter API Key (Invio per dopo): " OPENROUTER_KEY
    
    cat > .env << EOF
# ═══════════════════════════════════════════════════════════════
# KLIPTO - Configurazione API
# ═══════════════════════════════════════════════════════════════

DEEPGRAM_API_KEY=$DEEPGRAM_KEY
OPENROUTER_API_KEY=$OPENROUTER_KEY
EOF
    chmod 600 .env
fi

chmod +x *.sh 2>/dev/null

print_status "Configurazione completata"

# ═══════════════════════════════════════════════════════════════
# COMPLETATO
# ═══════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          ${GREEN}✅ KLIPTO INSTALLATO CON SUCCESSO!${CYAN}                  ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}📋 Prossimi passi:${NC}"
echo ""
echo -e "   1. Configura API keys: ${CYAN}nano .env${NC}"
echo -e "   2. Avvia Klipto: ${CYAN}./run.sh${NC}"
echo ""

echo -e "${BOLD}🔍 Verifica:${NC}"
python -c "import yt_dlp; print('   ✓ yt-dlp')" 2>/dev/null || echo "   ✗ yt-dlp"
python -c "import librosa; print('   ✓ librosa')" 2>/dev/null || echo "   ⚠ librosa"
command -v ffmpeg &> /dev/null && echo "   ✓ ffmpeg" || echo "   ✗ ffmpeg"
echo ""

#!/bin/bash

# ═══════════════════════════════════════════════════════════════
# KLIPTO - Script di Avvio per Linux/macOS
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

# Verifica installazione
if [ ! -d "venv" ]; then
    echo ""
    echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ❌ KLIPTO NON INSTALLATO                                    ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "   Esegui prima: ./install.sh"
    echo ""
    exit 1
fi

# Attiva venv
source venv/bin/activate

# Verifica .env
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  File .env non trovato!${NC}"
    echo "   Esegui ./install.sh o crea manualmente il file .env"
fi

# Funzione menu
show_menu() {
    clear
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                                                              ║${NC}"
    echo -e "${CYAN}║             ${MAGENTA}✂️  K L I P T O   v4.0${CYAN}                          ║${NC}"
    echo -e "${CYAN}║         ${NC}AI YouTube Shorts Generator${CYAN}                          ║${NC}"
    echo -e "${CYAN}║                                                              ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo " Cosa vuoi fare?"
    echo ""
    echo -e " ${CYAN}┌─── PRODUCTION PIPELINE (V4.0) ──────────────────────────────┐${NC}"
    echo -e " ${CYAN}│${NC} [1] 🎯 Full Process          Dl+Transcribe+Analyze+Cut      ${CYAN}│${NC}"
    echo -e " ${CYAN}│${NC} [2] 🧪 Dry Run               Anteprima interattiva          ${CYAN}│${NC}"
    echo -e " ${CYAN}│${NC} [3] 🏭 Batch Processing      Elabora cartella trascrizioni  ${CYAN}│${NC}"
    echo -e " ${CYAN}├─── TOOLS ───────────────────────────────────────────────────┤${NC}"
    echo -e " ${CYAN}│${NC} [4] 📝 Solo Trascrizione     Estrai testo da URL/File       ${CYAN}│${NC}"
    echo -e " ${CYAN}│${NC} [5] 🧠 Solo Analisi          Trova momenti virali           ${CYAN}│${NC}"
    echo -e " ${CYAN}│${NC} [6] ✂️  Taglia Clips          Da analisi esistente           ${CYAN}│${NC}"
    echo -e " ${CYAN}├─── SETTINGS ────────────────────────────────────────────────┤${NC}"
    echo -e " ${CYAN}│${NC} [7] 📊 Modelli LLM           Gestisci modelli AI            ${CYAN}│${NC}"
    echo -e " ${CYAN}│${NC} [8] 🎙️  Modelli Deepgram      Gestisci modelli trascrizione  ${CYAN}│${NC}"
    echo -e " ${CYAN}│${NC} [9] 💾 Cache                 Gestisci file temporanei       ${CYAN}│${NC}"
    echo -e " ${CYAN}│${NC} [10]⚙️  Configurazione        Modifica API keys              ${CYAN}│${NC}"
    echo -e " ${CYAN}│${NC} [11]📖 Aiuto                 Guida comandi                  ${CYAN}│${NC}"
    echo -e " ${CYAN}└─────────────────────────────────────────────────────────────┘${NC}"
    echo -e " ${CYAN}│${NC} [0] 🚪 Esci                                                  ${CYAN}│${NC}"
    echo ""
}

# Funzione per pausa
pause() {
    echo ""
    read -p "Premi Invio per continuare..."
}

# Loop principale
while true; do
    show_menu
    read -p "Seleziona opzione (0-11): " choice
    
    case $choice in
        1)
            # ═══════════════════════════════════════════════════════════
            # FULL PROCESS
            # ═══════════════════════════════════════════════════════════
            clear
            echo ""
            echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
            echo -e "${CYAN}║  🎯 FULL PIPELINE (Download -> Analyze -> Cut)               ║${NC}"
            echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
            echo ""
            echo " Sorgenti supportate:"
            echo " • YouTube: https://youtube.com/watch?v=VIDEO_ID"
            echo " • File locale: /path/to/video.mp4"
            echo ""
            read -p "🔗 URL o percorso file (o 'b' per tornare): " SOURCE
            
            if [ -z "$SOURCE" ] || [ "$SOURCE" = "b" ]; then
                continue
            fi
            
            echo ""
            read -p "📊 Numero massimo clips [default: 5]: " MAX_CLIPS
            MAX_CLIPS=${MAX_CLIPS:-5}
            
            echo ""
            echo "🤖 Modalità Analyzer:"
            echo "   [1] Enhanced Multi-Agent (TOON v3.0)  *Consigliato*"
            echo "   [2] Standard (Veloce)"
            read -p "   Seleziona [default: 1]: " A_CHOICE
            
            ANALYZER="multiagent"
            if [ "$A_CHOICE" = "2" ]; then
                ANALYZER="standard"
            fi
            
            echo ""
            echo -e "${GREEN}🚀 Avvio pipeline...${NC}"
            echo ""
            
            python src/main.py process "$SOURCE" --max-clips $MAX_CLIPS --analyzer $ANALYZER --debug
            
            pause
            ;;
            
        2)
            # ═══════════════════════════════════════════════════════════
            # DRY RUN
            # ═══════════════════════════════════════════════════════════
            clear
            echo ""
            echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
            echo -e "${CYAN}║  🧪 DRY RUN - ANTEPRIMA INTERATTIVA                          ║${NC}"
            echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
            echo ""
            echo " Richiede un file di trascrizione esistente (.json)."
            echo ""
            echo " 📁 Trascrizioni disponibili:"
            echo " ─────────────────────────────────────────────────────────────────"
            ls -1 data/transcripts/*_transcript.json 2>/dev/null || echo "    (nessuna trovata)"
            echo " ─────────────────────────────────────────────────────────────────"
            echo ""
            
            read -p "📄 File trascrizione (o 'b' per tornare): " TRANSCRIPT
            
            if [ -z "$TRANSCRIPT" ] || [ "$TRANSCRIPT" = "b" ]; then
                continue
            fi
            
            echo ""
            python src/pipeline.py --transcript "$TRANSCRIPT" --dry-run
            
            pause
            ;;

        3)
            # ═══════════════════════════════════════════════════════════
            # BATCH
            # ═══════════════════════════════════════════════════════════
            clear
            echo ""
            echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
            echo -e "${CYAN}║  🏭 BATCH PROCESSING                                         ║${NC}"
            echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
            echo ""
            read -p "📂 Pattern [default: data/transcripts/*.json]: " PATTERN
            PATTERN=${PATTERN:-"data/transcripts/*.json"}
            
            echo ""
            python src/pipeline.py --batch "$PATTERN"
            
            pause
            ;;

        4)
            # ═══════════════════════════════════════════════════════════
            # TRASCRIZIONE
            # ═══════════════════════════════════════════════════════════
            clear
            echo ""
            echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
            echo -e "${CYAN}║  📝 KLIPTO - TRASCRIZIONE                                    ║${NC}"
            echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
            echo ""
            read -p "🔗 URL o percorso video (o 'b' per tornare): " SOURCE
            
            if [ -z "$SOURCE" ] || [ "$SOURCE" = "b" ]; then
                continue
            fi
            
            echo ""
            echo "🎙️ Modello Deepgram (default impostato):"
            echo "   Premi invio per default, oppure:"
            echo "   [1] nova-2"
            echo "   [2] nova-3"
            echo "   [3] nova-2-video"
            echo "   [4] whisper-large"
            read -p "   Seleziona: " MODEL_CHOICE
            
            MODEL_ARG=""
            case $MODEL_CHOICE in
                1) MODEL_ARG="--model nova-2" ;;
                2) MODEL_ARG="--model nova-3" ;;
                3) MODEL_ARG="--model nova-2-video" ;;
                4) MODEL_ARG="--model whisper-large" ;;
            esac
            
            echo ""
            python src/main.py transcribe "$SOURCE" $MODEL_ARG
            
            pause
            ;;
            
        5)
            # ═══════════════════════════════════════════════════════════
            # ANALISI
            # ═══════════════════════════════════════════════════════════
            clear
            echo ""
            echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
            echo -e "${CYAN}║  🧠 KLIPTO - ANALISI VIRALE                                  ║${NC}"
            echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
            echo ""
            read -p "🔗 URL o percorso video (o 'b' per tornare): " SOURCE
            
            if [ -z "$SOURCE" ] || [ "$SOURCE" = "b" ]; then
                continue
            fi
            
            read -p "📊 Numero massimo clips [default: 10]: " MAX_CLIPS
            MAX_CLIPS=${MAX_CLIPS:-10}
            
            echo ""
            python src/main.py analyze "$SOURCE" --max-clips $MAX_CLIPS
            
            pause
            ;;
            
        6)
            # ═══════════════════════════════════════════════════════════
            # TAGLIA CLIPS
            # ═══════════════════════════════════════════════════════════
            clear
            echo ""
            echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
            echo -e "${CYAN}║  ✂️  KLIPTO - TAGLIA CLIPS                                    ║${NC}"
            echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
            echo ""
            echo " 📁 File di analisi disponibili:"
            echo " ─────────────────────────────────────────────────────────────────"
            ls -1 data/analysis/*_analysis.json 2>/dev/null || echo "    (nessuno trovato)"
            echo " ─────────────────────────────────────────────────────────────────"
            echo ""
            
            echo "(Lascia vuoto o 'b' per tornare)"
            read -p "🎬 Percorso video: " VIDEO
            [ -z "$VIDEO" ] || [ "$VIDEO" = "b" ] && continue
            
            read -p "📄 Percorso file analisi: " ANALYSIS
            [ -z "$ANALYSIS" ] || [ "$ANALYSIS" = "b" ] && continue
            
            echo ""
            echo "⚡ Metodo di taglio:"
            echo "   [1] hybrid    (consigliato)"
            echo "   [2] accurate"
            echo "   [3] fast"
            read -p "   Seleziona [default: 1]: " METHOD_CHOICE
            
            METHOD="hybrid"
            case $METHOD_CHOICE in
                2) METHOD="accurate" ;;
                3) METHOD="fast" ;;
            esac
            
            echo ""
            python src/main.py cut "$VIDEO" --analysis "$ANALYSIS" --method $METHOD
            
            pause
            ;;
            
        7)
            # ═══════════════════════════════════════════════════════════
            # MODELLI LLM
            # ═══════════════════════════════════════════════════════════
            clear
            echo ""
            python src/main.py models --interactive
            pause
            ;;
            
        8)
            # ═══════════════════════════════════════════════════════════
            # MODELLI DEEPGRAM
            # ═══════════════════════════════════════════════════════════
            clear
            echo ""
            python src/main.py transcriber --interactive
            pause
            ;;

        9)
            # ═══════════════════════════════════════════════════════════
            # CACHE
            # ═══════════════════════════════════════════════════════════
            clear
            echo ""
            echo " [1] 📊 Info"
            echo " [2] 🗑️  Pulisci tutto"
            echo ""
            read -p "Seleziona: " CACHE_ACTION
            
            if [ "$CACHE_ACTION" = "1" ]; then
                python src/main.py cache --info
            elif [ "$CACHE_ACTION" = "2" ]; then
                python src/main.py cache --clear
            fi
            pause
            ;;
            
        10)
            # ═══════════════════════════════════════════════════════════
            # CONFIGURAZIONE
            # ═══════════════════════════════════════════════════════════
            clear
            echo ""
            echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
            echo -e "${CYAN}║  ⚙️  KLIPTO - CONFIGURAZIONE                                  ║${NC}"
            echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
            echo ""
            
            if [ ! -f ".env" ]; then
                cat > .env << 'EOF'
# KLIPTO - Configurazione API
DEEPGRAM_API_KEY=
OPENROUTER_API_KEY=
EOF
            fi
            
            read -p "Premi Invio per aprire l'editor..."
            
            if command -v nano &> /dev/null; then
                nano .env
            elif command -v vim &> /dev/null; then
                vim .env
            elif command -v vi &> /dev/null; then
                vi .env
            else
                echo "Nessun editor trovato. Modifica: .env"
                pause
            fi
            ;;
            
        11)
            # ═══════════════════════════════════════════════════════════
            # AIUTO
            # ═══════════════════════════════════════════════════════════
            clear
            echo ""
            python src/main.py --help
            echo ""
            pause
            ;;

        0)
            echo ""
            echo -e "${CYAN}👋 Bye!${NC}"
            echo ""
            exit 0
            ;;
            
        *)
            echo -e "${RED}Opzione non valida${NC}"
            sleep 1
            ;;
    esac
done

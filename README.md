# ✂️ Klipto - AI YouTube Shorts Generator 

**Genera automaticamente clip virali da video lunghi per TikTok, YouTube Shorts e Instagram Reels.**

![Klipto Banner](https://via.placeholder.com/800x200/1a1a2e/00d9ff?text=✂️+KLIPTO+v4.0)

---

## ✨ Nuove Funzionalità

### 🏭 Production Pipeline (`src/pipeline.py`)
Uno script unificato e robusto per l'uso in produzione:
- **Smart Caching**: Evita di ri-analizzare video già processati (MD5 hashing).
- **Batch Processing**: Elabora intere cartelle di trascrizioni in una volta sola.
- **Reporting Avanzato**: Genera report HTML e Markdown dettagliati per ogni analisi.

### ⚡ Hardware Acceleration & Cinematic Crop
- **Auto-Rilevamento GPU**: 
    - **NVIDIA**: Usa `h264_nvenc` (Windows/Linux) per rendering ultra-veloce.
    - **macOS**: Usa `h264_videotoolbox` nativo su Apple Silicon/Intel.
    - **CPU Fallback**: Ottimizzato con `libx264` se nessuna GPU è disponibile.
- **Cinematic Crop**: Zoom 115% + Crop Centrale + Bande Nere automatiche per un look verticale (9:16) perfetto.

### 🧪 Dry-Run Interattivo
- **Anteprima Console**: Visualizza le clip proposte e le scene rilevate senza generare video.
- **Modifica Live**: Modifica, aggiungi o rimuovi clip direttamente dalla riga di comando prima del rendering.

### 🧠 TOON
- **Multi-Agent System**: 4 fasi (Scene Detect -> Classify -> Clip Select -> Supervise).
- **Efficienza Token**: Risparmio del 70%+ sui costi LLM grazie al formato TOON Tabular ottimizzato.
- **Micro-Allineamento**: Tagli precisi guidati dai timestamp delle parole.

### 8. Gestione Modelli Deepgram
Klipto include un selettore intelligente "Model-First" o "Language-First" per aiutarti a scegliere il modello migliore.
1. Avvia Klipto: `run.bat`
2. Seleziona **[8] Modelli Deepgram** (o usa `python src/main.py transcriber --interactive`)
3. Segui il wizard:
   - Scegli **Mode** (Batch per file, Streaming per live)
   - Scegli **Lingua** (es. `it`, `en`, `multi`)
   - Scegli **Caso d'uso** (es. `general`, `meeting`, `medical`)
   - Seleziona il modello finale dalla lista filtrata.
4. Il modello scelto verrà salvato in `.env` come default.custom addestrati specificamente.

---

## 🚀 Installazione

### Requisiti
- Python 3.10+
- FFmpeg (deve essere nel PATH di sistema)

### Setup
```bash
# Clone e Installazione dipendenze
pip install -r requirements.txt
```

### Configurazione
Crea un file `.env` nella root:
```env
DEEPGRAM_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
```

---

## 💻 Utilizzo

### 1. Pipeline di Produzione (Consigliato)
Il nuovo standard per generare clip. Gestisce tutto: download, analisi, taglio e report.

```bash
# Analisi completa di un video da file locale + trascrizione
python src/pipeline.py -t data/transcripts/video.json -v data/videos/video.mp4

# Batch Processing (elabora tutti i json nella cartella)
python src/pipeline.py --batch data/transcripts/*.json

# Dry-Run (solo analisi e anteprima, niente video)
python src/pipeline.py -t transcript.json --dry-run
```

### 2. Main CLI (Legacy/Debug)
Per test specifici sui singoli moduli agenti.

```bash
# Analisi con debug dettagliato dei token
python src/main.py process "video.mp4" --analyzer multiagent --debug

# Gestione Modelli Trascrizione
python src/main.py transcriber --list        # Visualizza lista modelli
python src/main.py transcriber --interactive # Selezione guidata
```

---

## 📊 Performance & Costi

| Modulo | Costo Stimato (30min video) | Note |
|--------|-----------------------------|------|
| **Trascrizione** | ~$0.13 | Deepgram Nova-2 |
| **Analisi AI** | ~$0.01 - $0.05 | Dipende dal modello (Flash vs Pro) |
| **Tempo Rendering** | ~2 min (NVIDIA/Mac) | Hardware Accelerated |

---

## 📄 Licenza
MIT License - vedi [LICENSE](LICENSE)

<p align="center">
  Made with ❤️ for content creators
</p>


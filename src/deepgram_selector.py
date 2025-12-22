from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import json

# ---- Data structures ----

@dataclass
class ModelVariant:
    """Single API entry (often duplicated across language / batch / streaming / version)."""
    name: str 
    canonical_name: str
    architecture: str
    languages: List[str]
    version: str
    uuid: str
    batch: bool
    streaming: bool
    formatted_output: bool

@dataclass
class ModelCatalogItem:
    """Aggregated view of multiple variants -> one logical model."""
    canonical_name: str
    architecture: str
    use_case: str
    languages: Set[str] = field(default_factory=set)

    # capability aggregated across all variants
    batch_any: bool = False
    streaming_any: bool = False
    formatted_any: bool = False

    # keep raw variants if needed
    variants: List[ModelVariant] = field(default_factory=list)

# ---- Helpers (naming / grouping) ----

_USE_CASE_ORDER = [
    "general", "meeting", "phonecall", "voicemail", "video",
    "medical", "finance", "conversationalai", "automotive",
    "drivethru", "atc", "phoneme", "whisper"
]

_USE_CASE_DESCRIPTIONS = {
    "general": "Standard (Nova/Enhanced) - ✅ Consigliato per tutto",
    "whisper": "OpenAI Whisper (Cloud Deepgram)",
    "phoneme": "Trascrizione Fonetica (IPA) - ⚠️ No testo normale",
    "meeting": "Riunioni e più speaker",
    "phonecall": "Audio telefonico (8khz)",
    "voicemail": "Segreteria telefonica",
    "video": "Audio da video/social",
    "medical": "Terminologia medica",
    "finance": "Terminologia finanziaria",
    "automotive": "Comandi vocali auto",
    "conversationalai": "Interazioni Bot/AI",
    "drivethru": "Ordini vocali",
}

_FAMILY_RANK = {
    "nova-3": 0,
    "nova-2": 1,
    "polaris": 2, 
    "base": 3,
    "whisper": 4,
    "other": 5,
}

_FAMILY_LABELS = {
    "nova-3": "🚀 Nova-3 (Top Gamma)",
    "nova-2": "💎 Nova-2 (Best Value)",
    "polaris": "⚖️ Enhanced (Bilanciato)",
    "base": "⚡ Base (Veloce)",
    "whisper": "🧠 Whisper (OpenAI)",
    "other": "Other"
}

class DeepgramModelSelector:
    """
    Advanced selector for Deepgram models.
    Implements 'Language-First' strategy and deduplication by canonical_name.
    """
    
    def __init__(self, models_data: dict | list):
        # Handle dict wrapper from cache output or raw list
        if isinstance(models_data, dict) and "models" in models_data:
            raw_models = models_data["models"]
        elif isinstance(models_data, list):
            raw_models = models_data
        else:
            raw_models = []
            
        self.catalog = self._build_catalog(raw_models)

    def _infer_use_case(self, canonical_name: str) -> str:
        if canonical_name.startswith("whisper-") or canonical_name == "whisper":
            return "whisper"
            
        parts = canonical_name.split("-")
        if len(parts) >= 2:
            tail = parts[-1]
            if tail in _USE_CASE_ORDER:
                return tail
                
        if canonical_name in _USE_CASE_ORDER:
             return canonical_name
             
        return "general"

    def _architecture_family(self, arch: str, canonical_name: str) -> str:
        if arch in ("nova-3", "nova-2", "polaris", "base", "whisper"):
            return arch
        if canonical_name.startswith("nova-3"): return "nova-3"
        if canonical_name.startswith("nova-2") or canonical_name.startswith("2-"): return "nova-2"
        if canonical_name.startswith("enhanced-"): return "polaris"
        if canonical_name.startswith("whisper"): return "whisper"
        return "other"

    def _build_catalog(self, raw_models: list) -> Dict[str, ModelCatalogItem]:
        catalog: Dict[str, ModelCatalogItem] = {}
        
        for m in raw_models:
            canonical = m.get("canonical_name") or m.get("name")
            if not canonical: continue
            
            # Normalize languages
            langs_raw = m.get("languages") or m.get("language") or []
            if isinstance(langs_raw, str): langs_raw = [langs_raw]
            
            variant = ModelVariant(
                name=m.get("name", ""),
                canonical_name=canonical,
                architecture=m.get("architecture", "other"),
                languages=langs_raw,
                version=m.get("version", ""),
                uuid=m.get("uuid", ""),
                batch=bool(m.get("batch", False)),
                streaming=bool(m.get("streaming", False)),
                formatted_output=bool(m.get("formatted_output", False)),
            )
            
            if canonical not in catalog:
                use_case = self._infer_use_case(canonical)
                arch = self._architecture_family(variant.architecture, canonical)
                catalog[canonical] = ModelCatalogItem(
                    canonical_name=canonical,
                    architecture=arch,
                    use_case=use_case
                )
            
            item = catalog[canonical]
            item.variants.append(variant)
            item.languages.update(variant.languages)
            item.batch_any = item.batch_any or variant.batch
            item.streaming_any = item.streaming_any or variant.streaming
            item.formatted_any = item.formatted_any or variant.formatted_output
            
        return catalog

    def _pick_from_list(self, prompt: str, options: List[str], page: int = 15, auto_select_single: bool = False) -> str:
        if not options:
            return None
        
        # If only one option and auto-select is on, return it immediately
        if auto_select_single and len(options) == 1:
            return options[0]

        filtered = options[:]
        show_all = False
        
        while True:
            print("\n" + prompt)
            print("-" * 72)
            
            # Pagination / Truncation
            display_list = filtered if show_all else filtered[:page]
            
            for i, opt in enumerate(display_list, start=1):
                print(f"  [{i:2d}] {opt}")
            
            remaining = len(filtered) - len(display_list)
            if remaining > 0:
                print(f"  ... ({remaining} altri - scrivi per cercare o 'all' per vedere tutti)")
            
            print("-" * 72)
            cmd = input("👉 Seleziona numero, scrivi per cercare, 'all' espandi, 'q' esci: ").strip()
            
            if cmd.lower() in ['q', 'exit']:
                return None
            
            if cmd.lower() == 'all':
                show_all = True
                continue
                
            if cmd == "?" or cmd == "":
                if cmd == "?":
                    filtered = options[:]
                    show_all = False
                continue
                
            if cmd.isdigit():
                idx = int(cmd)
                if 1 <= idx <= len(display_list):
                    return display_list[idx - 1]
                print("❌ Numero non valido.")
                continue

            # filter
            if cmd:
                new_filtered = [o for o in options if cmd.lower() in o.lower()]
                if not new_filtered:
                    print("⚠️  Nessuna corrispondenza.")
                else:
                    filtered = new_filtered
                    show_all = False
                    if len(filtered) == 1:
                         print(f"✅ Trovato unico: {filtered[0]}")
                         return filtered[0]

    def interactive_select_model(self) -> Tuple[str, str, str]:
        """
        Returns (canonical_model_name, language, mode) or (None, None, None) if cancelled.
        """
        mode = "batch" 
        
        # Step 2: Language
        all_langs = set()
        for item in self.catalog.values():
            if not item.batch_any: continue
            all_langs.update(item.languages)
        
        priority_langs = ["it", "en", "multi", "es", "fr", "de"]
        sorted_others = sorted([l for l in all_langs if l not in priority_langs])
        
        options = []
        for pl in priority_langs:
            if pl in all_langs:
                options.append(pl)
        options.extend(sorted_others)
        
        print("\n🌍 Seleziona Lingua:")
        language = self._pick_from_list("Lingue più comuni in alto (scrivi es. 'ru' per cercare altre):", options, page=10)
        if not language: return None, None, None
        
        # Step 3: Use Case (With Descriptions)
        use_cases = set()
        for item in self.catalog.values():
            if not item.batch_any: continue
            if language not in item.languages: continue
            use_cases.add(item.use_case)
            
        sorted_use_cases = sorted(list(use_cases), key=lambda u: (_USE_CASE_ORDER.index(u) if u in _USE_CASE_ORDER else 999, u))
        
        # Create display labels for use cases
        uc_display_options = []
        uc_map = {}
        
        for uc in sorted_use_cases:
            desc = _USE_CASE_DESCRIPTIONS.get(uc, "")
            label = f"{uc:<18} | {desc}"
            uc_display_options.append(label)
            uc_map[label] = uc
            
        print(f"\n🎯 Seleziona Scenario per '{language}':")
        chosen_uc_label = self._pick_from_list("Scenari disponibili:", uc_display_options, auto_select_single=True)
        if not chosen_uc_label: return None, None, None
        
        use_case = uc_map[chosen_uc_label]
        
        # Step 4: Model
        candidates = []
        for item in self.catalog.values():
            if not item.batch_any: continue
            if language not in item.languages: continue
            if item.use_case != use_case: continue
            candidates.append(item)
            
        if not candidates:
            print("⚠️ Nessun modello specifico. Mostro tutti per questa lingua.")
            for item in self.catalog.values():
                if not item.batch_any: continue
                if language not in item.languages: continue
                candidates.append(item)
        
        candidates.sort(key=lambda x: (_FAMILY_RANK.get(x.architecture, 99), -len(x.languages)))
        
        display_options = []
        mapping = {}
        for it in candidates:
            # Friendly Family Name
            fam_label = _FAMILY_LABELS.get(it.architecture, it.architecture.capitalize())
            
            # Formatting capability
            fmt_icon = "✅ Punteggiatura" if it.formatted_any else "❌ No Punteggiatura"
            
            label = f"{it.canonical_name:<25} | {fam_label:<25} | {fmt_icon}"
            display_options.append(label)
            mapping[label] = it.canonical_name
            
        print(f"\n🎙️  Seleziona Modello ({len(candidates)} opzioni):")
        chosen_display = self._pick_from_list("Modelli compatibili:", display_options, page=10, auto_select_single=True)
        if not chosen_display: return None, None, None
        
        chosen_model = mapping[chosen_display]
        return chosen_model, language, mode

    def get_model_details(self, model_name: str) -> dict:
        item = self.catalog.get(model_name)
        if not item: return None
        return {
            "name": item.canonical_name,
            "architecture": item.architecture,
            "use_case": item.use_case,
            "languages": list(item.languages),
            "batch": item.batch_any
        }

import httpx
from config import DEEPGRAM_API_KEY
import json

def analyze_structure():
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json"
    }
    url = "https://api.deepgram.com/v1/models"
    
    print(f"Requesting {url}...")
    with httpx.Client() as client:
        resp = client.get(url, headers=headers)
        data = resp.json()
        
        models = data.get("stt", [])
        print(f"Total entries: {len(models)}")
        
        # Count occurences of names
        name_counts = {}
        for m in models:
            name = m.get("name")
            if name not in name_counts:
                name_counts[name] = []
            name_counts[name].append(m)
            
        print(f"Unique model names: {len(name_counts)}")
        
        # Inspect a duplicate
        for name, entries in name_counts.items():
            if len(entries) > 1:
                print(f"\nModel '{name}' has {len(entries)} entries.")
                print("First entry keys:", entries[0].keys())
                print("Comparing 'language' fields:")
                langs = [e.get("language") for e in entries]
                print(f"Languages found: {langs[:10]} ...")
                
                # Check if architecture or version differs
                versions = set(e.get("version") for e in entries)
                print(f"Versions: {versions}")
                break

if __name__ == "__main__":
    analyze_structure()

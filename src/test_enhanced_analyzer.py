# test_enhanced_analyzer.py
import json
import sys
from pathlib import Path
from semantic_analyzer import (
    EnhancedMultiAgentAnalyzer,
    estimate_token_savings,
    SceneDetector,
    TranscriptToToon,
)
import toon

# Dummy transcript if none exists, or use a sample
SAMPLE_TRANSCRIPT_PATH = Path("data/transcripts/5Qj-yachv1U_transcript.json")

def test_v3_features():
    """Test TOON v3.0 specific features."""
    print("\n" + "=" * 60)
    print("TESTING TOON v3.0 FEATURES")
    print("=" * 60)
    
    # Test Key Folding (Encoder)
    data = {
        "user": {
            "profile": {
                "name": "Alice",
                "age": 30
            }
        }
    }
    
    # Safe folding should produce user.profile.name if possible? 
    # Wait, flatten_depth=999 by default.
    # Structure: user -> profile -> name (value string)
    # user has 1 key (profile). profile has 2 keys (name, age).
    # So user -> profile can be folded. profile -> name/age cannot.
    # Result: user.profile: {name:.., age:..}
    encoded = toon.encode(data, key_folding="safe")
    print(f"Key Folding Output:\n{encoded}")
    
    assert "user.profile:" in encoded
    assert "name: Alice" in encoded
    
    # Test Path Expansion (Decoder)
    expanded = toon.decode(encoded, expand_paths="safe")
    print(f"Path Expansion Decoded: {expanded}")
    
    assert expanded["user"]["profile"]["name"] == "Alice"
    print("✅ Key Folding & Path Expansion passed")

    # Test Tabular List Item Optimization
    list_data = {
        "items": [
            {"id": 1, "val": "a"},
            {"id": 2, "val": "b"}
        ]
    }
    # Should wrap in object to trigger property encoding? 
    # encode({"group": [{"id":...}]})
    mixed_data = {
        "group": [
            {"sub": [{"id": 1, "a": 1}, {"id": 2, "a": 2}]}
        ]
    }
    # This structure: group is list. Item 0 is dict. Key "sub" is tabular list.
    # Output should use optimized format:
    # group[1]:
    #   - sub[2]{id,a}:
    #       1,1
    #       2,2
    encoded_list = toon.encode(mixed_data)
    print(f"List Optimization Output:\n{encoded_list}")
    assert "sub[2]{" in encoded_list
    print("✅ List Item Optimization passed")


def main():
    if not SAMPLE_TRANSCRIPT_PATH.exists():
        print(f"Sample transcript not found at {SAMPLE_TRANSCRIPT_PATH}")
        print("Please provide a path to a valid transcript.json")
        return

    # Carica transcript
    transcript = json.loads(SAMPLE_TRANSCRIPT_PATH.read_text(encoding="utf-8"))

    # Test v3.0 features
    test_v3_features()

    # Test token savings
    print("=" * 60)
    print("TOKEN SAVINGS ESTIMATE")
    print("=" * 60)
    savings = estimate_token_savings(transcript)
    for field, data in savings.items():
        print(f"{field}:")
        print(f"  Items: {data['items']}")
        print(f"  JSON: {data['json_tokens']} tokens")
        print(f"  TOON: {data['toon_tokens']} tokens")
        print(f"  Savings: {data['savings_percent']}%")
        print()

    # Test scene detection (local, no LLM)
    print("=" * 60)
    print("SCENE DETECTION (LOCAL)")
    print("=" * 60)
    detector = SceneDetector()
    scenes = detector.detect_scenes(transcript)
    for scene in scenes:
        status = "❌" if scene.is_excluded else "✓"
        print(f"[{scene.id}] {scene.start_time:.1f}-{scene.end_time:.1f}s "
              f"({scene.duration:.1f}s) {status} {scene.scene_type}")

    # Test TOON conversion
    print("\n" + "=" * 60)
    print("TOON OUTPUT SAMPLE (utterances)")
    print("=" * 60)
    toon_output = TranscriptToToon.utterances(transcript, max_text_length=80)
    print(toon_output[:1000] + "...")

    # Full analysis (requires API key)
    print("\n" + "=" * 60)
    print("FULL ANALYSIS (Dry Run check)")
    print("=" * 60)
    try:
        # We instantiate but don't run full analysis to save cost in test, 
        # unless user explicitly wants to run it.
        # Here we just check initialization.
        analyzer = EnhancedMultiAgentAnalyzer(debug=True)
        print("Analyzer initialized successfully.")
        
        # Uncomment to run full analysis
        # result = analyzer.analyze(
        #     transcript=transcript,
        #     video_title="Test Video Games Week",
        #     max_clips=5,
        #     language="it",
        #     target_platform="all",
        # )
        # analyzer.save_analysis(result, "5Qj-yachv1U")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

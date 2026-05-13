"""
test_pipeline.py
----------------
Validates the full pipeline step by step.
Run this to verify your setup before using compile_wiki.py and qa.py.

Usage:
    python test_pipeline.py
"""

import os
import sys

WIKI_DIR = os.path.join(os.path.dirname(__file__), "wiki")
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")

PASS = "✓"
FAIL = "✗"
WARN = "!"


def check(label, ok, detail=""):
    sym = PASS if ok else FAIL
    line = f"  [{sym}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def section(title):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


all_ok = True

section("1. Raw sources")
raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".md")]
all_ok &= check(f"Raw source files found", len(raw_files) >= 1,
                f"{len(raw_files)} files")
total_chars = sum(
    len(open(os.path.join(RAW_DIR, f)).read()) for f in raw_files
)
check(f"Total raw content", True, f"{total_chars:,} chars")

section("2. sentence-transformers (local embedding model)")
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    v = m.encode(["test query"], normalize_embeddings=True)
    all_ok &= check("Model loads and encodes", True, f"dim={v.shape[1]}")
except Exception as e:
    all_ok &= check("Model loads and encodes", False, str(e))
    print("    → Fix: pip install sentence-transformers")

section("3. Wiki articles")
wiki_files = [f for f in os.listdir(WIKI_DIR)
              if f.endswith(".md") and not f.startswith("_")]
wiki_ok = len(wiki_files) >= 1
check(f"Wiki articles present", wiki_ok,
      f"{len(wiki_files)} articles" if wiki_ok else "EMPTY — run compile_wiki.py first")

if wiki_ok:
    section("4. Embedding retrieval quality")
    import pickle, numpy as np
    emb_path = os.path.join(WIKI_DIR, "_embeddings.pkl")
    if not os.path.exists(emb_path):
        print("  Building index...")
        articles = {}
        for f in wiki_files:
            articles[f[:-3]] = open(os.path.join(WIKI_DIR, f)).read()
        vecs = m.encode(
            [f"{t}\n{' '.join(articles[t].split()[:500])}" for t in articles],
            normalize_embeddings=True, show_progress_bar=False
        )
        import pickle
        with open(emb_path, "wb") as pf:
            pickle.dump({"topics": list(articles.keys()), "vectors": vecs}, pf)

    with open(emb_path, "rb") as pf:
        cache = pickle.load(pf)

    test_cases = [
        ("how does the system detect sadness or distress?",
         {"AffectiveComputing", "SafetyAndEscalation", "EmpathyDesign"}),
        ("what is the voice interaction latency budget?",
         {"VoiceInterface", "PlatformArchitecture"}),
        ("how does matching work and how are results explained?",
         {"ExperienceMatching", "MentorMenteeSystem"}),
    ]

    for question, expected_topics in test_cases:
        qv = m.encode([question], normalize_embeddings=True)[0]
        scores = cache["vectors"] @ qv
        top4 = {cache["topics"][i] for i in sorted(range(len(scores)),
                                                    key=lambda i: -scores[i])[:4]}
        hit = bool(top4 & expected_topics)
        check(f'Retrieval: "{question[:45]}..."', hit,
              f"top4={sorted(top4)}")

section("5. Anthropic API key")
# Load .env if present
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ.setdefault("ANTHROPIC_API_KEY", line.strip().split("=", 1)[1])

key = os.environ.get("ANTHROPIC_API_KEY", "")
has_key = bool(key)
check("ANTHROPIC_API_KEY set", has_key,
      "key found" if has_key else "not set — create project2/.env with ANTHROPIC_API_KEY=sk-ant-...")

if has_key:
    try:
        import anthropic
        c = anthropic.Anthropic()
        r = c.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=5,
            messages=[{"role": "user", "content": "hi"}]
        )
        check("API call succeeds", True, f"response: '{r.content[0].text}'")
    except Exception as e:
        check("API call succeeds", False, str(e))
        all_ok = False

print(f"\n{'='*50}")
if wiki_ok and has_key:
    print("  All systems ready. Try:")
    print("    python qa.py -q 'How does crisis detection work?'")
    print("    python qa.py  (interactive mode)")
elif not wiki_ok and has_key:
    print("  API key set. Run the compiler:")
    print("    python compile_wiki.py")
    print("  Then run Q&A:")
    print("    python qa.py -q 'How does crisis detection work?'")
elif wiki_ok and not has_key:
    print("  Wiki present — Q&A works if you set your API key.")
    print("  Create project2/.env:")
    print("    ANTHROPIC_API_KEY=sk-ant-...")
else:
    print("  Setup needed:")
    print("  1. Create project2/.env with ANTHROPIC_API_KEY=sk-ant-...")
    print("  2. Run: python compile_wiki.py")
    print("  3. Run: python qa.py")
    print("\n  Get your API key at: https://console.anthropic.com/")
print(f"{'='*50}\n")

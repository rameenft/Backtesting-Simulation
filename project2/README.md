# Project 2: LLM Personal Knowledge Base
## Topic: AI-Powered Social Platform for Shared Human Experiences

INDENG 231 — Data Modeling

---

## How It Works

This is a [Karpathy-style](https://twitter.com/karpathy/status/1751350002281300461) personal knowledge base built with Claude.

**The pipeline:**

```
raw/                raw research notes (messy, informal, fragmented)
    ↓
compile_wiki.py     Claude Opus 4.7 reads all 10 notes at once,
                    synthesizes structured wiki articles, resolves
                    contradictions, surfaces cross-source insights
    ↓
wiki/               10 interlinked markdown articles with [[WikiLinks]]
    ↓
qa.py               embedding-based retrieval (sentence-transformers, local)
                    + Claude Haiku for answer generation with citations
```

The compilation step is where the LLM adds real value: it synthesizes across multiple messy sources, resolves contradictions between them, and surfaces implications not stated in any single note.

---

## Setup

**1. Install dependencies**
```bash
pip install anthropic sentence-transformers
```

**2. Get an Anthropic API key**

Go to [console.anthropic.com](https://console.anthropic.com/) → API Keys → Create key.

**3. Create a `.env` file in `project2/`**
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**4. Verify everything is set up**
```bash
python test_pipeline.py
```

---

## Usage

### Step 1: Compile the wiki (requires API key, ~5-10 min, costs ~$1-2)
```bash
python compile_wiki.py            # compile all 10 articles
python compile_wiki.py --force    # recompile even if wiki/ exists
python compile_wiki.py --topic AffectiveComputing  # one article only
```

### Step 2: Ask questions (requires API key for answer generation)
```bash
# Single question
python qa.py -q "How does the platform detect when a user is in crisis?"
python qa.py -q "What is the latency budget for voice interaction?"
python qa.py -q "How does mentor burnout get prevented?"
python qa.py -q "What makes this matching system better than tag-based matching?"

# Interactive mode
python qa.py

# List articles + link graph stats
python qa.py --list

# Rebuild embedding index (no API key needed)
python qa.py --index
```

---

## Project Structure

```
project2/
├── raw/                        ← 10 messy research notes (the input)
│   ├── 01_shared_experience_platforms.md
│   ├── 02_mentor_mentee_matching.md
│   ├── 03_affective_computing.md
│   ├── 04_speech_to_text_pipelines.md
│   ├── 05_text_to_speech_synthesis.md
│   ├── 06_dialogue_systems.md
│   ├── 07_empathetic_ai_design.md
│   ├── 08_experience_storage_retrieval.md
│   ├── 09_platform_integration_architecture.md
│   └── 10_llm_knowledge_base_design.md
│
├── wiki/                       ← Compiled wiki articles (LLM output)
│   ├── AffectiveComputing.md
│   ├── DialogueSystem.md
│   ├── ...
│   ├── _index.json             ← Link graph + word counts
│   └── _embeddings.pkl         ← Cached sentence-transformer embeddings
│
├── compile_wiki.py             ← LLM compiler (Claude Opus 4.7)
├── qa.py                       ← Q&A interface (sentence-transformers + Claude Haiku)
├── build_index.py              ← Rebuild _index.json (no API needed)
├── test_pipeline.py            ← Validates the full setup
└── generate_report2.py         ← Generates the PDF report
```

---

## Design Details

**Compilation** uses Claude Opus 4.7 with adaptive thinking. The prompt explicitly instructs the model to:
- Synthesize across *all* raw sources, not just one
- Identify and resolve contradictions between notes
- Surface implications that emerge only from reading multiple notes together
- Use `[[WikiLink]]` format to cross-reference related articles

**Retrieval** uses `sentence-transformers` (`all-MiniLM-L6-v2`, 80MB, runs locally). Embeddings are cached in `wiki/_embeddings.pkl`. Cosine similarity finds the 4 most relevant articles for each question — this correctly handles semantic matches that keyword overlap misses (e.g. "sad" → `AffectiveComputing`).

**Q&A** uses Claude Haiku 4.5 — fast and cost-effective for retrieval-augmented generation. Retrieved articles are injected as context and the model answers with citations.

# INDENG 231 — Data Modeling Projects

UC Berkeley, Spring 2026

This repository contains two independent course projects for INDENG 231. Each project lives in its own section below, with setup and usage instructions.

---

## Project 1: Backtesting Simulation for Trading Strategies

### Summary

Project 1 builds a daily-close backtesting simulation system on five years of Nasdaq-100 constituent data (April 2021 to April 2026, 1,255 trading days, 101 stocks). The system enforces strict no-lookahead constraints — strategies may only use price data available up to the current day — along with no short selling, no leverage, and no transaction costs.

The project is organized across three deliverables. The first tests five classic single-asset strategies on Apple (AAPL): momentum, mean reversion, MACD, Bollinger Bands, and RSI. Momentum emerged as the strongest performer (Sharpe 0.66), consistent with the documented momentum effect in large-cap equities. Mean-reversion and RSI strategies struggled because they repeatedly attempt to exit a sustained uptrend. The second deliverable scales to the full 101-stock universe using two portfolio construction methods — equal weighting and inverse-volatility weighting — applied to SMA and top-K momentum selection. Cross-sectional momentum with equal weighting achieved the best Sharpe ratio (1.01) and a 239% cumulative return over the period, capturing the AI-driven bull run of 2023-2025. The third deliverable introduces two new strategies that beat both benchmarks: a risk-adjusted momentum strategy with an RSI overbought filter (Sharpe 1.14), and a market-timed momentum strategy that moves to 100% cash when the broad 90-day market trend is negative (Sharpe 1.35, max drawdown -11.84%).

### Architecture

The system is organized into six modules: `data/` for loading and pivoting the price matrix, `engine/` for the core backtesting loop, `strategies/` for individual strategy implementations, `portfolio/` for weighting helpers, `metrics/` for performance calculations, and `output/` for saving results and plots. Every strategy subclasses `BaseStrategy` and implements a single `generate_weights()` method, so new strategies can be added without touching any other module.

### Setup and Usage

```bash
pip install pandas numpy matplotlib
python main.py                     # run all deliverables
python main.py --deliverable 3     # single-stock strategies only
python main.py --deliverable 4     # portfolio strategies
python main.py --deliverable 5     # new strategies vs benchmarks
```

Results are saved to `output/` as CSV metrics tables and NAV curve plots.

---

## Project 2: LLM-Powered Personal Knowledge Base

### Summary

Project 2 implements a Karpathy-style personal knowledge base using the Google Gemini API (free tier). The system demonstrates what an LLM compiler can add over a simple document store: it reads all raw source material simultaneously, resolves contradictions between sources, and surfaces cross-source implications that no single note contains. The topic is the design of an AI-powered social platform for shared human experiences, covering mentor-mentee matching, affective computing, voice dialogue systems, and empathetic AI design.

The pipeline has three layers. The first is a corpus of ten informal research notes written with deliberate gaps, redundancy, and contradictions to give the compiler real synthesis work. The second is an LLM compiler (`compile_wiki.py`) that reads all ten notes in a single context window using Gemini 2.5 Flash and generates ten structured, interlinked wiki articles. Each article includes Key Concepts, How It Works, Design Implications, Open Questions, and cross-references using `[[WikiLink]]` syntax. The compiler produced 6,085 words across ten articles with 90 WikiLinks. The third layer is a retrieval-augmented Q&A interface (`qa.py`) that uses local sentence-transformer embeddings (`all-MiniLM-L6-v2`) for cosine similarity retrieval and Gemini for generating cited answers. The embedding approach handles semantic queries that keyword matching would miss — for example, "how does the system detect sadness?" retrieves `AffectiveComputing` at 0.615 cosine similarity despite zero keyword overlap.

The entire pipeline runs at zero cost: Gemini 2.5 Flash on the free tier for compilation and Q&A, and sentence-transformers running locally for retrieval with no API calls.

### Architecture

```
raw/              10 messy research notes (input)
    |
compile_wiki.py   Gemini 2.5 Flash reads all notes, synthesizes wiki
    |
wiki/             10 interlinked markdown articles with WikiLinks
    |
qa.py             sentence-transformers retrieval + Gemini Q&A
```

### Setup and Usage

**1. Install dependencies**
```bash
pip install google-genai sentence-transformers
```

**2. Get a free Google API key** at aistudio.google.com (no credit card required)

**3. Create a `.env` file inside `project2/`**
```
GOOGLE_API_KEY=your-key-here
```

**4. Verify setup**
```bash
python project2/test_pipeline.py
```

**5. Compile the wiki** (run once, takes 5-10 minutes, uses the free API tier)
```bash
python project2/compile_wiki.py
python project2/compile_wiki.py --force       # recompile all articles
python project2/compile_wiki.py --topic AffectiveComputing  # single article
```

**6. Ask questions**
```bash
python project2/qa.py -q "How does the platform detect when a user is in crisis?"
python project2/qa.py -q "What is the latency budget for voice interaction?"
python project2/qa.py                          # interactive mode
python project2/qa.py --list                   # list articles with link counts
```

---

## Repository Structure

```
.
├── data/                     price loading and matrix construction
├── engine/                   backtesting loop
├── strategies/               individual strategy implementations (Project 1)
├── portfolio/                weighting helpers
├── metrics/                  performance metrics
├── output/                   saved results and plots
├── main.py                   Project 1 entry point
├── config.py                 Project 1 configuration
├── nasdaq100_daily_5y.csv    price data (101 stocks, 5 years)
├── requirements.txt          Project 1 dependencies
├── INDENG231_Project1_Report.pdf
│
└── project2/
    ├── raw/                  10 research notes (input corpus)
    ├── wiki/                 compiled wiki articles (LLM output)
    ├── compile_wiki.py       LLM compiler
    ├── qa.py                 Q&A interface
    ├── build_index.py        rebuild link graph index
    ├── test_pipeline.py      end-to-end validation
    ├── generate_report2.py   PDF report generator
    └── INDENG231_Project2_Report.pdf
```

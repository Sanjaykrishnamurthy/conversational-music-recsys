# AGENTS.md — RecSys2026 Project Guide

> This file provides context for AI coding assistants (e.g., Wibey, Claude Code, Copilot) working in this repository.
> Read this file before making changes to understand project conventions, structure, and constraints.

---

## 🎯 Project Overview

**RecSys Challenge 2026 — Music-CRS** is a competition project for the
[20th ACM Conference on Recommender Systems](https://www.recsyschallenge.com/2026/)
(Minneapolis, MN — Sep 28 – Oct 2, 2026).

The goal is to build a **Conversational Recommender System (CRS)** that:
1. Engages users in **multi-turn music preference dialogues**
2. **Retrieves relevant tracks** from a 1M+ track catalog
3. **Generates natural language responses** that are coherent and informative

**Evaluation Metrics:**

| Category     | Metric                                        |
|--------------|-----------------------------------------------|
| Retrieval    | nDCG@{1, 10, 20} — macro-averaged over sessions & turns |
| Diversity    | Catalog Coverage                              |
| Text Quality | Response Lexical Diversity                    |

Submissions are made via **CodaBench** leaderboard. Final ranking is on the **Blind B** split.

---

## 📦 Technology Stack

### Language & Runtime
- **Python** `>=3.11, <3.13` (prefer 3.11 for stability)
- **Package manager:** `pip` with `pyproject.toml` (hatchling build backend)

### Core ML & Data
| Package | Role |
|---|---|
| `torch>=2.3.0` | Neural model training & inference |
| `numpy`, `pandas`, `scipy` | Numerical computation & data wrangling |
| `scikit-learn>=1.5.0` | Classical ML utilities, preprocessing |

### HuggingFace Ecosystem
| Package | Role |
|---|---|
| `transformers>=4.44.0` | Pretrained LM backbone (encoder/decoder) |
| `datasets>=2.20.0` | Loading TalkPlayData-Challenge from HuggingFace Hub |
| `sentence-transformers>=3.0.0` | Dense embeddings for retrieval |
| `accelerate>=0.33.0` | Distributed training & mixed precision |
| `peft>=0.12.0` | LoRA / parameter-efficient fine-tuning |

### Retrieval & Search
| Package | Role |
|---|---|
| `rank-bm25>=0.2.2` | Sparse BM25 lexical retrieval |
| `faiss-cpu>=1.8.0` | Dense vector similarity search (ANN) |

### LLM / Generation
| Package | Role |
|---|---|
| `openai>=1.40.0` | GPT-4o for response generation |
| `anthropic>=0.34.0` | Claude for response generation |
| `tiktoken>=0.7.0` | Token counting & context management |

### Experiment Tracking
| Package | Role |
|---|---|
| `wandb>=0.17.0` | Primary experiment tracking & sweep orchestration |
| `mlflow>=2.15.0` | Secondary tracking & model registry |

### Configuration
| Package | Role |
|---|---|
| `hydra-core>=1.3.2` | Hierarchical config management |
| `omegaconf>=2.3.0` | YAML-based config composition |

### Code Quality
| Tool | Config | Line Length |
|---|---|---|
| `black` | `pyproject.toml` | 100 |
| `isort` | `pyproject.toml` (profile: black) | 100 |
| `ruff` | `pyproject.toml` | 100 |

---

## 🗂️ Project Structure

```
RecSys2026/
├── AGENTS.md              ← You are here
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── data/
│   ├── raw/               # Raw downloads from HuggingFace (do NOT modify)
│   ├── processed/         # Preprocessed / featurized data
│   └── splits/            # Train / Dev / BlindA / BlindB splits
│
├── src/                   # Main source package
│   ├── models/            # CRS model implementations (retrieval + generation)
│   ├── retrieval/         # BM25, dense, hybrid retrieval pipelines
│   ├── evaluation/        # Metrics: nDCG, coverage, lexical diversity
│   └── utils/             # Data loaders, config helpers, logging
│
├── baselines/             # Cloned official challenge baselines (treat as read-only reference)
│
├── configs/               # Hydra YAML configs
│   ├── model/             # Model architecture configs
│   ├── retrieval/         # Retrieval pipeline configs
│   └── experiment/        # Full experiment sweep configs
│
├── experiments/
│   └── runs/              # Per-run artifacts (auto-generated, gitignored)
│
├── notebooks/             # Jupyter EDA and prototyping notebooks
│
├── scripts/               # Standalone CLI scripts (training, inference, eval)
│
├── submissions/           # CodaBench submission files (JSON/JSONL)
│
├── tests/                 # Pytest test suite
│   └── test_*.py
│
└── docs/                  # Documentation, paper notes, references
```

### Key Source Modules

```
src/
├── models/
│   ├── crs_pipeline.py    # End-to-end CRS: retrieval → re-rank → generate
│   ├── retriever.py       # Unified retriever interface
│   └── generator.py       # LLM-based response generator
├── retrieval/
│   ├── bm25.py            # BM25 sparse retrieval
│   ├── dense.py           # FAISS + sentence-transformers dense retrieval
│   └── hybrid.py          # Reciprocal Rank Fusion (RRF) hybrid
├── evaluation/
│   ├── ndcg.py            # nDCG@{1,10,20} per-session/turn
│   ├── diversity.py       # Catalog coverage metric
│   └── lexical.py         # Response lexical diversity
└── utils/
    ├── data_loader.py     # HuggingFace datasets wrapper
    ├── config.py          # Hydra config utilities
    └── logging.py         # Loguru-based logging setup
```

---

## 📀 Dataset Reference (TalkPlayData-Challenge)

> Full documentation with schemas: **[`data/README.md`](data/README.md)**

Raw data lives in `data/raw/` and **must never be modified**. There are **6 datasets** across **14 parquet files** (~872 MB total):

| Dataset | Path under `data/raw/` | Key Rows | Purpose |
|---|---|---:|---|
| **Conversations** | `TalkPlayData-Challenge-Dataset/` | 15,199 train / 1,000 dev | Multi-turn dialogues (avg ~8 turns) with `session_id`, `user_id`, `conversations` (list of turns), `conversation_goal`, `goal_progress_assessments` |
| **Blind-A** | `TalkPlayData-Challenge-Blind-A/` | 80 test | Interim leaderboard conversations (same schema) |
| **Track Metadata** | `TalkPlayData-Challenge-Track-Metadata/` | 47,071 all / 7,405 test | Catalog: `track_id`, `track_name`, `artist_name`, `album_name`, `tag_list`, `popularity`, `release_date`, `duration` |
| **Track Embeddings** | `TalkPlayData-Challenge-Track-Embeddings/` | 47,071 all / 7,405 test | 6 embedding spaces per track (~755 MB) |
| **User Metadata** | `TalkPlayData-Challenge-User-Metadata/` | 8,772 | Demographics: `age`, `age_group`, `country_code`, `gender` |
| **User Embeddings** | `TalkPlayData-Challenge-User-Embeddings/` | 8,591 train / 371 warm / 129 cold | CF-BPR user embeddings. Cold = no listening history |

### Track Embedding Types (6 per track)

| Column | Model | Modality |
|---|---|---|
| `audio-laion_clap` | LAION-CLAP | Audio signal |
| `image-siglip2` | SigLIP2 | Album artwork |
| `cf-bpr` | BPR | Collaborative filtering |
| `attributes-qwen3_embedding_0.6b` | Qwen3 0.6B | Track attributes text |
| `lyrics-qwen3_embedding_0.6b` | Qwen3 0.6B | Lyrics text |
| `metadata-qwen3_embedding_0.6b` | Qwen3 0.6B | Metadata text |

### Join Keys
- `user_id` → links Conversations ↔ User Metadata ↔ User Embeddings
- `track_id` → links Track Metadata ↔ Track Embeddings (extract from conversation content)

### Important Data Notes
- Track embeddings are sharded across 4 parquet files — read the whole `data/` directory to concatenate
- `test_cold` users (129) have **no prior listening history** — cold-start challenge
- Conversation turns contain a `thought` field (system internal reasoning) and `role` (user/system)
- The `user_profile` inside conversations is a nested struct with demographics + `user_split`

---

## 🛠️ Development Commands

### Environment Setup
```bash
# Create virtual environment (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -e ".[dev]"
# OR
pip install -r requirements.txt
```

### Code Quality
```bash
# Format code
black src/ tests/ scripts/ --line-length 100
isort src/ tests/ scripts/

# Lint
ruff check src/ tests/ scripts/

# All-in-one (run before committing)
black . && isort . && ruff check .
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_evaluation.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Data Pipeline
```bash
# Download dataset from HuggingFace
python scripts/download_data.py

# Preprocess raw data
python scripts/preprocess.py --config configs/data/default.yaml
```

### Training & Experiments
```bash
# Run experiment with Hydra config
python scripts/train.py experiment=baseline_bm25

# Override config values
python scripts/train.py experiment=dense_retrieval model.encoder=sentence-transformers/all-MiniLM-L6-v2

# WandB sweep
wandb sweep configs/sweep.yaml
wandb agent <sweep-id>
```

### Evaluation
```bash
# Evaluate on dev split
python scripts/evaluate.py --split dev --predictions experiments/runs/<run-id>/predictions.jsonl

# Full metrics report
python scripts/evaluate.py --split dev --verbose
```

### Submission
```bash
# Generate CodaBench submission file
python scripts/generate_submission.py --split blind_a --run-id <run-id>
# Output: submissions/blind_a_<timestamp>.zip
```

---

## 🏗️ Architecture & Design Patterns

### CRS Pipeline
```
User Dialogue (multi-turn)
         │
         ▼
  Context Encoder        ← transformers / sentence-transformers
         │
         ▼
  Query Formulation      ← extract intent + entities from conversation
         │
         ▼
┌────────┴──────────┐
│   Hybrid Retrieval │   ← BM25 (rank-bm25) + Dense (FAISS)
│   RRF Re-ranking   │   ← Reciprocal Rank Fusion
└────────┬──────────┘
         │ top-K tracks
         ▼
  Response Generator     ← OpenAI / Anthropic / local LLM
         │
         ▼
  Natural Language Response + Track Recommendations
```

### Config Management (Hydra)
- All experiments are driven by YAML configs in `configs/`
- Never hardcode hyperparameters in source files — always reference config
- Use `OmegaConf` structured configs for type safety
- Experiment configs compose from `model/`, `retrieval/`, and `data/` configs

### Experiment Tracking
- **WandB** is the primary tracker — log all metrics, artifacts, and configs
- Use `wandb.init(config=cfg)` at the start of every training run
- Save model checkpoints to `experiments/runs/<run-id>/checkpoints/`
- MLflow is secondary — use for model registry when deploying

---

## ✅ Best Practices for AI Assistants

### Before Making Changes
1. Read `pyproject.toml` for build config and tool settings
2. Check `configs/` for existing Hydra configs before creating new ones
3. Run `pytest tests/` to confirm baseline passes
4. Check existing patterns in `src/` before adding new modules
5. **If you are stuck or hitting an error**, check [`INSTRUCTIONS.md`](INSTRUCTIONS.md) first — it contains known pitfalls and fixes from previous sessions

### Maintaining INSTRUCTIONS.md
- If you fix a **non-obvious bug** or discover a **common pitfall** that could trip up another session, **add it to [`INSTRUCTIONS.md`](INSTRUCTIONS.md)**
- Each entry should include: date, symptom (error message), root cause, correct pattern, and incorrect pattern
- This is the project's institutional memory — keep it up to date

### Code Style Rules
- **Line length:** 100 characters (black + ruff enforced)
- **Python version:** Target Python 3.11 syntax; avoid 3.12+ only features
- **Imports:** Use `isort` with `profile = "black"` — absolute imports preferred
- **Type hints:** Always add type annotations for function signatures in `src/`
- **Docstrings:** Use Google-style docstrings for public functions and classes

### Data Handling
- **NEVER modify files in `data/raw/`** — treat as immutable source of truth
- Processed data goes in `data/processed/`, always with a preprocessing script to reproduce it
- Use `datasets` library (HuggingFace) to load `talkpl-ai/talkplay-data-challenge`
- Handle large datasets lazily — avoid loading entire 1M+ track catalog into memory

### Retrieval Module Conventions
- All retrievers must implement a common interface: `retrieve(query: str, top_k: int) -> list[dict]`
- Return format: `[{"track_id": str, "score": float, "metadata": dict}, ...]`
- FAISS indexes are stored in `data/processed/indexes/`
- BM25 index is serialized with `pickle` to `data/processed/bm25_index.pkl`

### Evaluation Conventions
- Always evaluate on **dev split** during development — never on Blind A/B
- Report all three nDCG values: nDCG@1, nDCG@10, nDCG@20
- Include diversity (Catalog Coverage) and text quality (Lexical Diversity) in every evaluation
- Save evaluation results as JSON to `experiments/runs/<run-id>/eval_results.json`

### Submission Rules
- Submissions are JSON/JSONL files uploaded to CodaBench
- **Blind A** = interim leaderboard (use sparingly — limited submissions per day)
- **Blind B** = final leaderboard (used only for final ranking)
- Archive submissions in `submissions/` with descriptive names: `blind_a_<model-name>_<date>.zip`

### Testing Guidelines
- Test files live in `tests/test_*.py` (pytest auto-discovery)
- Write unit tests for all evaluation metric implementations — these are critical
- Mock HuggingFace API calls and OpenAI/Anthropic calls in tests
- Use `pytest.fixture` for shared data; avoid downloading real data in tests

### Secrets & Credentials
- Store API keys in `.env` — never hardcode in source
- Required env vars:
  ```
  OPENAI_API_KEY=...
  ANTHROPIC_API_KEY=...
  WANDB_API_KEY=...
  HF_TOKEN=...
  ```
- Load with `python-dotenv`: `from dotenv import load_dotenv; load_dotenv()`

### Git Workflow
- Branch naming: `feat/<description>`, `fix/<description>`, `exp/<experiment-name>`
- Commit style: conventional commits (`feat:`, `fix:`, `chore:`, `exp:`)
- `experiments/runs/` is gitignored — do NOT commit run artifacts
- `data/` is gitignored — datasets are not tracked in git
- DO commit: configs, source code, scripts, notebooks (cleared outputs preferred)

---

## 📋 Gitignored Paths (Do Not Create Files Here Without Noting It)

```
data/raw/
data/processed/
experiments/runs/
.env
*.pyc
__pycache__/
.venv/
wandb/
mlruns/
*.ckpt
*.pt
*.pkl      (except if explicitly versioned)
```

---

## 🔗 Key References

| Resource | Link |
|---|---|
| Challenge site | https://www.recsyschallenge.com/2026/ |
| Dataset (HuggingFace) | `talkpl-ai/talkplay-data-challenge` |
| Submission platform | CodaBench |
| Conference | ACM RecSys 2026, Minneapolis, Sep 28 – Oct 2 |

---

## 📝 Change Log

- **Always append** a concise description of every change to [`docs/ChangeLog.md`](docs/ChangeLog.md).
- Use the format: `- **YYYY-MM-DD** — <short description of what changed and why>`
- One line per logical change (group related edits into a single entry).
- Do this as the **last step** after completing any modification.

---

*Last updated: April 2026*

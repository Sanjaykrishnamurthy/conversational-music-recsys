# RecSys Challenge 2026 — Music-CRS 🎵

Conversational Music Recommendation System for the [RecSys Challenge 2026](https://www.recsyschallenge.com/2026/).

## 🏆 Challenge Overview

The challenge focuses on building **Conversational Recommender Systems (CRS)** that:
- Understand user music preferences through **multi-turn dialogue**
- Recommend relevant tracks from a 1M+ track catalog
- Generate coherent, informative **natural language responses**

**Conference:** 20th ACM Conference on Recommender Systems, Minneapolis, MN, USA (Sep 28 – Oct 2, 2026)

**Organizers:** KAIST, Pandora/SiriusXM, Deezer Research, Amazon, Politecnico di Bari, Maastricht University

---

## 📦 Dataset: TalkPlayData-Challenge

- **Source:** HuggingFace → `talkpl-ai/talkplay-data-challenge`
- **Splits:** Train / Development / Blind A (interim leaderboard) / Blind B (final leaderboard)
- **Conversations:** Multi-turn dialogues (avg. 8 turns per session)
- **Tracks:** 1M+ tracks with metadata (name, artist, album, tags, release date)
- **User Profiles:** Demographics + listening history

---

## 📊 Evaluation Metrics

| Category | Metric |
|---|---|
| Retrieval | nDCG@{1, 10, 20} — macro-averaged over sessions & turns |
| Diversity | Catalog Coverage |
| Text Quality | Response Lexical Diversity |

Submissions via **CodaBench** leaderboard.

---

## 🗂️ Project Structure

```
RecSys2026/
├── data/
│   ├── raw/              # Downloaded raw datasets from HuggingFace
│   ├── processed/        # Preprocessed / featurized data
│   └── splits/           # Train/Dev/BlindA/BlindB splits
├── notebooks/            # Exploratory Data Analysis (EDA) notebooks
├── src/
│   ├── models/           # CRS model implementations
│   ├── retrieval/        # BM25, dense retrieval, hybrid
│   ├── evaluation/       # Evaluation utilities (nDCG, diversity)
│   └── utils/            # Helpers, data loaders, config
├── baselines/            # Cloned official baselines
├── experiments/
│   ├── runs/             # Experiment artifacts
│   └── results/          # Evaluation results
├── submissions/          # Formatted submission files
├── configs/              # Hydra / YAML configs
├── tests/                # Unit tests
└── docs/                 # Notes & documentation
```

---

## 🚀 Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Download data
python scripts/download_data.py

# Run BM25 baseline
python src/retrieval/bm25_retrieval.py --config configs/bm25_baseline.yaml

# Evaluate
python src/evaluation/evaluate.py --predictions experiments/results/bm25_preds.json
```

---

## 🔗 Key Links

- [Challenge Website](https://nlp4musa.github.io/music-crs-challenge/)
- [Dataset on HuggingFace](https://huggingface.co/datasets/talkpl-ai/talkplay-data-challenge)
- [Official Baselines](https://github.com/nlp4musa/music-crs-baselines)
- [Evaluator Framework](https://github.com/nlp4musa/music-crs-evaluator)
- [ACM RecSys 2026](https://recsys.acm.org/recsys26/challenge/)

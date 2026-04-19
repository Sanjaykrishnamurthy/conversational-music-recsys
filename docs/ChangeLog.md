# Change Log

All notable changes to this project.

- **2026-04-19** — Fixed `notebooks/01_EDA.ipynb`: rewrote to load data from local parquet files instead of broken `load_split()`, corrected column names to match actual schema
- **2026-04-19** — Fixed `notebooks/02_BM25_Baseline.ipynb`: rebuilt from scratch (was corrupted JSON), now loads local parquet data and runs full BM25 retrieval + nDCG evaluation
- **2026-04-19** — Created `notebooks/03_Deep_EDA.ipynb`: comprehensive EDA covering conversations, goals, tracks, users, embeddings (t-SNE), and cross-dataset joins
- **2026-04-19** — Created `notebooks/04_Simple_Baseline.ipynb`: three baselines (Random, Popularity, CF-BPR embedding similarity) with dev evaluation and Blind-A submission export
- **2026-04-19** — Registered project venv as Jupyter kernel "RecSys 2026" via `ipykernel install`
- **2026-04-19** — Created `.agents/skills/codabench-submission/SKILL.md`: documents exact CodaBench submission JSON format, validation rules, and packaging instructions
- **2026-04-19** — Fixed t-SNE cell in `notebooks/03_Deep_EDA.ipynb`: filter out empty cf-bpr embeddings (len 0) before `np.stack()` to avoid inhomogeneous shape error
- **2026-04-19** — Added change log convention to `AGENTS.md` and seeded `docs/ChangeLog.md`

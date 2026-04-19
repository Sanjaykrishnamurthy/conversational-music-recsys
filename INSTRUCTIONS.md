# INSTRUCTIONS.md — Known Issues & Lessons Learned

> This file captures recurring pitfalls, bugs, and gotchas encountered during development.
> AI assistants and developers should **check here first** when stuck on an issue.
> If you fix a non-obvious bug that could recur, **add it here** so it isn't repeated.

---

## ⚠️ Embedding Data Quality — Empty Arrays (2026-04-19)

**Symptom:**
```
ValueError: setting an array element with a sequence.
The requested array has an inhomogeneous shape after 1 dimensions.
The detected shape was (47071,) + inhomogeneous part.
```

**Root cause:** Track and user embedding columns (`cf-bpr`, `audio-laion_clap`, etc.) contain
**variable-length arrays**. ~1.4% of tracks have **empty embeddings** (length 0 instead of the
expected dimensionality). This also applies to user embeddings for cold-start users.

A NaN-only check (`np.isnan(emb).any()`) **silently passes** empty arrays because
`np.isnan([]).any()` returns `False`. The empty arrays then cause `np.array()` to fail when
building a 2D matrix.

**Correct pattern:**
```python
valid_embs = []
valid_ids = []
for _, row in df.iterrows():
    emb = np.array(row['cf-bpr'])
    if len(emb) > 0 and not np.isnan(emb).any():  # BOTH checks required
        valid_embs.append(emb)
        valid_ids.append(row['track_id'])

emb_matrix = np.array(valid_embs)  # Now safe — all same shape
```

**Incorrect pattern (will crash):**
```python
# BAD — empty arrays slip through the NaN check
if not np.isnan(emb).any():
    valid_embs.append(emb)
```

**Applies to:** All 6 embedding types in `TalkPlayData-Challenge-Track-Embeddings`
and all user embedding splits in `TalkPlayData-Challenge-User-Embeddings`.

---

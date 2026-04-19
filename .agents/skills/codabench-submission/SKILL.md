---
name: codabench-submission
description: CodaBench submission format for the RecSys 2026 Music-CRS Challenge. Use when generating, validating, or debugging submission files for the TalkPlayData-Challenge leaderboard.
---

# CodaBench Submission Format — RecSys 2026 Music-CRS

## Submission File Format

The submission is a **single JSON file** (`predictions.json`) containing an array of prediction objects. Each object represents one turn prediction within a conversation session.

```json
[
  {
    "session_id": "9c37dcd7-d7c2-4686-8541-1e37c4814a09",
    "user_id": "c3233a5c-da6c-42a3-9459-83ee1134e207",
    "turn_number": 1,
    "predicted_track_ids": [
      "95bc6531-ebe5-403e-90f8-4bfa9531cc38",
      "916a3b1e-a91f-404c-91a6-4aab09742298",
      "741f7d83-bf2d-40cc-90c3-882fb790fdee",
      "..."
    ],
    "predicted_response": "Here are some songs you might enjoy."
  }
]
```

## Required Fields

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `session_id` | `string` | Unique session identifier | Must exactly match session IDs from the evaluation dataset |
| `user_id` | `string` | Unique user identifier | Must match the user for that session |
| `turn_number` | `int` | Conversation turn number | Range 1–8 |
| `predicted_track_ids` | `list[string]` | Ordered list of recommended track IDs | Up to 20 UUIDs, ranked by relevance (best first), no duplicates |
| `predicted_response` | `string` | Natural language response from the system | Can be empty string `""` if not generating responses |

## Submission Structure Per Split

### Development Set (local evaluation only)
- **1 entry per session per turn** (turns 1–8 for all sessions)
- 1,000 sessions x 8 turns = **8,000 entries**
- Evaluated locally using `baselines/music-crs-evaluator/evaluate_devset.py`
- NOT uploaded to CodaBench

### Blind A (interim leaderboard)
- **1 entry per session** — only the turn that needs prediction
- The prediction turn is the **last user turn with no corresponding `music` role response**
- 80 sessions = **80 entries**
- Turn numbers vary per session (some are turn 1, others up to turn 8)

### Blind B (final leaderboard)
- Released June 15, 2026
- Same format as Blind A

## How to Identify Prediction Turns (Blind Sets)

Each Blind-A/B session contains conversation history where some turns already have `music` and `assistant` responses. The turn to predict is determined by:

```python
for c in session['conversations']:
    if c['role'] == 'music':
        turns_with_music.add(c['turn_number'])
    if c['role'] == 'user':
        user_turns.add(c['turn_number'])

predict_turn = user_turns - turns_with_music  # exactly 1 turn per session
```

## Packaging for CodaBench Upload

Package the JSON file into a **ZIP archive** with the JSON named `predictions.json` inside:

```python
import json, zipfile

with open('predictions.json', 'w') as f:
    json.dump(submission, f, ensure_ascii=False)

with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('predictions.json', 'predictions.json')
```

Save to: `submissions/blind_a_<model-name>_<date>.zip`

## Valid Track IDs

All track IDs in `predicted_track_ids` must exist in the track metadata catalog:
- Dataset: `talkpl-ai/TalkPlayData-Challenge-Track-Metadata`
- Local path: `data/raw/TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet`
- 47,071 valid track UUIDs

## Evaluation Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| Retrieval | nDCG@1 | Normalized DCG at rank 1 |
| Retrieval | nDCG@10 | Normalized DCG at rank 10 |
| Retrieval | nDCG@20 | Normalized DCG at rank 20 |
| Diversity | Catalog Coverage | Unique recommended tracks / total catalog size |
| Text Quality | Lexical Diversity (Distinct-2) | Unique bigrams / total bigrams across all responses |

nDCG is macro-averaged: first over turns within a session, then over sessions.

## Baseline Results (Devset)

| Method | nDCG@1 | nDCG@10 | nDCG@20 | Catalog Diversity | Lexical Diversity |
|--------|-------:|--------:|--------:|------------------:|------------------:|
| Random | 0.0000 | 0.0001 | 0.0001 | 0.9652 | 0.0000 |
| Popularity | 0.0005 | 0.0018 | 0.0024 | 0.0004 | 0.0000 |
| LLaMA-1B + BM25 | 0.0098 | 0.0627 | 0.0815 | 0.3795 | 0.2558 |

## Validation Checklist

Before uploading to CodaBench, verify:

- [ ] File is valid JSON (parseable with `json.load()`)
- [ ] Top-level structure is a JSON array `[...]`
- [ ] All 5 required fields present in every entry
- [ ] Correct number of entries (80 for Blind-A, 8000 for devset)
- [ ] Each `predicted_track_ids` list has at most 20 entries
- [ ] No duplicate track IDs within any single `predicted_track_ids` list
- [ ] All track IDs are valid UUIDs from the track metadata catalog
- [ ] Track IDs are ordered by relevance (most relevant first)
- [ ] `session_id` and `turn_number` match the evaluation dataset exactly
- [ ] JSON saved with `ensure_ascii=False`
- [ ] Packaged as ZIP containing `predictions.json`

## Validation Code Snippet

```python
import json
import pandas as pd

# Load submission
with open('predictions.json') as f:
    submission = json.load(f)

# Load valid track IDs
tracks = pd.read_parquet('data/raw/TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet')
valid_ids = set(tracks['track_id'])

errors = []
for i, entry in enumerate(submission):
    # Required keys
    for key in ['session_id', 'user_id', 'turn_number', 'predicted_track_ids', 'predicted_response']:
        if key not in entry:
            errors.append(f'Entry {i}: missing "{key}"')

    tids = entry.get('predicted_track_ids', [])
    if len(tids) > 20:
        errors.append(f'Entry {i}: {len(tids)} tracks (max 20)')
    if len(tids) != len(set(tids)):
        errors.append(f'Entry {i}: duplicate track IDs')
    invalid = [t for t in tids if t not in valid_ids]
    if invalid:
        errors.append(f'Entry {i}: {len(invalid)} invalid track IDs')

if errors:
    print(f'FAILED ({len(errors)} errors)')
    for e in errors[:10]:
        print(f'  {e}')
else:
    print(f'PASSED: {len(submission)} valid entries')
```

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| "Duplicates detected" | Repeated track IDs in `predicted_track_ids` | Deduplicate while preserving rank order |
| "Missing predictions" | Not all sessions/turns covered | Ensure every session has an entry for each required turn |
| KeyError on `session_id` | Wrong field names | Use exact field names from the schema above |
| Invalid track ID | Track UUID not in catalog | Filter recommendations to only include IDs from `all_tracks` parquet |

## Timeline

| Date | Milestone |
|------|-----------|
| 15 April 2026 | Submission system open — Blind A leaderboard live |
| 15 June 2026 | Blind B released, Blind B submission system activated |
| 30 June 2026 | Challenge ends |
| 6 July 2026 | Final leaderboard and winners announced |

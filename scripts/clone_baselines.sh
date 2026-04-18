#!/usr/bin/env bash
# Clone official RecSys 2026 baseline and evaluator repos into baselines/
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$SCRIPT_DIR/../baselines"
mkdir -p "$BASELINES_DIR"

echo "📥 Cloning official baselines..."

# Official baselines
if [ ! -d "$BASELINES_DIR/music-crs-baselines" ]; then
    git clone https://github.com/nlp4musa/music-crs-baselines.git "$BASELINES_DIR/music-crs-baselines"
    echo "✅ Cloned music-crs-baselines"
else
    echo "⏭️  music-crs-baselines already exists, pulling latest..."
    git -C "$BASELINES_DIR/music-crs-baselines" pull
fi

# Official evaluator
if [ ! -d "$BASELINES_DIR/music-crs-evaluator" ]; then
    git clone https://github.com/nlp4musa/music-crs-evaluator.git "$BASELINES_DIR/music-crs-evaluator"
    echo "✅ Cloned music-crs-evaluator"
else
    echo "⏭️  music-crs-evaluator already exists, pulling latest..."
    git -C "$BASELINES_DIR/music-crs-evaluator" pull
fi

echo ""
echo "🎉 Done! Repos available in: $BASELINES_DIR"

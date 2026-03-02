#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[1/4] Rebuilding outputs_only.tgz..."
cat BestModel/model_archive/outputs_only.tgz.part_* > outputs_only.tgz

echo "[2/4] Verifying checksum..."
sha256sum -c outputs_only.tgz.sha256

echo "[3/4] Extracting to ./models ..."
mkdir -p models
tar -xzf outputs_only.tgz -C models

echo "[4/4] Cleaning up..."
rm -f outputs_only.tgz

echo "✅ Models restored under: $REPO_ROOT/models"

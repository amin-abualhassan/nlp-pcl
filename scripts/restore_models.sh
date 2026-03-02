#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARCH_DIR="BestModel/model_archive"

# Folder that contains either:
# - outputs_only.tgz.part_* + outputs_only.tgz.sha256
# OR a pre-extracted models/ directory (see fallback logic below).
GDRIVE_FOLDER_URL="${GDRIVE_FOLDER_URL:-https://drive.google.com/drive/folders/1FAUTcZRYLRKM-L8iZt_-mGZiH9xzGy89?usp=sharing}"

is_lfs_pointer() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  head -n 1 "$f" 2>/dev/null | grep -q "version https://git-lfs.github.com/spec/v1"
}

need_models_fetch() {
  # Need fetch if any part/sha256 is missing OR is an LFS pointer stub.
  local f
  for f in "$ARCH_DIR"/outputs_only.tgz.part_* "$ARCH_DIR"/outputs_only.tgz.sha256; do
    [[ -f "$f" ]] || return 0
    if is_lfs_pointer "$f"; then
      return 0
    fi
  done
  return 1
}

try_git_lfs_pull() {
  if command -v git >/dev/null 2>&1 && command -v git-lfs >/dev/null 2>&1; then
    echo "[LFS] Attempting git lfs pull (may fail if quota exceeded)..."
    git lfs pull --include="${ARCH_DIR}/*" || return 1
    return 0
  fi
  return 1
}

ensure_gdown() {
  if command -v gdown >/dev/null 2>&1; then
    return 0
  fi
  echo "[GDRIVE] gdown not found; installing via pip..."
  if command -v python3 >/dev/null 2>&1; then
    python3 -m pip install -q --user gdown >/dev/null 2>&1 || python3 -m pip install -q gdown
  else
    echo "[GDRIVE] ERROR: python3 not found, cannot install gdown."
    return 1
  fi
  command -v gdown >/dev/null 2>&1
}

download_from_gdrive_folder() {
  ensure_gdown

  echo "[GDRIVE] Downloading model artifacts from Google Drive folder..."
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT

  # Download the folder contents into $tmp
  gdown --folder "$GDRIVE_FOLDER_URL" -O "$tmp"

  # Enable recursive globs
  shopt -s nullglob globstar

  # Case A: folder contains the same split archive + sha256
  local parts=("$tmp"/**/outputs_only.tgz.part_*)
  local sha=("$tmp"/**/outputs_only.tgz.sha256)

  if (( ${#parts[@]} > 0 )) && (( ${#sha[@]} > 0 )); then
    echo "[GDRIVE] Found split archive + sha256; copying into ${ARCH_DIR}/"
    mkdir -p "$ARCH_DIR"
    cp -f "${parts[@]}" "$ARCH_DIR/"
    cp -f "${sha[0]}" "$ARCH_DIR/outputs_only.tgz.sha256"
    return 0
  fi

  # Case B: folder contains pre-extracted models/
  local models_dirs=("$tmp"/**/models)
  if (( ${#models_dirs[@]} > 0 )); then
    echo "[GDRIVE] Found pre-extracted models/; copying into ./models"
    mkdir -p models
    # copy contents of the first found models dir
    cp -a "${models_dirs[0]}"/. models/
    return 0
  fi

  echo "[GDRIVE] ERROR: Drive folder did not contain outputs_only.tgz.part_*+sha256 or a models/ directory."
  echo "[GDRIVE] Folder URL: $GDRIVE_FOLDER_URL"
  return 1
}

echo "[0/4] Checking archive parts under: $ARCH_DIR"
if need_models_fetch; then
  echo "[0/4] Missing parts/sha256 or they are LFS pointer stubs."

  # Try LFS first
  if ! try_git_lfs_pull; then
    echo "[0/4] LFS pull failed; falling back to Google Drive..."
    download_from_gdrive_folder
  fi

  # Re-check after attempting LFS/Drive
  if need_models_fetch; then
    echo "[0/4] ERROR: still missing valid archive parts after LFS/Drive fallback."
    exit 1
  fi
fi

# If models already exist (e.g., Drive provided models/), you can stop here.
if [[ -d "models/outputs" ]] && [[ -n "$(ls -A models/outputs 2>/dev/null || true)" ]]; then
  echo "✅ Models already present under: $REPO_ROOT/models (skipping extraction)"
  exit 0
fi

echo "[1/4] Rebuilding outputs_only.tgz..."
cat "$ARCH_DIR"/outputs_only.tgz.part_* > outputs_only.tgz

echo "[2/4] Verifying checksum..."
sha256sum -c "$ARCH_DIR"/outputs_only.tgz.sha256

echo "[3/4] Extracting to ./models ..."
mkdir -p models
tar -xzf outputs_only.tgz -C models

echo "[4/4] Cleaning up..."
rm -f outputs_only.tgz

echo "✅ Models restored under: $REPO_ROOT/models"


# #!/usr/bin/env bash
# set -euo pipefail

# REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# cd "$REPO_ROOT"

# ARCH="BestModel/model_archive"

# echo "[1/4] Rebuilding outputs_only.tgz..."
# cat "$ARCH"/outputs_only.tgz.part_* > outputs_only.tgz

# echo "[2/4] Verifying checksum..."
# sha256sum -c "$ARCH"/outputs_only.tgz.sha256

# echo "[3/4] Extracting to ./models ..."
# mkdir -p models
# tar -xzf outputs_only.tgz -C models

# echo "[4/4] Cleaning up..."
# rm -f outputs_only.tgz

# echo "✅ Models restored under: $REPO_ROOT/models"
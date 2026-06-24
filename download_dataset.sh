#!/usr/bin/env bash
#
# download_data.sh — Download and extract the DESPOT dataset from Zenodo.
#
# Zenodo record: https://zenodo.org/records/20829559
# DOI: 10.5281/zenodo.20829559
#
# Run this from the ROOT of the DESPOT GitHub repository. It creates a data/
# subdirectory, downloads the requested archives there, verifies their MD5
# checksums, and extracts them in place.
#
# Usage:
#   ./download_data.sh            # essential only: metadata + potentials (~1.9 GB)
#   ./download_data.sh --all      # all 5 archives (~23.7 GB download, ~35 GB extracted)
#   ./download_data.sh --help
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RECORD_BASE="https://zenodo.org/records/20829559/files"
DATA_DIR="data"

# Essential archives (downloaded by default).
ESSENTIAL=(metadata potentials)

# Optional archives (only with --all).
OPTIONAL=(casf_2016 crown_train hiqbind_train)

# MD5 checksums from the Zenodo record (file name -> md5).
declare -A MD5=(
  [metadata.tar.gz]="376915d65ffba5e6fc203ce5d22ed7ad"
  [potentials.tar.gz]="751b3634ddebbacd6c4a324ace7ebb3a"
  [casf_2016.tar.gz]="b433ef3da03ae94731d9bd0b67df8138"
  [crown_train.tar.gz]="c11adb95e319d8b4f1386c8f5e57ef49"
  [hiqbind_train.tar.gz]="877719aa4d58a158e497f202767b8bd1"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
usage() {
  cat <<'EOF'
Usage: ./download_data.sh [OPTION]

Download and extract the DESPOT dataset from Zenodo into ./data/.

Options:
  (no flag)    Download only the essential archives needed to run DESPOT:
                 metadata.tar.gz (~12 MB) and potentials.tar.gz (~1.9 GB).
  --all        Download all 5 archives, additionally including:
                 casf_2016.tar.gz (~4.2 GB), crown_train.tar.gz (~12.5 GB),
                 hiqbind_train.tar.gz (~5.0 GB). ~23.7 GB total download.
  -h, --help   Show this help message and exit.

Run from the root of the DESPOT repository.
EOF
}

log() { printf '[despot] %s\n' "$*"; }
err() { printf '[despot] ERROR: %s\n' "$*" >&2; }

# Pick an available downloader.
if command -v curl >/dev/null 2>&1; then
  DOWNLOADER="curl"
elif command -v wget >/dev/null 2>&1; then
  DOWNLOADER="wget"
else
  err "Neither curl nor wget is installed. Please install one and retry."
  exit 1
fi

# Pick an available md5 tool.
if command -v md5sum >/dev/null 2>&1; then
  md5_of() { md5sum "$1" | awk '{print $1}'; }
elif command -v md5 >/dev/null 2>&1; then
  md5_of() { md5 -q "$1"; }            # macOS / BSD
else
  md5_of() { echo ""; }                # checksum verification skipped
  err "No md5sum/md5 tool found; checksum verification will be skipped."
fi

download() {
  local url="$1" out="$2"
  if [[ "$DOWNLOADER" == "curl" ]]; then
    # -L follow redirects, -C - resume partial downloads, --fail on HTTP errors.
    curl -L -C - --fail --retry 3 -o "$out" "$url"
  else
    # -c continue partial downloads.
    wget -c -O "$out" "$url"
  fi
}

# Download (with resume), verify md5, and extract a single archive.
fetch_and_extract() {
  local name="$1"
  local archive="${name}.tar.gz"
  local url="${RECORD_BASE}/${archive}?download=1"
  local path="${DATA_DIR}/${archive}"
  local expected="${MD5[$archive]:-}"

  # Skip if the extracted target directory already exists.
  if [[ -d "${DATA_DIR}/${name}" ]]; then
    log "${name}/ already exists in ${DATA_DIR}/ — skipping."
    return 0
  fi

  # If a verified archive is already present, reuse it instead of re-downloading.
  if [[ -f "$path" && -n "$expected" ]]; then
    if [[ "$(md5_of "$path")" == "$expected" ]]; then
      log "${archive} already downloaded and verified — skipping download."
    else
      log "${archive} present but checksum mismatch — re-downloading."
      rm -f "$path"
    fi
  fi

  if [[ ! -f "$path" ]]; then
    log "Downloading ${archive} ..."
    download "$url" "$path"
  fi

  # Verify checksum.
  if [[ -n "$expected" ]]; then
    log "Verifying ${archive} checksum ..."
    local actual
    actual="$(md5_of "$path")"
    if [[ "$actual" != "$expected" ]]; then
      err "Checksum mismatch for ${archive}!"
      err "  expected: ${expected}"
      err "  actual:   ${actual}"
      err "Delete ${path} and retry."
      exit 1
    fi
    log "Checksum OK for ${archive}."
  fi

  log "Extracting ${archive} into ${DATA_DIR}/ ..."
  tar -xzf "$path" -C "$DATA_DIR"

  log "Done with ${name}."
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
MODE="essential"
case "${1:-}" in
  --all)        MODE="all" ;;
  -h|--help)    usage; exit 0 ;;
  "")           MODE="essential" ;;
  *)            err "Unknown option: $1"; echo; usage; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# Sanity check: warn if this doesn't look like the repo root.
if [[ ! -e "environment.yml" && ! -e "pyproject.toml" && ! -e "setup.py" ]]; then
  err "This doesn't look like the DESPOT repo root (no environment.yml/pyproject.toml/setup.py)."
  err "Run the script from the root of the cloned repository."
  exit 1
fi

mkdir -p "$DATA_DIR"

TARGETS=("${ESSENTIAL[@]}")
if [[ "$MODE" == "all" ]]; then
  TARGETS+=("${OPTIONAL[@]}")
  log "Mode: ALL — downloading all 5 archives (~23.7 GB)."
else
  log "Mode: ESSENTIAL — downloading metadata + potentials only."
  log "(Use --all to also fetch casf_2016, crown_train, and hiqbind_train.)"
fi

for name in "${TARGETS[@]}"; do
  fetch_and_extract "$name"
done

log "All requested archives downloaded and extracted into ${DATA_DIR}/."

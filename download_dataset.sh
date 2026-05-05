#!/bin/bash

# ─── Configuration ───────────────────────────────────────────────────────────
ZENODO_RECORD="19218614"
BASE_URL="https://zenodo.org/api/records/${ZENODO_RECORD}/files"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Argument parsing ────────────────────────────────────────────────────────
WITH_CASF=0
OUTPUT_DIR="."

usage() {
    cat <<EOF
Usage: $0 [--with-casf] [OUTPUT_DIR]

Options:
  --with-casf    Also download the CASF-2016 benchmark (large; slow).
                 Off by default to save bandwidth and disk space.
  -h, --help     Show this help message and exit.

Arguments:
  OUTPUT_DIR     Target directory (default: current working directory).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-casf) WITH_CASF=1; shift ;;
        -h|--help)   usage; exit 0 ;;
        --) shift; break ;;
        -*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
        *)  OUTPUT_DIR="$1"; shift ;;
    esac
done

# ─── Target subdirectories (adjust these to match your repo layout) ──────────
DIR_DATA="${OUTPUT_DIR}/data"
DIR_METADATA="${OUTPUT_DIR}/data/metadata"
DIR_MODELS="${OUTPUT_DIR}/data/potentials"
DIR_CASF="${OUTPUT_DIR}/data"

# ─── MD5 checksums from Zenodo ───────────────────────────────────────────────
# Core files (always downloaded)
declare -A CHECKSUMS=(
    ["atom_type_counts.csv"]="6e55bab544fb69221c5603f617355643"
    ["casf_pdb_ids.txt"]="67f958b184c12862481268ba24fb080a"
    ["despot_counts_crown_leaky.npz"]="9e0900a27fd254a065fbc7bc0f10aebe"
    ["despot_counts_crown_train.npz"]="375907b82a312b0740e02deddad5d7cb"
    ["despot_counts_crown_xtal.npz"]="be3c8e6e322511461c18bff5ac160c5b"
    ["despot_ds_scores_crown_leaky.npz"]="eb9a909c611fe5a5bb911aebadb7f0a4"
    ["despot_ds_scores_crown_train.npz"]="be3665875ee5a76ae4fba0c92e54c5d1"
    ["despot_ds_scores_crown_xtal.npz"]="36ec48834625d5841ec698ec9ba7dd0c"
    ["despot_scores_crown_leaky.npz"]="529fd7056351929386ccadd6982b5873"
    ["despot_scores_crown_train.npz"]="ca5d00f3c0d57bd124ddbd96770f5bda"
    ["despot_scores_crown_xtal.npz"]="4d2d4f80d536001aee0e80a7a40e95c9"
)

# Optional benchmark
CASF_FILE="casf_2016.tar.gz"
CASF_MD5="daa9cb4088844c28d179f3c1e5e5be3c"

# ─── Helpers ─────────────────────────────────────────────────────────────────
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
fail()  { echo -e "\033[1;31m[FAIL]\033[0m  $*"; exit 1; }

verify_md5() {
    local file="$1" expected="$2"
    local actual
    actual=$(md5sum "$file" | awk '{print $1}')
    if [[ "$actual" == "$expected" ]]; then
        ok "$(basename "$file")"
    else
        fail "Checksum mismatch for $(basename "$file"): expected $expected, got $actual"
    fi
}

download_and_verify() {
    local filename="$1" expected_md5="$2" dest="$3"
    info "Downloading $filename..."
    wget --progress=bar:force:noscroll -O "$dest" "${BASE_URL}/${filename}/content" \
        || fail "Download failed for $filename"
    verify_md5 "$dest" "$expected_md5"
}

# ─── Step 1: Setup ──────────────────────────────────────────────────────────
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

info "Creating directory structure..."
mkdir -p "$DIR_MODELS" "$DIR_METADATA"

# ─── Step 2: Download core files individually ───────────────────────────────
info "Downloading files from Zenodo record ${ZENODO_RECORD}..."
for file in "${!CHECKSUMS[@]}"; do
    download_and_verify "$file" "${CHECKSUMS[$file]}" "${TMPDIR}/${file}"
done

# ─── Step 3: Optionally download CASF-2016 ──────────────────────────────────
if [[ "$WITH_CASF" -eq 1 ]]; then
    download_and_verify "$CASF_FILE" "$CASF_MD5" "${TMPDIR}/${CASF_FILE}"
else
    warn "Skipping ${CASF_FILE} (pass --with-casf to include it)"
fi

ok "All requested files downloaded and verified."

# ─── Step 4: Move files to their target locations ───────────────────────────
SRC="$TMPDIR"

info "Installing files..."

# Definition / metadata files → data/metadata/
mv "$SRC/atom_type_counts.csv" "$DIR_METADATA/"
mv "$SRC/casf_pdb_ids.txt"     "$DIR_METADATA/"

# Model files (counts + scores) → data/potentials/
mv "$SRC"/despot_counts_*.npz    "$DIR_MODELS/"
mv "$SRC"/despot_scores_*.npz    "$DIR_MODELS/"
mv "$SRC"/despot_ds_scores_*.npz "$DIR_MODELS/"

# CASF-2016 benchmark → data/
if [[ "$WITH_CASF" -eq 1 ]]; then
    mv "$SRC/$CASF_FILE" "${DIR_CASF}/"
fi

ok "All files installed."

# ─── Summary ────────────────────────────────────────────────────────────────
echo ""
info "Installation summary:"
echo "  ${DIR_METADATA}/atom_type_counts.csv"
echo "  ${DIR_METADATA}/casf_pdb_ids.txt"
echo "  ${DIR_MODELS}/despot_counts_*.npz       (3 files)"
echo "  ${DIR_MODELS}/despot_scores_*.npz       (3 files)"
echo "  ${DIR_MODELS}/despot_ds_scores_*.npz    (3 files)"
if [[ "$WITH_CASF" -eq 1 ]]; then
    echo "  ${DIR_CASF}/${CASF_FILE}"
else
    echo "  (CASF-2016 benchmark skipped — re-run with --with-casf to fetch it)"
fi

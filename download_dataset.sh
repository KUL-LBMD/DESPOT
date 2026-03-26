#!/bin/bash

# ─── Configuration ───────────────────────────────────────────────────────────
ZENODO_RECORD="19218614"
ARCHIVE_URL="https://zenodo.org/api/records/${ZENODO_RECORD}/files-archive"
ARCHIVE_NAME="despot_zenodo.zip"
 
# Default output directory = repo root (script assumes it lives in the repo)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-.}"  # Pass a custom output dir as first arg, or default to cwd

# ─── Target subdirectories (adjust these to match your repo layout) ──────────
DIR_DATA="${OUTPUT_DIR}/data"
DIR_METADATA="${OUTPUT_DIR}/data/metadata"
DIR_MODELS="${OUTPUT_DIR}/data/potentials"
DIR_CASF="${OUTPUT_DIR}/data"

# ─── MD5 checksums from Zenodo ───────────────────────────────────────────────
declare -A CHECKSUMS=(
    ["atom_type_counts.csv"]="6e55bab544fb69221c5603f617355643"
    ["casf_2016.tar.gz"]="daa9cb4088844c28d179f3c1e5e5be3c"
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

# ─── Helpers ─────────────────────────────────────────────────────────────────
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
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

# ─── Step 1: Download the archive ───────────────────────────────────────────
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT
 
info "Downloading archive from Zenodo record ${ZENODO_RECORD}..."
wget --progress=bar:force:noscroll -O "${TMPDIR}/${ARCHIVE_NAME}" "$ARCHIVE_URL"
ok "Download complete."
 
# ─── Step 2: Extract the zip ────────────────────────────────────────────────
info "Extracting archive..."
unzip -o "${TMPDIR}/${ARCHIVE_NAME}" -d "${TMPDIR}/extracted"
ok "Extraction complete."
 
# ─── Step 3: Verify checksums ───────────────────────────────────────────────
info "Verifying file integrity..."
for file in "${!CHECKSUMS[@]}"; do
    filepath="${TMPDIR}/extracted/${file}"
    if [[ -f "$filepath" ]]; then
        verify_md5 "$filepath" "${CHECKSUMS[$file]}"
    else
        fail "Expected file not found: $file"
    fi
done
ok "All checksums verified."

# ─── Step 4: Create target directories ──────────────────────────────────────
info "Creating directory structure..."
mkdir -p "$DIR_MODELS" "$DIR_METADATA/"

# ─── Step 5: Move files to their target locations ───────────────────────────
SRC="${TMPDIR}/extracted"
 
info "Installing files..."
 
# Definition / metadata files → data/
mv "$SRC/atom_type_counts.csv" "$DIR_METADATA/"
mv "$SRC/casf_pdb_ids.txt"     "$DIR_METADATA/"

# Model files (counts + scores) → data/models/
mv "$SRC"/despot_counts_*.npz    "$DIR_MODELS/"
mv "$SRC"/despot_scores_*.npz    "$DIR_MODELS/"
mv "$SRC"/despot_ds_scores_*.npz "$DIR_MODELS/"

# CASF-2016 benchmark → extract into data/casf/
mv "$SRC/casf_2016.tar.gz" "${OUTPUT_DIR}/data/"
ok "All files installed."

echo ""
info "Installation summary:"
echo "  ${DIR_METADATA}/atom_type_counts.csv"
echo "  ${DIR_METADATA}/casf_pdb_ids.txt"
echo "  ${DIR_MODELS}/despot_counts_*.npz       (3 files)"
echo "  ${DIR_MODELS}/despot_scores_*.npz       (3 files)"
echo "  ${DIR_MODELS}/despot_ds_scores_*.npz    (3 files)"
echo "  ${DIR_CASF}/                            (extracted benchmark)"

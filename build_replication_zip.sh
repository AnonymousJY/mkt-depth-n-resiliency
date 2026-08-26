#!/usr/bin/env bash
#
# build_replication_zip.sh
#
# Builds a JBF-ready replication zip from the current repository.
#
# What is included:
#   - All source code (Library/, Scripts/)
#   - Public spot price snapshots (data/snapshots/prices_*.csv)
#   - Cached P-MLE parameter estimates (Study/Estimated Parameters PMLE/)
#   - Cached Q-calibrated parameters (Study/Estimated Parameters QLSQ/)
#   - Cached vol surface outputs (Study/Vol Surface From Model/)
#   - Study/Collar Asian/ notebooks and generated PDFs (Figures 1, 2, 6)
#   - Skew calibration output PDFs (Scripts/SPX_VOL_SKEW_*.pdf,
#     Scripts/COIN_VOL_SKEW_*.pdf)
#   - Environment files (environment.yml, requirements.txt)
#   - Documentation (README.md, DATA_AVAILABILITY.md, LICENSE)
#
# What is EXCLUDED (per DATA_AVAILABILITY.md):
#   - Raw Cboe option chain data (proprietary; not present in repo anyway)
#   - Any temporary or user-specific files (__pycache__, .ipynb_checkpoints,
#     .DS_Store, .git, .venv, .env, etc.)
#   - The scratch pad notebook
#
# Usage:
#   ./build_replication_zip.sh                      # default output name
#   ./build_replication_zip.sh my_zip_name.zip      # custom output name
#
# Requires: zip (standard on macOS / Linux).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

OUTPUT="${1:-ald_replication_package.zip}"
STAGING_DIR="/tmp/ald_replication_staging_$$"

echo "Building replication zip at: ${OUTPUT}"
echo "Staging in: ${STAGING_DIR}"

# Clean staging.
rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}/ald_replication"

# Copy required content.
copy_if_exists() {
    local src="$1"
    if [ -e "${src}" ]; then
        cp -R "${src}" "${STAGING_DIR}/ald_replication/"
        echo "  + ${src}"
    else
        echo "  - MISSING (skipped): ${src}"
    fi
}

echo "Copying content:"
copy_if_exists Library
copy_if_exists Scripts
copy_if_exists Study
copy_if_exists data
copy_if_exists environment.yml
copy_if_exists requirements.txt
copy_if_exists README.md
copy_if_exists DATA_AVAILABILITY.md
copy_if_exists LICENSE

# Purge excluded items from the staged copy.
echo "Removing excluded artifacts from staging:"
find "${STAGING_DIR}/ald_replication" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find "${STAGING_DIR}/ald_replication" -type d -name '.ipynb_checkpoints' -exec rm -rf {} + 2>/dev/null || true
find "${STAGING_DIR}/ald_replication" -type d -name '.git' -exec rm -rf {} + 2>/dev/null || true
find "${STAGING_DIR}/ald_replication" -type f -name '.DS_Store' -delete 2>/dev/null || true
find "${STAGING_DIR}/ald_replication" -type f -name '*.pyc' -delete 2>/dev/null || true
find "${STAGING_DIR}/ald_replication" -type f -name 'scratch_pad.ipynb' -delete 2>/dev/null || true

# Sanity check: no raw Cboe option data should be present.
if [ -d "${STAGING_DIR}/ald_replication/data/options" ]; then
    # Only the README.md placeholder is allowed under data/options.
    find "${STAGING_DIR}/ald_replication/data/options" -type f ! -name 'README.md' -delete
    echo "  Cleared any files under data/options/ except README.md."
fi

# Remove all legacy .pkl caches under Study/. The pipeline now writes and
# reads parquet via Library.Serialization; shipping pickle files in a JBF
# replication package is not portable (pickle is Python-version-dependent
# and can execute arbitrary code on load). The parquet equivalents are the
# canonical format going forward.
find "${STAGING_DIR}/ald_replication/Study" -type f -name '*.pkl' -delete 2>/dev/null || true
echo "  Removed all .pkl caches under Study/."

# Verify no CBOE-proprietary raw data slipped in.
echo "Checking for accidental inclusion of proprietary raw option data..."
SUSPECT=$(find "${STAGING_DIR}/ald_replication" -type f \
    \( -name 'cboe_*' -o -name 'livevol_*' -o -name '*_options_raw*' \) 2>/dev/null || true)
if [ -n "${SUSPECT}" ]; then
    echo "WARNING: possibly proprietary files detected:"
    echo "${SUSPECT}"
    echo "Review before shipping the zip."
fi

# Create the zip.
echo "Compressing to ${OUTPUT}..."
rm -f "${OUTPUT}"
(cd "${STAGING_DIR}" && zip -r "${REPO_ROOT}/${OUTPUT}" ald_replication > /dev/null)

# Clean up.
rm -rf "${STAGING_DIR}"

SIZE=$(du -h "${OUTPUT}" | cut -f1)
FILE_COUNT=$(unzip -l "${OUTPUT}" | tail -1 | awk '{print $2}')

echo ""
echo "Done."
echo "Output:     ${OUTPUT}"
echo "Size:       ${SIZE}"
echo "File count: ${FILE_COUNT}"
echo ""
echo "Verify contents with:"
echo "  unzip -l ${OUTPUT}"

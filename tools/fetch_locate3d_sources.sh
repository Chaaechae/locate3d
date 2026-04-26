#!/usr/bin/env bash
#
# Pull the missing Python sources from facebookresearch/locate-3d into the
# local ``locate-3d/`` checkout so ``tools/eval_locate3d_baseline.py`` can
# import Meta's model + dataset code (Locate3DDataset, Locate3D, etc.).
#
# Idempotent: skips files that already exist locally.
#
# Usage from repo root:
#   bash tools/fetch_locate3d_sources.sh

set -euo pipefail

DEST_ROOT="locate-3d"
RAW_BASE="https://raw.githubusercontent.com/facebookresearch/locate-3d/main"

# Files we need for Meta-side inference + dataset loading.
FILES=(
    "locate3d_data/locate3d_dataset.py"
    "locate3d_data/scannet_dataset.py"
    "locate3d_data/scannetpp_dataset.py"
    "locate3d_data/data_utils.py"
    "locate3d_data/vis_utils.py"
    "models/locate_3d.py"
    "models/locate_3d_decoder.py"
    "models/encoder_3djepa.py"
    "models/point_transformer_v3.py"
    "models/model_utils/__init__.py"
    "models/model_utils/bbox_utils.py"
)

mkdir -p "${DEST_ROOT}"

for f in "${FILES[@]}"; do
    target="${DEST_ROOT}/${f}"
    if [[ -f "${target}" ]]; then
        echo "[skip] ${target}"
        continue
    fi
    mkdir -p "$(dirname "${target}")"
    url="${RAW_BASE}/${f}"
    echo "[fetch] ${url}"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "${url}" -o "${target}"
    else
        wget -q -O "${target}" "${url}"
    fi
    if head -c 64 "${target}" 2>/dev/null | grep -qi "<html\|not found"; then
        echo "[error] ${f} returned an HTML/error page; removing"
        rm -f "${target}"
        exit 1
    fi
done

# Meta uses namespace-package style (no __init__.py) for ``models/`` and
# ``locate3d_data/`` -- but Python 3 implicit namespace packages only
# discover sub-modules if the parent dir is on sys.path. Our eval script
# already inserts ``locate-3d/`` into sys.path, so the imports
#     from models.locate_3d import Locate3D
#     from locate3d_data.locate3d_dataset import Locate3DDataset
# work without explicit __init__.py.

echo
echo "Done. Files now under: ${DEST_ROOT}/"
find "${DEST_ROOT}/locate3d_data" "${DEST_ROOT}/models" -type f -name "*.py" | sort

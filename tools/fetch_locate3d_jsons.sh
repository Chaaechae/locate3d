#!/usr/bin/env bash
#
# Download the Locate-3D annotation JSONs for ScanNet and ScanNet++ from
# the official Meta repo and drop them alongside the ARKit JSONs in
# ``locate-3d/locate3d_data/``. This is a no-op if the files already
# exist locally (e.g. the user copied them over manually).
#
# Usage from repo root:
#   bash tools/fetch_locate3d_jsons.sh
#
# After this, the 0h-combined config can find
# ``locate-3d/locate3d_data/{train,val}_{scannet,scannetpp}.json``
# without any further path edits.

set -euo pipefail

DEST_DIR="locate-3d/locate3d_data"
RAW_BASE="https://raw.githubusercontent.com/facebookresearch/locate-3d/main/locate3d_data/dataset"

mkdir -p "${DEST_DIR}"

for name in train_scannet.json val_scannet.json \
            train_scannetpp.json val_scannetpp.json; do
    target="${DEST_DIR}/${name}"
    if [[ -f "${target}" ]]; then
        echo "[skip] ${target} (already exists)"
        continue
    fi
    url="${RAW_BASE}/${name}"
    echo "[fetch] ${url}"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "${url}" -o "${target}"
    else
        wget -q -O "${target}" "${url}"
    fi
    # sanity: confirm we didn't get a 404 HTML page
    if [[ ! -s "${target}" ]] || \
       head -c 64 "${target}" | grep -qi "<html\|not found"; then
        echo "[error] ${name} did not download a real JSON file; removing"
        rm -f "${target}"
        exit 1
    fi
done

echo
echo "Done. Annotation JSONs are in: ${DEST_DIR}"
ls -1 "${DEST_DIR}"/*.json

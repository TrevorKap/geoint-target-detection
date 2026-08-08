#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# download_data.sh — fetch sample training data for the Tactical GEOINT Analyzer
#
# All sources below are public and download over plain HTTPS (curl) — no AWS CLI,
# no Python, no account required. xView is the exception: it needs a manual,
# authenticated download (see the note at the bottom).
#
# Usage:
#   bash scripts/download_data.sh dota8            # ~1.3 MB  tiny OBB smoke test
#   bash scripts/download_data.sh rareplanes-test  # ~0.9 GB  real aircraft tiles
#   bash scripts/download_data.sh rareplanes-train # ~1.9 GB  real aircraft tiles
#   bash scripts/download_data.sh dota-v1          # ~1.9 GB  full DOTAv1 (YOLO-OBB)
#   bash scripts/download_data.sh all-samples      # dota8 + rareplanes-test
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data/raw"

RP_BASE="https://rareplanes-public.s3.amazonaws.com/real"

# Download $1 -> $2 (skips if already present and non-empty).
fetch() {
  local url="$1" out="$2"
  if [[ -s "$out" ]]; then
    echo "  ✓ exists, skipping: $(basename "$out")"
    return
  fi
  echo "  ↓ $url"
  curl -fL --progress-bar "$url" -o "$out"
}

# Extract a .tar.gz or .zip into a directory.
#   .zip     -> unzip (GNU tar can't read zip); .tar.gz -> tar.
# Falls back to Windows' bundled bsdtar (System32/tar.exe), which handles both.
extract() {
  local archive="$1" dest="$2"
  mkdir -p "$dest"
  echo "  ⤷ extracting $(basename "$archive") -> $dest"
  case "$archive" in
    *.zip)
      if command -v unzip >/dev/null 2>&1; then
        unzip -o -q "$archive" -d "$dest"
      else
        /c/Windows/System32/tar.exe -xf "$archive" -C "$dest"
      fi
      ;;
    *.tar.gz | *.tgz | *.tar)
      tar -xf "$archive" -C "$dest"
      ;;
    *)
      echo "  ! unknown archive type: $archive" >&2
      return 1
      ;;
  esac
}

dota8() {
  echo "[dota8] tiny 8-image DOTA sample in YOLO-OBB format (~1.3 MB)"
  local dst="$DATA/dota"
  fetch "https://github.com/ultralytics/assets/releases/download/v0.0.0/dota8.zip" "$dst/dota8.zip"
  extract "$dst/dota8.zip" "$dst"
}

dota_v1() {
  echo "[dota-v1] full DOTAv1, images + OBB labels, YOLO-OBB format (~1.9 GB)"
  local dst="$DATA/dota"
  fetch "https://github.com/ultralytics/yolov5/releases/download/v1.0/DOTAv1.zip" "$dst/DOTAv1.zip"
  extract "$dst/DOTAv1.zip" "$dst"
}

rareplanes_test() {
  echo "[rareplanes-test] real aircraft: pre-tiled PS-RGB + tiled GeoJSON + COCO (~0.9 GB)"
  local dst="$DATA/rareplanes"
  fetch "$RP_BASE/tarballs/test/RarePlanes_test_PS-RGB_tiled.tar.gz"        "$dst/test_PS-RGB_tiled.tar.gz"
  fetch "$RP_BASE/tarballs/test/RarePlanes_test_geojson_aircraft_tiled.tar.gz" "$dst/test_geojson_aircraft_tiled.tar.gz"
  fetch "$RP_BASE/metadata_annotations/RarePlanes_Test_Coco_Annotations_tiled.json" "$dst/RarePlanes_Test_Coco_Annotations_tiled.json"
  extract "$dst/test_PS-RGB_tiled.tar.gz"           "$dst/test"
  extract "$dst/test_geojson_aircraft_tiled.tar.gz" "$dst/test"
}

rareplanes_train() {
  echo "[rareplanes-train] real aircraft: pre-tiled PS-RGB + tiled GeoJSON + COCO (~1.9 GB)"
  local dst="$DATA/rareplanes"
  fetch "$RP_BASE/tarballs/train/RarePlanes_train_PS-RGB_tiled.tar.gz"        "$dst/train_PS-RGB_tiled.tar.gz"
  fetch "$RP_BASE/tarballs/train/RarePlanes_train_geojson_aircraft_tiled.tar.gz" "$dst/train_geojson_aircraft_tiled.tar.gz"
  fetch "$RP_BASE/metadata_annotations/RarePlanes_Train_Coco_Annotations_tiled.json" "$dst/RarePlanes_Train_Coco_Annotations_tiled.json"
  extract "$dst/train_PS-RGB_tiled.tar.gz"           "$dst/train"
  extract "$dst/train_geojson_aircraft_tiled.tar.gz" "$dst/train"
}

case "${1:-}" in
  dota8)            dota8 ;;
  dota-v1)          dota_v1 ;;
  rareplanes-test)  rareplanes_test ;;
  rareplanes-train) rareplanes_train ;;
  all-samples)      dota8; rareplanes_test ;;
  *)
    echo "Usage: bash scripts/download_data.sh {dota8|rareplanes-test|rareplanes-train|dota-v1|all-samples}"
    exit 1
    ;;
esac

echo ""
echo "Done. Data under: $DATA"
echo ""
echo "── xView (manual) ───────────────────────────────────────────────────────"
echo "xView requires a free account + authenticated download; it cannot be"
echo "scripted. Register and download from https://challenge.xviewdataset.org/"
echo "then unpack into: $DATA/xview"

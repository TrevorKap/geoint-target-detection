# Training Data

Datasets are **not** committed to git — download them with the script:

```bash
bash scripts/download_data.sh all-samples   # dota8 (~1.3 MB) + rareplanes-test (~0.9 GB)
```

## Sources & layout

| Dataset | Command | Size | Format | Why it's here |
|---|---|---|---|---|
| **DOTA8** | `dota8` | ~1.3 MB | YOLO-OBB (img + `.txt`) | Instant smoke test for the oriented-bbox pipeline |
| **RarePlanes (test)** | `rareplanes-test` | ~0.9 GB | PS-RGB tiles + GeoJSON + COCO | Real aircraft, geospatial labels, fine-grained roles |
| **RarePlanes (train)** | `rareplanes-train` | ~1.9 GB | PS-RGB tiles + GeoJSON + COCO | Full real training split |
| **DOTAv1 (full)** | `dota-v1` | ~1.9 GB | YOLO-OBB | 15-class oriented detection at scale |
| **xView** | *manual* | ~20+ GB | GeoTIFF + GeoJSON | 60-class NGA benchmark — needs account |

```
data/raw/
├─ rareplanes/
│  ├─ test/            # PS-RGB .png tiles + *_geojson_aircraft tiled labels
│  └─ RarePlanes_Test_Coco_Annotations_tiled.json
├─ dota/               # images/ + labels/ (YOLO-OBB), or dota8/
└─ xview/              # (manual) train_images/ + xView_train.geojson
```

## Licensing / usage notes

- **RarePlanes** — CC BY 4.0 (CosmiQ Works / AI.Reverie). Public AWS Open Data.
- **DOTA** — academic / non-commercial use per the DOTA terms.
- **xView** — see the challenge rules; account required to download.

Keep this in mind for any public portfolio/LinkedIn framing: cite the datasets.

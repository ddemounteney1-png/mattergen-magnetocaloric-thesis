import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import ANALYSIS_WORK, LOSS_CURVES  # noqa: E402

import re
import csv
import os

LOG_PATH = f"{ANALYSIS_WORK}/.out_files/pretrain_4.out"
OUT_CSV  = f"{LOSS_CURVES}/pretrain_losses.csv"

EPOCH_RE = re.compile(
    r'Epoch (\d+):\s+100%.*?'
    r'loss_train_epoch=([\d.]+).*?'
    r'pos_train_epoch=([\d.]+).*?'
    r'cell_train_epoch=([\d.]+).*?'
    r'atomic_numbers_train_epoch=([\d.]+)'
)

records: dict[int, dict] = {}

with open(LOG_PATH, "r", errors="replace") as fh:
    for raw_line in fh:
        for chunk in raw_line.split("\r"):
            m = EPOCH_RE.search(chunk)
            if m:
                epoch = int(m.group(1))
                records[epoch] = {
                    "epoch": epoch,
                    "loss_train_epoch": float(m.group(2)),
                    "pos_train_epoch": float(m.group(3)),
                    "cell_train_epoch": float(m.group(4)),
                    "atomic_numbers_train_epoch": float(m.group(5)),
                }

sorted_records = [records[k] for k in sorted(records)]

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "epoch", "loss_train_epoch",
        "pos_train_epoch", "cell_train_epoch", "atomic_numbers_train_epoch"
    ])
    writer.writeheader()
    writer.writerows(sorted_records)

losses = [r["loss_train_epoch"] for r in sorted_records]
print(f"Extracted {len(sorted_records)} epochs → {OUT_CSV}")
print(f"Epoch range  : {sorted_records[0]['epoch']} – {sorted_records[-1]['epoch']}")
print(f"Loss: {losses[0]:.4f} → {losses[-1]:.4f}  |  min={min(losses):.4f}  max={max(losses):.4f}")

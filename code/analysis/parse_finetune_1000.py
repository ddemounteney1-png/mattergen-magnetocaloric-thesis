import re
import csv
import os
from datetime import date

LOG_PATH = "/home/daniel/Data_analysis/.out_files/finetune_1000_epoch.out"
OUT_CSV  = "/home/daniel/Data_analysis/finetune_1000_losses.csv"

# Matches the FINAL entry per epoch: "Epoch N: 100%|..." with loss_train_epoch present
# The 100%-complete entry that also has loss_train_epoch is the authoritative one.
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
        # tqdm writes \r between updates; split on \r to get individual "frames"
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

os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "epoch", "loss_train_epoch",
        "pos_train_epoch", "cell_train_epoch", "atomic_numbers_train_epoch"
    ])
    writer.writeheader()
    writer.writerows(sorted_records)

print(f"Extracted {len(sorted_records)} epochs → {OUT_CSV}")
if sorted_records:
    first, last = sorted_records[0], sorted_records[-1]
    print(f"Epoch range : {first['epoch']} – {last['epoch']}")
    print(f"Loss range  : {first['loss_train_epoch']:.4f} → {last['loss_train_epoch']:.4f}")
    losses = [r["loss_train_epoch"] for r in sorted_records]
    print(f"Min loss    : {min(losses):.4f}   Max loss: {max(losses):.4f}")

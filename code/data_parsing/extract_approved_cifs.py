"""
extract_approved_cifs.py

Reads cif_quality_report.csv and copies the approved CIFs into a single
clean output folder ready for CHGNet relaxation.

Rules applied (matching the decisions made in analysis):
  PASS  → always copy (use after_cif)
  WARN  → copy with per-formula overrides (see EXCLUDED_WARN below)
  FAIL  → never copy
  SKIP  → never copy

MnCoGe special case: it is a FAIL due to SQS raising symmetry, but the
ordered before_cif is fine — copy that instead.

Run from anywhere:
    python extract_approved_cifs.py

Output folder is created next to the script if it does not exist.
A log file (extraction_log.txt) is written alongside the script.
"""

import os
import shutil
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

CSV_PATH    = "/home/daniel/Pre-mattergen_dataprocessing/cif_quality_report.csv"
OUTPUT_DIR  = "/home/daniel/Pre-mattergen_dataprocessing/approved_cifs"
LOG_PATH    = "/home/daniel/Pre-mattergen_dataprocessing/extraction_log.txt"

# WARN rows to exclude (per manual review decisions)
EXCLUDED_WARN = {
    "Tb(Co0.95Fe0.5)2",    # 7.7% stoich, previously confirmed bad
    "La0.75Ag0.25MnO3",    # previously excluded in manual review
    "Mn2.97Co0.03GaC",     # Co below SQS resolution — missing element
    "Mn2.95Co0.05GaC",     # Co below SQS resolution — missing element
}

# FAIL rows to rescue (use before_cif, which is the ordered structure)
RESCUED_FAIL = {
    "MnCoGe",   # SQS erroneously raised SG; ordered structure is correct
}

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    copied      = []
    skipped     = []
    not_found   = []
    rescued     = []

    for _, row in df.iterrows():
        formula = str(row["formula"]).strip()
        status  = str(row["status"]).strip()
        after   = str(row["after_cif"]).strip()
        before  = str(row["before_cif"]).strip()

        # ── Decision logic ────────────────────────────────────────────────────

        if status == "PASS":
            src = after

        elif status == "WARN":
            if formula in EXCLUDED_WARN:
                skipped.append((formula, status, f"manually excluded WARN"))
                continue
            src = after

        elif status == "FAIL":
            if formula in RESCUED_FAIL:
                src = before   # use ordered CIF, not the broken SQS one
                rescued.append(formula)
            else:
                skipped.append((formula, status, "FAIL — not rescued"))
                continue

        elif status == "SKIP":
            skipped.append((formula, status, "no CIF found by quality check"))
            continue

        else:
            skipped.append((formula, status, f"unknown status '{status}'"))
            continue

        # ── Copy ──────────────────────────────────────────────────────────────

        if not os.path.isfile(src):
            not_found.append((formula, src))
            continue

        dest = os.path.join(OUTPUT_DIR, os.path.basename(src))

        # Handle filename collisions (different source dirs, same basename)
        if os.path.exists(dest) and os.path.abspath(src) != os.path.abspath(dest):
            base, ext = os.path.splitext(os.path.basename(src))
            dest = os.path.join(OUTPUT_DIR, f"{base}_dup{ext}")

        shutil.copy2(src, dest)
        copied.append((formula, os.path.basename(dest)))

    # ── Report ────────────────────────────────────────────────────────────────

    lines = []
    lines.append("=" * 68)
    lines.append("EXTRACTION REPORT")
    lines.append("=" * 68)
    lines.append(f"  Output folder : {OUTPUT_DIR}")
    lines.append(f"  Source CSV    : {CSV_PATH}")
    lines.append(f"")
    lines.append(f"  Copied        : {len(copied)}")
    lines.append(f"  Rescued FAILs : {len(rescued)}  {rescued}")
    lines.append(f"  Skipped       : {len(skipped)}")
    lines.append(f"  File not found: {len(not_found)}")
    lines.append("=" * 68)

    if not_found:
        lines.append("\n── FILES NOT FOUND (path in CSV does not exist on disk) ──")
        for formula, path in not_found:
            lines.append(f"  {formula:<50s}  {path}")

    if skipped:
        lines.append("\n── SKIPPED ──")
        for formula, status, reason in skipped:
            lines.append(f"  [{status}]  {formula:<50s}  {reason}")

    lines.append("\n── COPIED ──")
    for formula, fname in copied:
        lines.append(f"  {formula:<50s}  →  {fname}")

    report = "\n".join(lines)
    print(report)

    with open(LOG_PATH, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")

    print(f"\nLog written → {LOG_PATH}")


if __name__ == "__main__":
    main()

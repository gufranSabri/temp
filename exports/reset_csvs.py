#!/usr/bin/env python3
"""
Resets this folder back to its pristine, pre-judging state:
  1. Removes the `llm_hallucination_label` column from every CSV here
     (if a CSV doesn't have it, it's left untouched).
  2. Deletes the exports/subsets/ folder (the per-file subset review copies).
  3. Deletes hallucination_rate_report.txt, if present.

Usage:
    python3 reset_csvs.py
"""

import csv
import glob
import os
import shutil

EXPORTS_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_COLUMN = "llm_hallucination_label"
SUBSETS_DIR = os.path.join(EXPORTS_DIR, "subsets")
REPORT_PATH = os.path.join(EXPORTS_DIR, "hallucination_rate_report.txt")


def strip_column(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if NEW_COLUMN not in fieldnames:
            return False
        rows = list(reader)

    fieldnames = [c for c in fieldnames if c != NEW_COLUMN]
    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row.pop(NEW_COLUMN, None)
            writer.writerow(row)
    os.replace(tmp_path, csv_path)
    return True


def main():
    csv_paths = sorted(glob.glob(os.path.join(EXPORTS_DIR, "*.csv")))

    changed = 0
    for path in csv_paths:
        if strip_column(path):
            print(f"  stripped '{NEW_COLUMN}' from {os.path.basename(path)}")
            changed += 1
    print(f"CSVs updated: {changed}/{len(csv_paths)}")

    if os.path.isdir(SUBSETS_DIR):
        shutil.rmtree(SUBSETS_DIR)
        print(f"Removed {SUBSETS_DIR}")

    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)
        print(f"Removed {REPORT_PATH}")

    print("Reset complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
For each CSV in this folder:
  1. Sample SUBSET_PERCENTAGE of the rows (seeded, reproducible), stratified
     by the existing `hallucination_label` column so the subset's 0/1 split
     mirrors the full dataset's hallucination rate (e.g. a dataset that's
     70% hallucinated yields a subset that's ~70% hallucinated too).
  2. For each sampled row, ask an LLM (via Novita AI's OpenAI-compatible
     chat completions API) to compare `generated_response` against the
     `gold` answers and output a single character: 1 = hallucination,
     0 = not a hallucination.
  3. Write these judgments into a new `large_llm` column
     (blank for rows that were not sampled).
  4. Compute, per CSV:
       - hallucination rate over the FULL dataset using the existing
         `hallucination_label` column
       - hallucination rate over the SUBSET using the existing
         `hallucination_label` column
       - hallucination rate over the SUBSET using the new LLM judgments
     and write a summary report to a .txt file.

This is the cloud-LLM counterpart to hallucination_llm_judge.py (which
judges via a local/small LLM through LM Studio and writes `small_llm`).
Both scripts can be run against the same CSVs - each only ever touches
its own column, so the two judgments live side by side.

Usage:
    python3 hallucination_llm_judge_cloud.py
"""

import ast
import csv
import glob
import os
import random
import subprocess
import sys
import threading
import time

from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUBSET_PERCENTAGE = 0.1          # fraction of each dataset to sample (0.0-1.0)
RANDOM_SEED = 42                  # fixed seed -> reproducible subset across runs

# Novita AI's OpenAI-compatible chat completions API, via the official
# openai SDK (pip install openai) pointed at Novita's base_url.
NOVITA_BASE_URL = "https://api.novita.ai/openai"
MODEL_NAME = "deepseek/deepseek-v4-pro"
NOVITA_API_KEY = "sk_A2vhbspxw6zvw4jzIaI2Dx4xJyBzfY1R8Uo57A1Dos0"

_client = OpenAI(api_key=NOVITA_API_KEY, base_url=NOVITA_BASE_URL)

NEW_COLUMN = "large_llm"

EXPORTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(EXPORTS_DIR, "hallucination_rate_report_cloud.txt")
REPO_DIR = os.path.dirname(EXPORTS_DIR)

# Every sampled subset row is also written here, one CSV per source file,
# containing only the sampled rows plus the LLM's judgment column - so the
# subset (and any disagreement with the existing labels) can be reviewed
# directly instead of having to diff it out of the full CSV.
SUBSETS_DIR = os.path.join(EXPORTS_DIR, "subsets")

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
# Cloud API calls have more latency/queueing variance than a local model, so
# this is kept generous enough to absorb normal slowdowns without letting one
# stalled request stall the whole run for minutes; a hard per-attempt
# wall-clock cap is enforced in query_llm.
REQUEST_TIMEOUT = 30

# Save progress back to the CSV after this many newly-judged rows, so an
# interrupted run (Ctrl-C, crash, network hiccup) loses at most this many
# judgments. Re-running the script afterwards resumes: any row that already
# has a value in NEW_COLUMN is left as-is and not re-queried.
CHECKPOINT_EVERY = 20

SYSTEM_PROMPT = (
    "You are a strict hallucination judge. You will be given a model's "
    "generated response and a list of gold (reference) answers. Decide "
    "whether the generated response hallucinates, i.e. it does NOT match "
    "/ is NOT supported by / contradicts or omits the information in the "
    "gold answers. If the generated response agrees with (or is a "
    "reasonable paraphrase / subset / superset that is consistent with) "
    "any one of the gold answers, it is NOT a hallucination.\n\n"
    "Respond with exactly one character and nothing else:\n"
    "1 = hallucination (response does not match the gold answers)\n"
    "0 = not a hallucination (response matches/is supported by a gold answer)\n"
    "Do not output any explanation, punctuation, or whitespace other than "
    "the single digit."
)


def stratified_subset_indices(rows):
    """Pick SUBSET_PERCENTAGE of `rows`, stratified by the existing
    `hallucination_label` column so the subset's 0/1 proportions match the
    full dataset's (e.g. a dataset that's 70% hallucinated yields a subset
    that's ~70% hallucinated too, not a plain uniform sample)."""
    n_total = len(rows)
    n_subset = round(n_total * SUBSET_PERCENTAGE)

    hallucinated = [i for i, row in enumerate(rows) if row.get("hallucination_label", "").strip() == "1"]
    not_hallucinated = [i for i, row in enumerate(rows) if row.get("hallucination_label", "").strip() == "0"]
    other = [i for i in range(n_total) if i not in set(hallucinated) and i not in set(not_hallucinated)]

    rng = random.Random(RANDOM_SEED)

    if not hallucinated and not not_hallucinated:
        # No usable existing labels at all - fall back to plain uniform
        # sampling over every row.
        return set(rng.sample(range(n_total), n_subset)) if n_subset > 0 else set()

    existing_rate = len(hallucinated) / (len(hallucinated) + len(not_hallucinated))
    n_from_hallucinated = round(n_subset * existing_rate)
    n_from_not_hallucinated = n_subset - n_from_hallucinated

    n_from_hallucinated = min(n_from_hallucinated, len(hallucinated))
    n_from_not_hallucinated = min(n_from_not_hallucinated, len(not_hallucinated))

    picked = set(rng.sample(hallucinated, n_from_hallucinated))
    picked |= set(rng.sample(not_hallucinated, n_from_not_hallucinated))

    # If rounding/limited pool sizes left us short of n_subset, top up from
    # whichever pool (including rows with no usable label) still has room,
    # so the subset size still matches SUBSET_PERCENTAGE as closely as possible.
    shortfall = n_subset - len(picked)
    if shortfall > 0:
        remaining_pool = [i for i in hallucinated + not_hallucinated + other if i not in picked]
        rng.shuffle(remaining_pool)
        picked |= set(remaining_pool[:shortfall])

    return picked


def parse_gold(raw_gold):
    """gold is stored as a stringified Python list, e.g. "['white']"."""
    try:
        parsed = ast.literal_eval(raw_gold)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
        return [str(parsed)]
    except (ValueError, SyntaxError):
        return [raw_gold]


def build_user_input(generated_response, gold_answers):
    gold_block = "\n".join(f"- {g}" for g in gold_answers)
    return (
        f"Gold answers:\n{gold_block}\n\n"
        f"Generated response:\n{generated_response}\n\n"
        "Does the generated response hallucinate relative to the gold "
        "answers? Answer with 1 or 0 only."
    )


def _do_request(messages, result_box):
    """Runs in a worker thread so the caller can enforce a hard wall-clock
    deadline: the SDK's own `timeout=` only bounds individual socket ops, and
    in practice a request can stall past it without ever raising, leaving
    the whole script hung."""
    try:
        result_box["response"] = _client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=100,
            extra_body={"enable_thinking": False},
        )
    except Exception as e:  # noqa: BLE001 - surfaced to the caller below
        result_box["error"] = e


def query_llm(generated_response, gold_answers):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_input(generated_response, gold_answers)},
    ]

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        result_box = {}
        worker = threading.Thread(target=_do_request, args=(messages, result_box), daemon=True)
        worker.start()
        worker.join(timeout=REQUEST_TIMEOUT)

        if worker.is_alive():
            # Hard-hung request: the SDK didn't raise on its own timeout.
            # We can't kill the thread, but we stop waiting on it (it's a
            # daemon thread, so it won't block process exit) and move on.
            last_err = f"request hard-hung past {REQUEST_TIMEOUT}s"
        elif "error" in result_box:
            last_err = str(result_box["error"])
        else:
            try:
                response = result_box["response"]
                content = (response.choices[0].message.content or "").strip()
                label = extract_binary_label(content)
                if label is not None:
                    return label
                last_err = f"unparseable response: content={content!r}"
            except (KeyError, IndexError, AttributeError) as e:
                last_err = str(e)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    print(f"    WARNING: giving up after {MAX_RETRIES} attempts ({last_err}); leaving blank", file=sys.stderr)
    return None


def extract_binary_label(text, from_end=False):
    """Pull a 0 or 1 out of the model's reply, tolerating stray whitespace/punctuation."""
    chars = reversed(text) if from_end else text
    for ch in chars:
        if ch == "1":
            return "1"
        if ch == "0":
            return "0"
    return None


def save_csv(csv_path, fieldnames, rows):
    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, csv_path)  # atomic on POSIX; avoids a half-written CSV


def save_subset_csv(fname, fieldnames, rows, subset_indices):
    """Write just the sampled subset rows (with whatever llm judgments exist
    so far) to exports/subsets/<fname>, for easy manual review."""
    os.makedirs(SUBSETS_DIR, exist_ok=True)
    subset_path = os.path.join(SUBSETS_DIR, fname)
    subset_rows = [rows[i] for i in sorted(subset_indices)]
    save_csv(subset_path, fieldnames, subset_rows)


def process_csv(csv_path, overall_bar=None):
    fname = os.path.basename(csv_path)
    print(f"\n=== {fname} ===")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    n_total = len(rows)
    subset_indices = stratified_subset_indices(rows)
    n_subset = len(subset_indices)

    had_column_already = NEW_COLUMN in fieldnames
    if not had_column_already:
        fieldnames = fieldnames + [NEW_COLUMN]

    # Resume support: a row already carrying a valid 0/1 in NEW_COLUMN from a
    # prior (possibly interrupted) run is treated as already judged.
    already_done = 0
    if had_column_already:
        for i, row in enumerate(rows):
            if i in subset_indices and row.get(NEW_COLUMN, "").strip() in ("0", "1"):
                already_done += 1

    print(f"  total rows: {n_total} | subset size ({SUBSET_PERCENTAGE:.0%}): {n_subset}"
          + (f" | resuming, {already_done} already judged" if already_done else ""))

    # Write the subset-only review copy right away (even before any judging
    # happens this run), so it's there to look at from the start.
    for i in subset_indices:
        rows[i].setdefault(NEW_COLUMN, "")
    save_subset_csv(fname, fieldnames, rows, subset_indices)

    newly_judged_since_checkpoint = 0

    with tqdm(total=n_subset, initial=already_done, desc=fname, unit="row") as bar:
        for i, row in enumerate(rows):
            row.setdefault(NEW_COLUMN, "")
            if i not in subset_indices:
                continue
            if row[NEW_COLUMN].strip() in ("0", "1"):
                continue  # already judged in a previous run

            gold_answers = parse_gold(row.get("gold", ""))
            generated_response = row.get("generated_response", "")

            label = query_llm(generated_response, gold_answers)
            row[NEW_COLUMN] = label if label is not None else ""

            newly_judged_since_checkpoint += 1
            bar.update(1)
            if overall_bar is not None:
                overall_bar.update(1)

            if newly_judged_since_checkpoint >= CHECKPOINT_EVERY:
                save_csv(csv_path, fieldnames, rows)
                save_subset_csv(fname, fieldnames, rows, subset_indices)
                newly_judged_since_checkpoint = 0

    save_csv(csv_path, fieldnames, rows)
    save_subset_csv(fname, fieldnames, rows, subset_indices)

    # ---- compute rates ----
    def rate(values):
        vals = [v for v in values if v in ("0", "1")]
        if not vals:
            return None
        return sum(int(v) for v in vals) / len(vals)

    full_rate = rate(row["hallucination_label"] for row in rows)
    subset_existing_rate = rate(
        row["hallucination_label"] for i, row in enumerate(rows) if i in subset_indices
    )
    subset_llm_rate = rate(
        row[NEW_COLUMN] for i, row in enumerate(rows) if i in subset_indices
    )

    return {
        "file": fname,
        "n_total": n_total,
        "n_subset": n_subset,
        "full_dataset_rate_existing_labels": full_rate,
        "subset_rate_existing_labels": subset_existing_rate,
        "subset_rate_llm_labels": subset_llm_rate,
    }


def fmt_rate(r):
    return f"{r:.4f} ({r * 100:.2f}%)" if r is not None else "N/A"


def render_report(results):
    lines = []
    lines.append("Hallucination Rate Report")
    lines.append(f"Subset percentage: {SUBSET_PERCENTAGE:.0%}  |  Random seed: {RANDOM_SEED}")
    lines.append(f"Judge model: {MODEL_NAME} ({NOVITA_BASE_URL})")
    lines.append("=" * 70)

    for r in results:
        lines.append("")
        lines.append(r["file"])
        lines.append("-" * len(r["file"]))
        lines.append(f"  Total rows                              : {r['n_total']}")
        lines.append(f"  Subset size                              : {r['n_subset']}")
        lines.append(f"  Full-dataset hallucination rate (existing): {fmt_rate(r['full_dataset_rate_existing_labels'])}")
        lines.append(f"  Subset hallucination rate (existing)      : {fmt_rate(r['subset_rate_existing_labels'])}")
        lines.append(f"  Subset hallucination rate (LLM judged)     : {fmt_rate(r['subset_rate_llm_labels'])}")

    return "\n".join(lines) + "\n"


def count_remaining(csv_path):
    """Rows left to judge in this file: subset size minus already-judged rows."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    subset_indices = stratified_subset_indices(rows)
    n_subset = len(subset_indices)

    already_done = 0
    if NEW_COLUMN in fieldnames:
        for i, row in enumerate(rows):
            if i in subset_indices and row.get(NEW_COLUMN, "").strip() in ("0", "1"):
                already_done += 1

    return n_subset, already_done


def push_to_github():
    """Commit and push the whole repo (git@github.com:gufranSabri/temp.git)
    so the run's outputs are available remotely once this script finishes."""
    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=REPO_DIR, capture_output=True, text=True
        )

    run("add", "-A")
    diff = run("diff", "--cached", "--quiet")
    if diff.returncode == 0:
        print("Nothing new to push.")
        return

    commit = run(
        "commit", "-m",
        "Add hallucination judge run outputs\n\n"
        "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>",
    )
    if commit.returncode != 0:
        print(f"WARNING: git commit failed: {commit.stderr}", file=sys.stderr)
        return

    push = run("push", "-u", "origin", "main")
    if push.returncode != 0:
        print(f"WARNING: git push failed: {push.stderr}", file=sys.stderr)
        return

    print("Pushed run outputs to origin/main.")


def main():
    csv_paths = sorted(glob.glob(os.path.join(EXPORTS_DIR, "*.csv")))
    if not csv_paths:
        print("No CSV files found in exports folder.")
        return

    # Pre-scan every file so the overall bar's total/ETA covers the whole
    # run from the start (and correctly discounts rows already judged from
    # a prior interrupted run).
    per_file_subset = {}
    total_subset = 0
    total_already_done = 0
    for path in csv_paths:
        n_subset, already_done = count_remaining(path)
        per_file_subset[path] = n_subset
        total_subset += n_subset
        total_already_done += already_done

    print(f"Overall: {total_subset} rows to judge across {len(csv_paths)} files "
          f"({total_already_done} already done from a previous run)" if total_already_done
          else f"Overall: {total_subset} rows to judge across {len(csv_paths)} files")

    # Write the report after each file so an interrupted run (this can take
    # hours across all CSVs) still leaves a report covering whatever
    # finished so far, not just a report from a fully completed run.
    results = []
    with tqdm(total=total_subset, initial=total_already_done, desc="OVERALL", unit="row") as overall_bar:
        for path in csv_paths:
            results.append(process_csv(path, overall_bar=overall_bar))
            report_text = render_report(results)
            with open(REPORT_PATH, "w", encoding="utf-8") as f:
                f.write(report_text)

    print("\n" + report_text)
    print(f"Report written to: {REPORT_PATH}")

    push_to_github()


if __name__ == "__main__":
    main()

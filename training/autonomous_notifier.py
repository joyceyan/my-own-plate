"""
Companion notifier for the autonomous experiment loop.

Polls `autonomous_status.json` and the experiment logs, detects state changes,
writes a human-readable summary to `autonomous_summary.md`, and sends macOS
desktop notifications via `osascript` when available.

Intended to run in the background next to `autonomous_loop.py`.

Usage:
    cd /Users/jyan/src/my-own-plate
    .venv/bin/python training/autonomous_notifier.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATUS_FILE = ROOT / "training" / "autonomous_status.json"
SUMMARY_FILE = ROOT / "training" / "autonomous_summary.md"
LOGS_DIR = ROOT / "training" / "logs"
RESULTS_TSV = ROOT / "training" / "results.tsv"

POLL_INTERVAL = 60  # seconds


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"[{now_utc()}] ERROR reading {path}: {e}", file=sys.stderr)
    return None


def read_tail(path: Path, n: int = 20):
    if not path.exists():
        return ""
    try:
        with open(path) as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception:
        return ""


def notify(title: str, message: str):
    """Send a macOS desktop notification if osascript is available."""
    if sys.platform != "darwin":
        return
    try:
        # AppleScript only requires " to be escaped as \". Keep the message short
        # and ASCII to avoid emoji/title rendering issues on some macOS versions.
        clean = message.replace('\\', '\\\\').replace('"', '\\"').replace("\n", " ")
        clean = clean.encode("ascii", "ignore").decode("ascii")
        subprocess.run(
            ["osascript", "-e", f'display notification "{clean}" with title "{title}"'],
            check=True,
            capture_output=True,
            timeout=10,
        )
    except Exception as e:
        print(f"[{now_utc()}] Notification failed: {e}", file=sys.stderr)


def write_summary(text: str):
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_FILE, "a") as f:
        f.write(f"\n---\n{now_utc()}\n{text}\n")


def read_last_tsv_line():
    if not RESULTS_TSV.exists():
        return None
    with open(RESULTS_TSV) as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) <= 1:
        return None
    return lines[-1]


def parse_tsv_line(line: str):
    cols = line.split("\t")
    if len(cols) < 9:
        return None
    return {
        "experiment": cols[0],
        "calories": cols[1],
        "protein": cols[2],
        "fat": cols[3],
        "carbs": cols[4],
        "avg": cols[5],
        "parse_failures": cols[6],
        "status": cols[7],
        "description": cols[8],
    }


def estimate_train_duration_steps(exp_id: int):
    """Estimate remaining training time from the current training log."""
    log = LOGS_DIR / f"exp{exp_id}_train.log"
    if not log.exists():
        return None
    try:
        with open(log) as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "it/s" in line or "s/it" in line:
                # Look for the bracketed timing block, e.g.
                #   0%|          | 30/33504 [00:36<10:51:04,  1.17s/it]
                if "[" in line and "<" in line:
                    bracket = line.split("[")[-1].split("]")[0]
                    if "<" in bracket:
                        remaining = bracket.split("<")[1]
                        # remaining is like "10:51:04,  1.17s/it"
                        remaining = remaining.split(",")[0].strip()
                        return remaining
        return None
    except Exception:
        return None


def format_experiment_start(exp_id: int, description: str, params: dict):
    eta = estimate_train_duration_steps(exp_id)
    eta_str = f"(~{eta} remaining)" if eta else "(ETA ~10-12 hours)"
    params_str = json.dumps(params)
    return (
        f"STARTED Exp {exp_id}: {description}\n"
        f"Params: {params_str}\n"
        f"Training ETA: {eta_str}"
    )


def format_training_complete(exp_id: int):
    return (
        f"TRAINING COMPLETE Exp {exp_id}.\n"
        f"Starting evaluation now (ETA ~15-20 minutes)."
    )


def format_evaluation_started(exp_id: int):
    return (
        f"EVAL STARTED Exp {exp_id}.\n"
        f"ETA ~15-20 minutes for 349 validation samples."
    )


def format_experiment_completed(exp_id: int):
    line = read_last_tsv_line()
    if not line:
        return f"FINISHED Exp {exp_id} (result line not yet in results.tsv)."
    result = parse_tsv_line(line)
    if not result or result["experiment"] != str(exp_id):
        return f"FINISHED Exp {exp_id} (awaiting results.tsv update)."
    status_text = "KEPT" if result["status"] == "kept" else "REVERTED"
    return (
        f"{status_text} Exp {exp_id}: {result['description']}\n"
        f"Results: cal {result['calories']}% / pro {result['protein']}% / "
        f"fat {result['fat']}% / carb {result['carbs']}% = avg {result['avg']}%\n"
        f"Parse failures: {result['parse_failures']}/349"
    )


def main():
    print(f"[{now_utc()}] Starting autonomous notifier")
    write_summary("Notifier started. Watching `autonomous_status.json`.")
    notify("My Own Plate — Autonomous Loop", "Notifier started. Watching experiments.")

    last_state = None
    last_completed_id = None

    while True:
        status = read_json(STATUS_FILE)
        if status is None:
            time.sleep(POLL_INTERVAL)
            continue

        current = status.get("current")
        history = status.get("history", [])
        completed_id = history[-1]["id"] if history else None

        # Detect new experiment start
        if current is not None:
            current_id = current.get("id")
            phase = current.get("phase")
            state_key = (current_id, phase)

            if last_state != state_key:
                if phase == "training" or phase == "config_committed":
                    summary = format_experiment_start(
                        current_id,
                        current.get("description", ""),
                        current.get("params", {}),
                    )
                    write_summary(summary)
                    notify("Exp Started", summary)

                elif phase == "evaluating":
                    summary = format_evaluation_started(current_id)
                    write_summary(summary)
                    notify("Eval Started", summary)

                last_state = state_key

        # Detect experiment completion (current becomes null and history grew)
        if current is None and completed_id != last_completed_id:
            summary = format_experiment_completed(completed_id)
            write_summary(summary)
            notify("Exp Completed", summary)
            last_completed_id = completed_id
            last_state = None

        # Detect training-to-evaluation transition while current stays the same
        if current is not None and last_state is not None:
            last_id, last_phase = last_state
            current_id = current.get("id")
            current_phase = current.get("phase")
            if last_id == current_id and last_phase == "training" and current_phase == "evaluating":
                summary = format_training_complete(current_id)
                write_summary(summary)
                notify("Training Complete", summary)
                last_state = (current_id, current_phase)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

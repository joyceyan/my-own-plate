"""
Autonomous experiment loop for the HF Nutrition5k fine-tuning pipeline.

Reads an experiment queue from `experiment_queue.json`, runs each experiment
(train + evaluate on val), compares results to the best-so-far, records
outcomes in `results.tsv` and `notes.md`, and continues to the next experiment.

Intended to run for long periods unattended. Progress is written to
`autonomous_status.json` and per-experiment log files in `logs/`.

Usage:
    cd /Users/jyan/src/my-own-plate && .venv/bin/python training/autonomous_loop.py
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = ROOT / "training"
STATUS_FILE = TRAINING_DIR / "autonomous_status.json"
QUEUE_FILE = TRAINING_DIR / "experiment_queue.json"
RESULTS_TSV = TRAINING_DIR / "results.tsv"
NOTES_MD = TRAINING_DIR / "notes.md"
OUTPUT_DIR = TRAINING_DIR / "output"
LOGS_DIR = TRAINING_DIR / "logs"

PYTHON = ROOT / ".venv" / "bin" / "python"
TRAIN_SCRIPT = TRAINING_DIR / "train.py"
EVAL_SCRIPT = TRAINING_DIR / "evaluate.py"

DEFAULT_TRAIN_ARGS = {
    "model": "training/cache/Qwen3-VL-2B-Instruct",
    "train_data": "~/src/my-own-plate/data/nutrition5k_hf_chat",
    "output_dir": "~/src/my-own-plate/training/output",
}

NUTRIENTS = ["calories", "protein", "fat", "carbs"]


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_status():
    if STATUS_FILE.exists():
        return load_json(STATUS_FILE)
    return {"current": None, "best": None, "history": []}


def save_status(status):
    save_json(STATUS_FILE, status)


def load_queue():
    if QUEUE_FILE.exists():
        return load_json(QUEUE_FILE)
    return {"experiments": []}


def save_queue(queue):
    save_json(QUEUE_FILE, queue)


def read_eval_summary(mode: str = "val"):
    summary_file = TRAINING_DIR / "eval_results_hf" / f"eval_summary_{mode}.json"
    if summary_file.exists():
        return load_json(summary_file)
    return None


def is_process_running(pid: int):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def run_in_background(cmd: list[str], log_file: Path) -> int:
    """Run `cmd` with stdout/stderr redirected to `log_file`; return PID."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
            env=env,
        )
    return proc.pid


def wait_for_process(pid: int, log_file: Path, timeout_seconds: float = None) -> bool:
    """Poll until process exits. Returns True if exited, False on timeout."""
    start = time.time()
    while is_process_running(pid):
        if timeout_seconds is not None and time.time() - start > timeout_seconds:
            return False
        time.sleep(30)
    return True


def git_commit(message: str, paths: list[Path]):
    """Stage the given paths and commit with `message`."""
    subprocess.run(["git", "add"] + [str(p) for p in paths], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)


def git_reset_hard_head_1():
    """Revert the most recent commit and reset working tree."""
    subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=ROOT, check=True)


def git_amend(message: str, paths: list[Path]):
    """Stage paths and amend the current commit."""
    subprocess.run(["git", "add"] + [str(p) for p in paths], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "--amend", "-m", message], cwd=ROOT, check=True)


def update_results_tsv(exp_id: int, metrics: dict, parse_failures: int, status: str, description: str):
    rows = []
    if RESULTS_TSV.exists():
        with open(RESULTS_TSV) as f:
            rows = [line.rstrip("\n") for line in f]
    else:
        rows = [
            "experiment\tcal_mae_pct\tprotein_mae_pct\tfat_mae_pct\tcarbs_mae_pct\tavg_mae_pct\tparse_failures\tstatus\tdescription"
        ]

    avg = sum(metrics[n]["mae_pct"] for n in NUTRIENTS) / len(NUTRIENTS)
    new_row = (
        f"{exp_id}\t"
        f"{metrics['calories']['mae_pct']}\t"
        f"{metrics['protein']['mae_pct']}\t"
        f"{metrics['fat']['mae_pct']}\t"
        f"{metrics['carbs']['mae_pct']}\t"
        f"{avg:.1f}\t"
        f"{parse_failures}\t"
        f"{status}\t"
        f"{description}"
    )
    rows.append(new_row)
    with open(RESULTS_TSV, "w") as f:
        f.write("\n".join(rows) + "\n")


def append_notes_md(exp_id: int, description: str, result_metrics: dict, parse_failures: int,
                    status: str, params: dict, comparison: dict = None):
    avg = sum(result_metrics[n]["mae_pct"] for n in NUTRIENTS) / len(NUTRIENTS)
    lines = ["", f"### Exp {exp_id}: {description} — {status.upper()}", ""]
    lines.append(f"**Params**: {json.dumps(params)}")
    lines.append("")
    result_str = "/".join(f"{result_metrics[n]['mae_pct']:.1f}" for n in NUTRIENTS)
    lines.append(f"**Result**: {result_str} = **{avg:.1f}% avg**, {parse_failures} parse failures.")
    lines.append("")
    if comparison:
        lines.append("**Comparison vs best-so-far**:")
        for n in NUTRIENTS:
            prev = comparison.get(n)
            curr = result_metrics[n]["mae_pct"]
            delta = curr - prev if prev is not None else 0.0
            sign = "+" if delta > 0 else ""
            lines.append(f"- {n}: {prev:.1f}% → {curr:.1f}% ({sign}{delta:.1f}pp)")
        prev_avg = sum(comparison.values()) / len(comparison) if comparison else avg
        avg_delta = avg - prev_avg
        sign = "+" if avg_delta > 0 else ""
        lines.append(f"- avg: {prev_avg:.1f}% → {avg:.1f}% ({sign}{avg_delta:.1f}pp)")
        lines.append("")
    lines.append("**Insight**: TBD — update manually if needed.")
    lines.append("")

    with open(NOTES_MD, "a") as f:
        f.write("\n".join(lines) + "\n")


def build_train_args(exp: dict) -> list[str]:
    args = [str(PYTHON), str(TRAIN_SCRIPT)]
    params = {**DEFAULT_TRAIN_ARGS, **exp.get("params", {})}
    for key, value in params.items():
        arg_key = key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(f"--{arg_key}")
            else:
                args.append(f"--no-{arg_key}")
        elif isinstance(value, list):
            args.append(f"--{arg_key}")
            args.extend([str(v) for v in value])
        else:
            args.extend([f"--{arg_key}", str(value)])
    return args


def run_training(exp: dict, log_file: Path) -> int:
    cmd = build_train_args(exp)
    print(f"[{now_utc()}] Starting training for exp {exp['id']}: {' '.join(cmd)}")
    pid = run_in_background(cmd, log_file)
    print(f"[{now_utc()}] Training PID: {pid}")
    return pid


def run_evaluation(exp: dict, log_file: Path) -> int:
    cmd = [str(PYTHON), str(EVAL_SCRIPT), "--mode", "val"]
    params = exp.get("params", {})

    # Pass through params that affect evaluation
    if "image_size" in params:
        cmd.extend(["--image-size", str(params["image_size"])])
    if "lora_rank_vision" in params:
        cmd.extend(["--lora-rank-vision", str(params["lora_rank_vision"])])
    if "lora_alpha_vision" in params:
        cmd.extend(["--lora-alpha-vision", str(params["lora_alpha_vision"])])
    if "lora_rank_llm" in params:
        cmd.extend(["--lora-rank-projector", str(params["lora_rank_llm"])])
    if "lora_alpha_llm" in params:
        cmd.extend(["--lora-alpha-projector", str(params["lora_alpha_llm"])])

    print(f"[{now_utc()}] Starting evaluation for exp {exp['id']}: {' '.join(cmd)}")
    pid = run_in_background(cmd, log_file)
    print(f"[{now_utc()}] Evaluation PID: {pid}")
    return pid


def is_kept(metrics: dict, best: dict | None, parse_failures: int, baseline_parse_failures: int = 1):
    """
    Success criteria from program.md:
    1. Average MAE% decreases vs best-so-far.
    2. No individual nutrient increases by >5pp vs best-so-far per-nutrient.
    3. Parse failures don't increase significantly (>2x baseline rate).
    """
    if best is None:
        return True

    curr_avg = sum(metrics[n]["mae_pct"] for n in NUTRIENTS) / len(NUTRIENTS)
    best_avg = sum(best[n]["mae_pct"] for n in NUTRIENTS) / len(NUTRIENTS)
    if curr_avg >= best_avg:
        return False

    for n in NUTRIENTS:
        if metrics[n]["mae_pct"] > best[n]["mae_pct"] + 5.0:
            return False

    baseline_rate = baseline_parse_failures / 349.0
    current_rate = parse_failures / 349.0
    if current_rate > 2 * baseline_rate:
        return False

    return True


def find_exp_by_id(queue: dict, exp_id: int):
    for exp in queue.get("experiments", []):
        if exp.get("id") == exp_id:
            return exp
    return None


def find_next_pending(queue: dict):
    for exp in queue.get("experiments", []):
        if exp.get("status") not in ("kept", "reverted"):
            return exp
    return None


def ensure_training_completed(exp: dict, status: dict, train_log: Path) -> int:
    """Resume or start training. Returns the PID of the completed training run."""
    current = status.get("current", {})
    train_pid = current.get("train_pid")

    if train_pid and is_process_running(train_pid):
        print(f"[{now_utc()}] Resuming training for exp {exp['id']} (PID {train_pid})")
        if not wait_for_process(train_pid, train_log):
            raise RuntimeError(f"Training for exp {exp['id']} timed out")
    else:
        # Start fresh training
        status["current"]["phase"] = "training"
        save_status(status)
        train_pid = run_training(exp, train_log)
        status["current"]["train_pid"] = train_pid
        save_status(status)
        if not wait_for_process(train_pid, train_log):
            raise RuntimeError(f"Training for exp {exp['id']} timed out")

    train_output = train_log.read_text() if train_log.exists() else ""
    if "Training complete" not in train_output:
        raise RuntimeError(f"Training for exp {exp['id']} did not complete successfully. See {train_log}")
    return train_pid


def ensure_evaluation_completed(exp: dict, status: dict, eval_log: Path) -> int:
    """Resume or start evaluation. Returns the PID of the completed evaluation run."""
    current = status.get("current", {})
    eval_pid = current.get("eval_pid")

    if eval_pid and is_process_running(eval_pid):
        print(f"[{now_utc()}] Resuming evaluation for exp {exp['id']} (PID {eval_pid})")
        if not wait_for_process(eval_pid, eval_log):
            raise RuntimeError(f"Evaluation for exp {exp['id']} timed out")
    else:
        status["current"]["phase"] = "evaluating"
        save_status(status)
        eval_pid = run_evaluation(exp, eval_log)
        status["current"]["eval_pid"] = eval_pid
        save_status(status)
        if not wait_for_process(eval_pid, eval_log):
            raise RuntimeError(f"Evaluation for exp {exp['id']} timed out")

    return eval_pid

def run_experiment(exp_id: int, status: dict):
    """Run or resume an experiment end-to-end (train + eval + record + keep/revert)."""
    queue = load_queue()
    exp = find_exp_by_id(queue, exp_id)
    if exp is None:
        raise RuntimeError(f"Experiment {exp_id} not found in queue")

    description = exp["description"]
    params = exp.get("params", {})
    train_log = LOGS_DIR / f"exp{exp_id}_train.log"
    eval_log = LOGS_DIR / f"exp{exp_id}_eval.log"

    # Detect whether we're resuming an in-progress experiment.
    current = status.get("current")
    resuming = (
        current is not None
        and current.get("id") == exp_id
        and current.get("phase") in ("config_committed", "training", "evaluating")
    )

    if resuming:
        print(f"[{now_utc()}] Resuming exp {exp_id}: {description}")
        config_message = f"exp {exp_id}: {description}"
    else:
        # Mark experiment as running in status; this also creates a diff so the
        # subsequent config commit has changes to stage.
        status["current"] = {
            "id": exp_id,
            "phase": "config_committed",
            "description": description,
            "start_time": now_utc(),
            "params": params,
        }
        save_status(status)

        # Commit experiment config before training.
        config_message = f"exp {exp_id}: {description}"
        print(f"[{now_utc()}] Committing config: {config_message}")
        git_commit(config_message, [QUEUE_FILE, STATUS_FILE])

    # Ensure training is completed (resume or start fresh).
    ensure_training_completed(exp, status, train_log)

    # Ensure evaluation is completed.
    ensure_evaluation_completed(exp, status, eval_log)

    summary = read_eval_summary("val")
    if summary is None:
        raise RuntimeError(f"Evaluation summary not found for exp {exp_id}")

    metrics = summary["metrics"]
    parse_failures = summary["parse_failures"]
    avg = sum(metrics[n]["mae_pct"] for n in NUTRIENTS) / len(NUTRIENTS)

    best = status.get("best")
    kept = is_kept(metrics, best, parse_failures)

    if best is None:
        comparison = None
    else:
        comparison = {n: best[n]["mae_pct"] for n in NUTRIENTS}

    # Record results
    status_str = "kept" if kept else "reverted"
    update_results_tsv(exp_id, metrics, parse_failures, status_str, description)
    append_notes_md(exp_id, description, metrics, parse_failures, status_str, params, comparison)

    result_message = (
        f"exp {exp_id} results: {metrics['calories']['mae_pct']:.1f}/"
        f"{metrics['protein']['mae_pct']:.1f}/"
        f"{metrics['fat']['mae_pct']:.1f}/"
        f"{metrics['carbs']['mae_pct']:.1f} = {avg:.1f}% avg ({status_str})"
    )
    print(f"[{now_utc()}] {result_message}")

    # Update queue and status
    queue = load_queue()
    for qexp in queue["experiments"]:
        if qexp["id"] == exp_id:
            qexp["status"] = status_str
            qexp["avg_mae_pct"] = round(avg, 1)
            qexp["completed_at"] = now_utc()
            break
    save_queue(queue)
    save_status(status)

    if kept:
        # Amend the config commit to include results and status files.
        git_amend(f"{config_message} — {status_str.upper()}", [RESULTS_TSV, NOTES_MD, QUEUE_FILE, STATUS_FILE])
        status["best"] = {n: {"mae_pct": metrics[n]["mae_pct"]} for n in NUTRIENTS}
        status["best_exp"] = exp_id
        save_status(status)
    else:
        # Revert the config commit and restore adapter weights.
        git_reset_hard_head_1()
        # After reset, results.tsv/notes.md are back to previous state, but OUTPUT_DIR may still
        # contain the failed adapters. The next training run will overwrite them.
        status["current"]["phase"] = "reverted"
        save_status(status)

    status["history"].append({
        "id": exp_id,
        "description": description,
        "status": status_str,
        "avg_mae_pct": round(avg, 1),
        "completed_at": now_utc(),
    })
    status["current"] = None
    save_status(status)

    return kept


def main():
    print(f"[{now_utc()}] Starting autonomous loop")
    status = load_status()

    # Resume any in-progress experiment.
    current = status.get("current")
    if current is not None and current.get("id") is not None:
        exp_id = current["id"]
        print(f"[{now_utc()}] Resuming in-progress experiment {exp_id} (phase: {current.get('phase')})")
        try:
            run_experiment(exp_id, status)
        except Exception as e:
            print(f"[{now_utc()}] ERROR resuming exp {exp_id}: {e}", file=sys.stderr)
            status["current"] = None
            save_status(status)
            raise

    while True:
        # Reload queue from disk each iteration so live edits take effect.
        queue = load_queue()
        exp = find_next_pending(queue)
        if exp is None:
            break

        print(f"[{now_utc()}] Running exp {exp['id']}: {exp['description']}")
        try:
            run_experiment(exp["id"], status)
        except Exception as e:
            print(f"[{now_utc()}] ERROR in exp {exp['id']}: {e}", file=sys.stderr)
            status["current"] = None
            save_status(status)
            raise

    print(f"[{now_utc()}] Autonomous loop complete. Queue empty.")


if __name__ == "__main__":
    main()

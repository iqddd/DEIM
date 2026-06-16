"""
Sequential hyperparameter sweep runner for DEIM fine-tuning on the 9-class subset.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/deim_dfine/deim_hgnetv2_x_train_dataset_include_1_9_tuning_768_freeze_stage1.yml"
DEFAULT_WEIGHTS = REPO_ROOT / "weights/deim_hgnetv2_x_obj365_pretrained.pth"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "deim_outputs/hparam_sweeps"


EXPERIMENTS = [
    {
        "name": "fs1_lr1e4_bblr1e6_bs16",
        "config": str(DEFAULT_CONFIG),
        "updates": {
            "train_dataloader.total_batch_size": 16,
            "val_dataloader.total_batch_size": 32,
            "optimizer.lr": 0.0001,
            "optimizer.params": [
                {"params": "^(?=.*backbone)(?!.*norm|bn).*$", "lr": 0.000001},
                {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$", "weight_decay": 0.0},
            ],
            "warmup_iter": 100,
            "print_freq": 10,
            "checkpoint_freq": 8,
        },
    },
    {
        "name": "fs1_lr7p5e5_bblr7p5e7_bs16",
        "config": str(DEFAULT_CONFIG),
        "updates": {
            "train_dataloader.total_batch_size": 16,
            "val_dataloader.total_batch_size": 32,
            "optimizer.lr": 0.000075,
            "optimizer.params": [
                {"params": "^(?=.*backbone)(?!.*norm|bn).*$", "lr": 0.00000075},
                {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$", "weight_decay": 0.0},
            ],
            "warmup_iter": 150,
            "print_freq": 10,
            "checkpoint_freq": 8,
        },
    },
    {
        "name": "fs1_lr1p25e4_bblr1p25e6_bs16",
        "config": str(DEFAULT_CONFIG),
        "updates": {
            "train_dataloader.total_batch_size": 16,
            "val_dataloader.total_batch_size": 32,
            "optimizer.lr": 0.000125,
            "optimizer.params": [
                {"params": "^(?=.*backbone)(?!.*norm|bn).*$", "lr": 0.00000125},
                {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$", "weight_decay": 0.0},
            ],
            "warmup_iter": 100,
            "print_freq": 10,
            "checkpoint_freq": 8,
        },
    },
    {
        "name": "stem_lr1e4_bblr2e6_bs16",
        "config": str(DEFAULT_CONFIG),
        "updates": {
            "train_dataloader.total_batch_size": 16,
            "val_dataloader.total_batch_size": 32,
            "HGNetv2.freeze_stem_only": True,
            "HGNetv2.freeze_at": 0,
            "optimizer.lr": 0.0001,
            "optimizer.params": [
                {"params": "^(?=.*backbone)(?!.*norm|bn).*$", "lr": 0.000002},
                {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$", "weight_decay": 0.0},
            ],
            "warmup_iter": 100,
            "print_freq": 10,
            "checkpoint_freq": 8,
        },
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--epochs-override", type=int)
    parser.add_argument("--max-experiments", type=int)
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_latest_symlink(output_root: Path, run_root: Path) -> None:
    latest = output_root / "latest"
    if latest.is_symlink() or latest.is_file():
        latest.unlink()
    elif latest.is_dir():
        shutil.rmtree(latest)
    latest.symlink_to(run_root.name)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def cli_literal(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".12f").rstrip("0").rstrip(".")
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(cli_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        parts = [f"{key}: {cli_literal(item)}" for key, item in value.items()]
        return "{" + ", ".join(parts) + "}"
    raise TypeError(f"Unsupported CLI literal type: {type(value)}")


def extract_best_metrics(output_dir: Path) -> dict:
    log_path = output_dir / "log.txt"
    if not log_path.exists():
        return {}

    best = {
        "best_ap": None,
        "best_ap50": None,
        "best_ap75": None,
        "best_epoch": None,
    }
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        metrics = row.get("test_coco_eval_bbox")
        if not isinstance(metrics, list) or not metrics:
            continue
        ap = metrics[0]
        if best["best_ap"] is None or ap > best["best_ap"]:
            best["best_ap"] = ap
            best["best_ap50"] = metrics[1] if len(metrics) > 1 else None
            best["best_ap75"] = metrics[2] if len(metrics) > 2 else None
            best["best_epoch"] = row.get("epoch")
    return best


def run_experiment(exp: dict, args, run_root: Path) -> dict:
    exp_root = run_root / exp["name"]
    output_dir = exp_root / "output"
    summary_dir = output_dir / "summary"
    log_file = exp_root / "train.log"
    exp_root.mkdir(parents=True, exist_ok=True)

    updates = dict(exp["updates"])
    updates["output_dir"] = str(output_dir)
    updates["summary_dir"] = str(summary_dir)
    if args.epochs_override is not None:
        updates["epoches"] = args.epochs_override

    cmd = [
        sys.executable,
        "train.py",
        "-c",
        exp["config"],
        "-t",
        str(Path(args.weights).resolve()),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if args.use_amp:
        cmd.append("--use-amp")
    if updates:
        cmd.extend(["-u"] + [f"{key}={cli_literal(value)}" for key, value in updates.items()])

    header = {
        "name": exp["name"],
        "config": exp["config"],
        "weights": str(Path(args.weights).resolve()),
        "updates": updates,
        "command": cmd,
        "started_at": utc_now(),
    }
    write_json(exp_root / "run_meta.json", header)

    started = time.time()
    with log_file.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=True) + "\n")
        fh.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        assert process.stdout is not None
        for line in process.stdout:
            tagged = f"[{exp['name']}] {line}"
            sys.stdout.write(tagged)
            sys.stdout.flush()
            fh.write(line)
        return_code = process.wait()

    finished = time.time()
    result = {
        "name": exp["name"],
        "config": exp["config"],
        "updates": updates,
        "return_code": return_code,
        "started_at": header["started_at"],
        "finished_at": utc_now(),
        "duration_sec": round(finished - started, 2),
        "output_dir": str(output_dir),
        "log_file": str(log_file),
        "best_stg1_exists": (output_dir / "best_stg1.pth").exists(),
        "best_stg2_exists": (output_dir / "best_stg2.pth").exists(),
        "last_exists": (output_dir / "last.pth").exists(),
    }
    result.update(extract_best_metrics(output_dir))
    write_json(exp_root / "result.json", result)
    return result


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_root = output_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    ensure_latest_symlink(output_root, run_root)

    state = {
        "started_at": utc_now(),
        "repo_root": str(REPO_ROOT),
        "weights": str(Path(args.weights).resolve()),
        "experiments": [],
        "status": "running",
    }
    write_json(run_root / "summary.json", state)

    experiments = EXPERIMENTS[: args.max_experiments] if args.max_experiments else EXPERIMENTS

    for exp in experiments:
        state["current_experiment"] = exp["name"]
        write_json(run_root / "summary.json", state)
        result = run_experiment(exp, args, run_root)
        state["experiments"].append(result)
        if result["return_code"] != 0:
            state["status"] = "failed"
            state["failed_experiment"] = exp["name"]
            write_json(run_root / "summary.json", state)
            if args.stop_on_failure:
                break
        else:
            state["status"] = "running"
            write_json(run_root / "summary.json", state)

    if state.get("status") != "failed":
        state["status"] = "completed"
    state["finished_at"] = utc_now()
    state.pop("current_experiment", None)
    write_json(run_root / "summary.json", state)


if __name__ == "__main__":
    main()

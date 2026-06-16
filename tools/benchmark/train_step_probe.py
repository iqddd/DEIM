"""
One-step training probe for estimating the maximum batch size on the current GPU.
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str)
    parser.add_argument("-t", "--tuning", required=True, type=str)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--max-scale", default=None, type=int)
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--steps", default=1, type=int)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=args.seed)

    cfg = YAMLConfig(
        args.config,
        tuning=args.tuning,
        use_amp=args.use_amp,
        resume=None,
    )

    if args.tuning and "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    train_cfg = cfg.yaml_cfg["train_dataloader"]
    train_cfg.pop("total_batch_size", None)
    train_cfg["batch_size"] = args.batch_size
    train_cfg["num_workers"] = args.num_workers

    val_cfg = cfg.yaml_cfg.get("val_dataloader", None)
    if val_cfg is not None:
        val_cfg["num_workers"] = 0

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.train()

    if args.max_scale is not None and solver.train_dataloader.collate_fn.scales is not None:
        solver.train_dataloader.collate_fn.scales = [args.max_scale]

    solver.train_dataloader.set_epoch(args.epoch)
    if dist_utils.is_dist_available_and_initialized():
        solver.train_dataloader.sampler.set_epoch(args.epoch)

    device = solver.device
    scaler = solver.scaler
    model = solver.model
    criterion = solver.criterion
    optimizer = solver.optimizer
    ema = solver.ema

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    data_iter = iter(solver.train_dataloader)
    loss = None

    for step in range(args.steps):
        try:
            samples, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(solver.train_dataloader)
            samples, targets = next(data_iter)

        samples = samples.to(device, non_blocking=True)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        metas = dict(epoch=args.epoch, step=step, global_step=step, epoch_step=len(solver.train_dataloader))

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=str(device), cache_enabled=True):
            outputs = model(samples, targets=targets)

        with torch.autocast(device_type=str(device), enabled=False):
            loss_dict = criterion(outputs, targets, **metas)

        loss = sum(loss_dict.values())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if ema is not None:
            ema.update(model)

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    print(
        f"SUCCESS batch_size={args.batch_size} "
        f"scale={args.max_scale} "
        f"steps={args.steps} "
        f"loss={loss.item():.6f} "
        f"peak_allocated_gb={peak_allocated:.2f} "
        f"peak_reserved_gb={peak_reserved:.2f} "
        f"step_sec={elapsed:.2f}"
    )

    dist_utils.cleanup()


if __name__ == "__main__":
    main()

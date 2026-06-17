#!/usr/bin/env python3
"""
Standalone inference/export script for the current 9-class DEIM/DFINE HGNetv2-X setup.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def get_font_paths() -> List[str]:
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]

    windir = os.environ.get("WINDIR")
    if windir:
        win_fonts = Path(windir) / "Fonts"
        paths.extend([
            str(win_fonts / "arialbd.ttf"),
            str(win_fonts / "arial.ttf"),
            str(win_fonts / "seguisb.ttf"),
            str(win_fonts / "segoeuib.ttf"),
            str(win_fonts / "segoeui.ttf"),
            str(win_fonts / "tahomabd.ttf"),
            str(win_fonts / "tahoma.ttf"),
        ])

    return paths

CLASS_INFO: Dict[int, Tuple[int, str]] = {
    0: (1, "breasts"),
    1: (2, "cleavage"),
    2: (3, "sideboob"),
    3: (4, "underboob"),
    4: (5, "hips"),
    5: (6, "thighs"),
    6: (7, "ass"),
    7: (8, "face"),
    8: (9, "head"),
}

CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (230, 57, 70),
    1: (241, 91, 181),
    2: (255, 159, 28),
    3: (46, 196, 182),
    4: (42, 157, 143),
    5: (69, 123, 157),
    6: (29, 53, 87),
    7: (131, 56, 236),
    8: (62, 180, 137),
}

INFERENCE_CONFIG = {
    "task": "detection",
    "model": "DEIM",
    "postprocessor": "PostProcessor",
    "use_focal_loss": True,
    "num_classes": 9,
    "remap_mscoco_category": False,
    "eval_spatial_size": [768, 768],
    "DEIM": {
        "backbone": "HGNetv2",
        "encoder": "HybridEncoder",
        "decoder": "DFINETransformer",
    },
    "HGNetv2": {
        "name": "B5",
        "return_idx": [1, 2, 3],
        "pretrained": False,
    },
    "HybridEncoder": {
        "in_channels": [512, 1024, 2048],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 384,
        "use_encoder_idx": [2],
        "num_encoder_layers": 1,
        "nhead": 8,
        "dim_feedforward": 2048,
        "dropout": 0.0,
        "enc_act": "gelu",
        "expansion": 1.0,
        "depth_mult": 1,
        "act": "silu",
    },
    "DFINETransformer": {
        "feat_channels": [384, 384, 384],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 256,
        "num_levels": 3,
        "nhead": 8,
        "num_layers": 6,
        "dim_feedforward": 1024,
        "eval_idx": -1,
        "num_queries": 300,
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "reg_max": 32,
        "reg_scale": 8,
        "layer_scale": 1,
        "num_points": [3, 6, 3],
        "cross_attn_method": "default",
        "query_select_method": "default",
        "activation": "relu",
        "mlp_act": "relu",
    },
    "PostProcessor": {
        "num_top_queries": 300,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference for the fixed 9-class DEIM/DFINE HGNetv2-X model and "
            "export JPEG visualizations plus YOLO-like text files."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Directory with input images.")
    parser.add_argument("--checkpoint", required=True, help="Path to DEIM checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered JPEGs and labels/.")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device to use. Defaults to cuda:0 and falls back to cpu if CUDA is unavailable.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Confidence threshold for rendering and label export.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality for rendered outputs.",
    )
    return parser.parse_args()


def fail(message: str, exit_code: int = 1) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def check_dependencies() -> None:
    missing: List[str] = []
    details: List[str] = []
    for module_name in ("torch", "torchvision", "PIL"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - exercised through CLI
            missing.append(module_name)
            details.append(f"{module_name}: {exc}")

    if missing:
        fail(
            "Missing required dependencies/imports:\n"
            + "\n".join(f"  - {detail}" for detail in details)
        )


def validate_args(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    input_dir = Path(args.input_dir).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        fail(f"--input-dir does not exist or is not a directory: {input_dir}")
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        fail(f"--checkpoint does not exist or is not a file: {checkpoint_path}")
    if not (0.0 <= args.threshold <= 1.0):
        fail(f"--threshold must be in [0, 1], got {args.threshold}")
    if not (1 <= args.jpeg_quality <= 100):
        fail(f"--jpeg-quality must be in [1, 100], got {args.jpeg_quality}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    return input_dir, checkpoint_path, output_dir


def list_images(input_dir: Path) -> List[Path]:
    images = [
        path for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        fail(f"No supported images found in {input_dir}")

    seen_stems = set()
    duplicates = []
    for path in images:
        if path.stem in seen_stems:
            duplicates.append(path.stem)
        seen_stems.add(path.stem)
    if duplicates:
        fail(
            "Found duplicate filename stems that would overwrite outputs: "
            + ", ".join(sorted(set(duplicates)))
        )
    return images


def load_runtime():
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image, ImageDraw, ImageFont

    from deim_runtime.backbone import HGNetv2
    from deim_runtime.deim import DEIM, DFINETransformer, HybridEncoder, PostProcessor

    return {
        "torch": torch,
        "nn": nn,
        "T": T,
        "Image": Image,
        "ImageDraw": ImageDraw,
        "ImageFont": ImageFont,
        "HGNetv2": HGNetv2,
        "DEIM": DEIM,
        "DFINETransformer": DFINETransformer,
        "HybridEncoder": HybridEncoder,
        "PostProcessor": PostProcessor,
    }


def resolve_device(torch_module, requested_device: str):
    try:
        device = torch_module.device(requested_device)
    except Exception as exc:
        fail(f"Invalid --device `{requested_device}`: {exc}")

    if device.type == "cuda" and not torch_module.cuda.is_available():
        print("WARNING: CUDA is unavailable, falling back to cpu.", file=sys.stderr)
        return torch_module.device("cpu")
    return device


def build_model(runtime, device):
    nn = runtime["nn"]
    cfg = copy.deepcopy(INFERENCE_CONFIG)

    backbone = runtime["HGNetv2"](
        name=cfg["HGNetv2"]["name"],
        return_idx=cfg["HGNetv2"]["return_idx"],
        pretrained=cfg["HGNetv2"]["pretrained"],
        freeze_stem_only=cfg["HGNetv2"].get("freeze_stem_only", True),
        freeze_at=cfg["HGNetv2"].get("freeze_at", 0),
        freeze_norm=cfg["HGNetv2"].get("freeze_norm", True),
    )
    encoder = runtime["HybridEncoder"](
        in_channels=cfg["HybridEncoder"]["in_channels"],
        feat_strides=cfg["HybridEncoder"]["feat_strides"],
        hidden_dim=cfg["HybridEncoder"]["hidden_dim"],
        nhead=cfg["HybridEncoder"]["nhead"],
        dim_feedforward=cfg["HybridEncoder"]["dim_feedforward"],
        dropout=cfg["HybridEncoder"]["dropout"],
        enc_act=cfg["HybridEncoder"]["enc_act"],
        use_encoder_idx=cfg["HybridEncoder"]["use_encoder_idx"],
        num_encoder_layers=cfg["HybridEncoder"]["num_encoder_layers"],
        expansion=cfg["HybridEncoder"]["expansion"],
        depth_mult=cfg["HybridEncoder"]["depth_mult"],
        act=cfg["HybridEncoder"]["act"],
        eval_spatial_size=cfg["eval_spatial_size"],
    )
    decoder = runtime["DFINETransformer"](
        num_classes=cfg["num_classes"],
        hidden_dim=cfg["DFINETransformer"]["hidden_dim"],
        num_queries=cfg["DFINETransformer"]["num_queries"],
        feat_channels=cfg["DFINETransformer"]["feat_channels"],
        feat_strides=cfg["DFINETransformer"]["feat_strides"],
        num_levels=cfg["DFINETransformer"]["num_levels"],
        num_points=cfg["DFINETransformer"]["num_points"],
        nhead=cfg["DFINETransformer"]["nhead"],
        num_layers=cfg["DFINETransformer"]["num_layers"],
        dim_feedforward=cfg["DFINETransformer"]["dim_feedforward"],
        dropout=cfg["HybridEncoder"]["dropout"],
        activation=cfg["DFINETransformer"]["activation"],
        num_denoising=cfg["DFINETransformer"]["num_denoising"],
        label_noise_ratio=cfg["DFINETransformer"]["label_noise_ratio"],
        box_noise_scale=cfg["DFINETransformer"]["box_noise_scale"],
        eval_spatial_size=cfg["eval_spatial_size"],
        eval_idx=cfg["DFINETransformer"]["eval_idx"],
        cross_attn_method=cfg["DFINETransformer"]["cross_attn_method"],
        query_select_method=cfg["DFINETransformer"]["query_select_method"],
        reg_max=cfg["DFINETransformer"]["reg_max"],
        reg_scale=cfg["DFINETransformer"]["reg_scale"],
        layer_scale=cfg["DFINETransformer"]["layer_scale"],
        mlp_act=cfg["DFINETransformer"]["mlp_act"],
    )
    train_model = runtime["DEIM"](backbone=backbone, encoder=encoder, decoder=decoder)
    postprocessor = runtime["PostProcessor"](
        num_classes=cfg["num_classes"],
        use_focal_loss=cfg["use_focal_loss"],
        num_top_queries=cfg["PostProcessor"]["num_top_queries"],
        remap_mscoco_category=cfg["remap_mscoco_category"],
    )

    class InferenceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = train_model
            self.postprocessor = postprocessor

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = InferenceModel().to(device)
    model.eval()
    return model


def load_checkpoint(runtime, model, checkpoint_path: Path, device) -> None:
    torch = runtime["torch"]

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if "ema" in checkpoint and isinstance(checkpoint["ema"], dict) and "module" in checkpoint["ema"]:
        state = checkpoint["ema"]["module"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        fail(
            "Checkpoint does not contain `ema.module` or `model`. "
            "Expected a standard DEIM training checkpoint."
        )

    try:
        model.model.load_state_dict(state)
    except RuntimeError as exc:
        fail(
            "Checkpoint is incompatible with the fixed 9-class DEIM/DFINE HGNetv2-X "
            f"inference architecture: {exc}"
        )

    model.model = model.model.deploy().to(device)
    model.postprocessor = model.postprocessor.deploy().to(device)
    model.eval()


def build_transforms(runtime):
    T = runtime["T"]
    spatial_size = tuple(INFERENCE_CONFIG["eval_spatial_size"])
    return T.Compose([
        T.Resize(spatial_size),
        T.ToTensor(),
    ])


def load_font(runtime, image_size: Sequence[int]):
    ImageFont = runtime["ImageFont"]
    font_size = max(24, min(image_size) // 24)
    for font_path in get_font_paths():
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()


def clamp_box(box: Sequence[float], width: int, height: int) -> Tuple[float, float, float, float] | None:
    x1 = min(max(float(box[0]), 0.0), float(width))
    y1 = min(max(float(box[1]), 0.0), float(height))
    x2 = min(max(float(box[2]), 0.0), float(width))
    y2 = min(max(float(box[3]), 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def export_id_and_name(label: int) -> Tuple[int, str]:
    export_id, class_name = CLASS_INFO.get(label, (label + 1, f"class_{label}"))
    return export_id, class_name


def collect_detections(labels, boxes, scores, threshold: float, width: int, height: int) -> List[Dict[str, float]]:
    detections: List[Dict[str, float]] = []
    for label_tensor, box_tensor, score_tensor in zip(labels, boxes, scores):
        score = float(score_tensor.item())
        if score < threshold:
            continue

        label = int(label_tensor.item())
        clamped = clamp_box(box_tensor.tolist(), width, height)
        if clamped is None:
            continue

        x1, y1, x2, y2 = clamped
        export_id, class_name = export_id_and_name(label)
        detections.append({
            "label": label,
            "export_id": export_id,
            "class_name": class_name,
            "score": score,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })
    return detections


def render_detections(runtime, image, detections: Iterable[Dict[str, float]]) -> None:
    ImageDraw = runtime["ImageDraw"]
    draw = ImageDraw.Draw(image)
    font = load_font(runtime, image.size)
    line_width = max(5, min(image.size) // 240)

    for det in detections:
        color = CLASS_COLORS.get(det["label"], (255, 255, 255))
        x1 = int(round(det["x1"]))
        y1 = int(round(det["y1"]))
        x2 = int(round(det["x2"]))
        y2 = int(round(det["y2"]))

        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        text = f'{det["class_name"]} {det["score"]:.2f}'
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = x1
        text_y = y1 - text_h - 10
        if text_y < 0:
            text_y = y1 + 8

        bg_box = (text_x, text_y, text_x + text_w + 14, text_y + text_h + 8)
        draw.rounded_rectangle(bg_box, radius=5, fill=color)
        draw.text((text_x + 7, text_y + 4), text=text, fill=(255, 255, 255), font=font)


def save_label_file(label_path: Path, detections: Iterable[Dict[str, float]], width: int, height: int) -> None:
    lines: List[str] = []
    for det in detections:
        box_w = det["x2"] - det["x1"]
        box_h = det["y2"] - det["y1"]
        cx = (det["x1"] + det["x2"]) / 2.0 / float(width)
        cy = (det["y1"] + det["y2"]) / 2.0 / float(height)
        bw = box_w / float(width)
        bh = box_h / float(height)
        lines.append(
            f'{det["export_id"]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {det["score"]:.6f}'
        )

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def process_image(runtime, model, transforms, device, image_path: Path, output_dir: Path, threshold: float, jpeg_quality: int) -> int:
    torch = runtime["torch"]
    Image = runtime["Image"]

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    orig_size = torch.tensor([[width, height]], device=device)
    image_tensor = transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        labels, boxes, scores = model(image_tensor, orig_size)

    detections = collect_detections(labels[0], boxes[0], scores[0], threshold, width, height)
    rendered = image.copy()
    render_detections(runtime, rendered, detections)

    output_image_path = output_dir / f"{image_path.stem}.jpg"
    output_label_path = output_dir / "labels" / f"{image_path.stem}.txt"
    rendered.save(output_image_path, format="JPEG", quality=jpeg_quality)
    save_label_file(output_label_path, detections, width, height)
    return len(detections)


def main() -> None:
    args = parse_args()
    check_dependencies()
    runtime = load_runtime()

    input_dir, checkpoint_path, output_dir = validate_args(args)
    images = list_images(input_dir)

    torch = runtime["torch"]
    device = resolve_device(torch, args.device)
    model = build_model(runtime, device)
    load_checkpoint(runtime, model, checkpoint_path, device)
    transforms = build_transforms(runtime)

    total_detections = 0
    for idx, image_path in enumerate(images, start=1):
        detections = process_image(
            runtime=runtime,
            model=model,
            transforms=transforms,
            device=device,
            image_path=image_path,
            output_dir=output_dir,
            threshold=args.threshold,
            jpeg_quality=args.jpeg_quality,
        )
        total_detections += detections
        print(f"[{idx}/{len(images)}] {image_path.name}: {detections} detections")

    print(
        f"Done. Processed {len(images)} images, exported {total_detections} detections "
        f"to {output_dir} and {output_dir / 'labels'}."
    )


if __name__ == "__main__":
    main()

"""
Convert a YOLO-style detection dataset into the COCO-like annotation format
expected by this repository's custom `CocoDetection` setup.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Category:
    original_id: int
    new_id: int
    name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a YOLO-like dataset to DEIM-compatible COCO annotations."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/workspace/train_dataset"),
        help="Dataset root containing data.yaml, images/, and labels/.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="Path to YOLO data.yaml. Defaults to <dataset-root>/data.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where COCO annotation json files will be written. Defaults to <dataset-root>/annotations.",
    )
    parser.add_argument(
        "--include-classes",
        nargs="*",
        default=None,
        help="Class ids or names to keep. Supports repeated values and comma-separated lists.",
    )
    parser.add_argument(
        "--exclude-classes",
        nargs="*",
        default=None,
        help="Class ids or names to drop. Supports repeated values and comma-separated lists.",
    )
    parser.add_argument(
        "--drop-empty-images",
        action="store_true",
        help="Drop images that have no annotations after filtering.",
    )
    parser.add_argument(
        "--train-key",
        type=str,
        default="train",
        help="Key in data.yaml that points to the training image folder.",
    )
    parser.add_argument(
        "--val-key",
        type=str,
        default="val",
        help="Key in data.yaml that points to the validation image folder.",
    )
    return parser.parse_args()


def flatten_tokens(values: Sequence[str] | None) -> List[str]:
    if not values:
        return []
    tokens: List[str] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                tokens.append(token)
    return tokens


def load_yolo_config(data_yaml_path: Path) -> Dict:
    with data_yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid yaml content in {data_yaml_path}")
    return data


def normalize_names(raw_names) -> Dict[int, str]:
    if isinstance(raw_names, dict):
        pairs = []
        for key, value in raw_names.items():
            pairs.append((int(key), str(value)))
        return dict(sorted(pairs))

    if isinstance(raw_names, list):
        return {idx: str(name) for idx, name in enumerate(raw_names)}

    raise ValueError("Expected `names` in data.yaml to be a list or dict.")


def resolve_classes(
    names_by_id: Dict[int, str],
    include_tokens: Sequence[str],
    exclude_tokens: Sequence[str],
) -> List[Category]:
    if include_tokens and exclude_tokens:
        raise ValueError(
            "--include-classes and --exclude-classes are mutually exclusive. "
            "Use include to keep only listed classes, or exclude to keep everything except listed classes."
        )

    def parse_tokens(tokens: Sequence[str]) -> List[int]:
        ids: List[int] = []
        known_names = {name: class_id for class_id, name in names_by_id.items()}
        for token in tokens:
            if token.isdigit():
                class_id = int(token)
            elif token in known_names:
                class_id = known_names[token]
            else:
                raise ValueError(
                    f"Unknown class selector `{token}`. Use one of ids {list(names_by_id.keys())} "
                    f"or names {list(names_by_id.values())}."
                )
            if class_id not in names_by_id:
                raise ValueError(f"Class id {class_id} is not present in data.yaml.")
            ids.append(class_id)
        return ids

    all_ids = list(names_by_id.keys())
    selected_ids = all_ids

    if include_tokens:
        selected_ids = parse_tokens(include_tokens)
    elif exclude_tokens:
        excluded = set(parse_tokens(exclude_tokens))
        selected_ids = [class_id for class_id in all_ids if class_id not in excluded]

    selected_ids = sorted(dict.fromkeys(selected_ids))
    if not selected_ids:
        raise ValueError("No classes remain after applying include/exclude filters.")

    categories = [
        Category(original_id=original_id, new_id=new_id, name=names_by_id[original_id])
        for new_id, original_id in enumerate(selected_ids)
    ]
    return categories


def resolve_split_dir(dataset_root: Path, config: Dict, split_key: str) -> Path:
    split_value = config.get(split_key)
    if not split_value:
        raise ValueError(f"`{split_key}` is missing from data.yaml.")
    split_path = Path(split_value)
    if split_path.is_absolute():
        return split_path
    return dataset_root / split_path


def list_images(image_dir: Path) -> List[Path]:
    images = [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(images)


def convert_yolo_box_to_coco(
    class_id: int,
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    image_width: int,
    image_height: int,
) -> Tuple[int, List[float], float] | None:
    x1 = (cx - bw / 2.0) * image_width
    y1 = (cy - bh / 2.0) * image_height
    x2 = (cx + bw / 2.0) * image_width
    y2 = (cy + bh / 2.0) * image_height

    x1 = max(0.0, min(float(image_width), x1))
    y1 = max(0.0, min(float(image_height), y1))
    x2 = max(0.0, min(float(image_width), x2))
    y2 = max(0.0, min(float(image_height), y2))

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width <= 0.0 or height <= 0.0:
        return None

    bbox = [round(x1, 4), round(y1, 4), round(width, 4), round(height, 4)]
    area = round(width * height, 4)
    return class_id, bbox, area


def read_label_file(
    label_path: Path,
    selected_id_map: Dict[int, int],
    image_width: int,
    image_height: int,
) -> List[Tuple[int, List[float], float]]:
    if not label_path.exists():
        return []

    annotations: List[Tuple[int, List[float], float]] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Invalid YOLO row in {label_path}:{line_no}: `{line}`")

            original_class_id = int(float(parts[0]))
            if original_class_id not in selected_id_map:
                continue

            cx, cy, bw, bh = [float(value) for value in parts[1:5]]
            converted = convert_yolo_box_to_coco(
                class_id=selected_id_map[original_class_id],
                cx=cx,
                cy=cy,
                bw=bw,
                bh=bh,
                image_width=image_width,
                image_height=image_height,
            )
            if converted is not None:
                annotations.append(converted)
    return annotations


def build_coco_for_split(
    image_dir: Path,
    label_dir: Path,
    split_name: str,
    categories: Sequence[Category],
    drop_empty_images: bool,
) -> Dict:
    selected_id_map = {category.original_id: category.new_id for category in categories}
    images = []
    annotations = []
    annotation_id = 0

    image_paths = list_images(image_dir)
    print(f"[{split_name}] Found {len(image_paths)} images in {image_dir}")

    for image_id, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            width, height = image.size

        label_path = label_dir / f"{image_path.stem}.txt"
        image_annotations = read_label_file(label_path, selected_id_map, width, height)
        if drop_empty_images and not image_annotations:
            continue

        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        for class_id, bbox, area in image_annotations:
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    coco = {
        "info": {
            "description": f"Converted from YOLO labels: {split_name}",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": category.new_id,
                "name": category.name,
                "original_id": category.original_id,
            }
            for category in categories
        ],
    }
    print(
        f"[{split_name}] Wrote {len(images)} images, {len(annotations)} annotations, "
        f"{len(categories)} categories"
    )
    return coco


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True)


def build_mapping_payload(categories: Sequence[Category]) -> Dict:
    return {
        "categories": [
            {
                "original_id": category.original_id,
                "new_id": category.new_id,
                "name": category.name,
            }
            for category in categories
        ],
        "original_to_new": {str(category.original_id): category.new_id for category in categories},
        "new_to_original": {str(category.new_id): category.original_id for category in categories},
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    data_yaml_path = (args.data_yaml or (dataset_root / "data.yaml")).resolve()
    output_dir = (args.output_dir or (dataset_root / "annotations")).resolve()

    yolo_config = load_yolo_config(data_yaml_path)
    names_by_id = normalize_names(yolo_config.get("names"))
    categories = resolve_classes(
        names_by_id=names_by_id,
        include_tokens=flatten_tokens(args.include_classes),
        exclude_tokens=flatten_tokens(args.exclude_classes),
    )

    train_image_dir = resolve_split_dir(dataset_root, yolo_config, args.train_key)
    val_image_dir = resolve_split_dir(dataset_root, yolo_config, args.val_key)
    train_label_dir = dataset_root / "labels" / Path(yolo_config[args.train_key]).name
    val_label_dir = dataset_root / "labels" / Path(yolo_config[args.val_key]).name

    train_coco = build_coco_for_split(
        image_dir=train_image_dir,
        label_dir=train_label_dir,
        split_name="train",
        categories=categories,
        drop_empty_images=args.drop_empty_images,
    )
    val_coco = build_coco_for_split(
        image_dir=val_image_dir,
        label_dir=val_label_dir,
        split_name="val",
        categories=categories,
        drop_empty_images=args.drop_empty_images,
    )

    train_output_path = output_dir / "instances_train.json"
    val_output_path = output_dir / "instances_val.json"
    mapping_output_path = output_dir / "category_mapping.json"

    write_json(train_output_path, train_coco)
    write_json(val_output_path, val_coco)
    write_json(mapping_output_path, build_mapping_payload(categories))

    print(f"Saved train annotations to {train_output_path}")
    print(f"Saved val annotations to {val_output_path}")
    print(f"Saved category mapping to {mapping_output_path}")


if __name__ == "__main__":
    main()

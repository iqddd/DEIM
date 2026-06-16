"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys
import os
import cv2  # Added for video processing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig

FONT_PATHS = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]


def load_font(image_size):
    font_size = max(24, min(image_size) // 24)
    for font_path in FONT_PATHS:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()


def normalize_label_name(name):
    return name.replace(' ', '_')


def build_label_map(cfg):
    if cfg.yaml_cfg.get('remap_mscoco_category', False):
        from engine.data.dataset.coco_dataset import mscoco_category2name, mscoco_label2category

        return {
            label: normalize_label_name(mscoco_category2name[category_id])
            for label, category_id in mscoco_label2category.items()
        }

    return {}


def resolve_label_name(cls_id, label_map):
    return label_map.get(cls_id, f'class_{cls_id}')


def draw(images, labels, boxes, scores, label_map, thrh=0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        font = load_font(im.size)
        line_width = max(5, min(im.size) // 240)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            x1, y1, x2, y2 = [int(v) for v in b.detach().tolist()]
            draw.rectangle((x1, y1, x2, y2), outline=(255, 48, 48), width=line_width)

            cls_id = lab[j].item()
            cls_name = resolve_label_name(cls_id, label_map)
            score = scrs[j].item()
            text = f"{cls_name} {score:.2f}"

            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            text_x = x1
            text_y = y1 - text_h - 10
            if text_y < 0:
                text_y = y1 + 10

            bg_box = (text_x, text_y, text_x + text_w + 16, text_y + text_h + 10)
            draw.rounded_rectangle(bg_box, radius=6, fill=(255, 48, 48), outline=(255, 255, 255), width=2)
            draw.text((text_x + 8, text_y + 4), text=text, fill=(255, 255, 255), font=font)

        im.save('torch_results.jpg')


def process_image(model, device, file_path, threshold, label_map):
    im_pil = Image.open(file_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    output = model(im_data, orig_size)
    labels, boxes, scores = output

    draw([im_pil], labels, boxes, scores, label_map, thrh=threshold)


def process_video(model, device, file_path, threshold, label_map):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('torch_results.mp4', fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        # Draw detections on the frame
        draw([frame_pil], labels, boxes, scores, label_map, thrh=threshold)

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'results_video.mp4'.")


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)
    label_map = build_label_map(cfg)

    # Check if the input file is an image or a video
    file_path = args.input
    if os.path.splitext(file_path)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Process as image
        process_image(model, device, file_path, args.threshold, label_map)
        print("Image processing complete.")
    else:
        # Process as video
        process_video(model, device, file_path, args.threshold, label_map)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-t', '--threshold', type=float, default=0.6)
    args = parser.parse_args()
    main(args)

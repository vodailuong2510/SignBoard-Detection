import argparse
import subprocess

def train_yolo(args):
    command = f"""
    yolo train model={args.model}
    data={args.data}
    epochs={args.epochs}
    imgsz={args.imgsz}
    batch={args.batch}
    weight_decay={args.weight_decay}
    dropout={args.dropout}
    patience={args.patience}
    crop_fraction={args.crop_fraction}
    erasing={args.erasing}
    copy_paste={args.copy_paste}
    mixup={args.mixup}
    mosaic={args.mosaic}
    bgr={args.bgr}
    perspective={args.perspective}
    shear={args.shear}
    multi_scale={args.multi_scale}
    scale={args.scale}
    translate={args.translate}
    degrees={args.degrees}
    fliplr={args.fliplr}
    flipud={args.flipud}
    hsv_v={args.hsv_v}
    hsv_s={args.hsv_s}
    hsv_h={args.hsv_h}
    """
    subprocess.run(command, shell=True, check=True)

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    parser.add_argument('--model', type=str, default='runs/detect/train13/weights/best.pt', help='Model file')
    parser.add_argument('--data', type=str, default='./dataset_signboard.yaml', help='Dataset config file')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=1024, help='Image size')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.4, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping')
    parser.add_argument('--crop_fraction', type=float, default=1.0, help='Crop fraction')
    parser.add_argument('--erasing', type=float, default=0.1, help='Erasing rate')
    parser.add_argument('--copy_paste', type=float, default=0.25, help='Copy-paste rate')
    parser.add_argument('--mixup', type=float, default=0.2, help='Mixup rate')
    parser.add_argument('--mosaic', type=float, default=0.5, help='Mosaic augmentation rate')
    parser.add_argument('--bgr', type=float, default=0.1, help='BGR color shift')
    parser.add_argument('--perspective', type=float, default=0.0001, help='Perspective distortion')
    parser.add_argument('--shear', type=int, default=20, help='Shear augmentation')
    parser.add_argument('--multi_scale', type=bool, default=True, help='Enable multi-scale training')
    parser.add_argument('--scale', type=float, default=0.1, help='Scale augmentation')
    parser.add_argument('--translate', type=float, default=0.2, help='Translation augmentation')
    parser.add_argument('--degrees', type=int, default=30, help='Rotation degrees')
    parser.add_argument('--fliplr', type=float, default=0.5, help='Flip left-right rate')
    parser.add_argument('--flipud', type=float, default=0.1, help='Flip up-down rate')
    parser.add_argument('--hsv_v', type=float, default=0.8, help='HSV value augmentation')
    parser.add_argument('--hsv_s', type=float, default=0.8, help='HSV saturation augmentation')
    parser.add_argument('--hsv_h', type=float, default=0.8, help='HSV hue augmentation')

    return parser.parse_args()
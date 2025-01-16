import random
from pathlib import Path
from PIL import Image, ImageDraw
import shutil
import os
from ultralytics import YOLO

def train_test_split(path:str, train_path:str, valid_path:str) -> None:
    os.makedirs(f'{train_path}/images', exist_ok=True)
    os.makedirs(f'{train_path}/labels', exist_ok=True)
    os.makedirs(f'{valid_path}/images', exist_ok=True)
    os.makedirs(f'{valid_path}/labels', exist_ok=True)

    path_data = Path(path)
    list_imgs_path = list(path_data.iterdir())

    len_val_data = len(list_imgs_path) // 5
    list_imgs_path_val = random.sample(list_imgs_path, len_val_data)

    for img_path in list_imgs_path:
    if img_path in list_imgs_path_val:
        shutil.copy(img_path, '{valid_path}/images')
        label_path = img_path.parent.parent / 'labels' / (img_path.stem + '.txt')
        if label_path.exists():
            shutil.copy(label_path, '{valid_path}/labels')
    else:
        shutil.copy(img_path, '{train_path}/images')
        label_path = img_path.parent.parent / 'labels' / (img_path.stem + '.txt')
        if label_path.exists():
            shutil.copy(label_path, '{train_path}/labels')

def plot_random_images(path:str, num_images:int = 5) -> None:
    path= Path(path)
    list_imgs_path = list(path.iterdir())
    list_imgs_path.sort()

    imgs_visualize = random.sample(list_imgs_path, num_images)

    for img_path in imgs_visualize:
        img = Image.open(img_path)
        width, height = img.size
        img.resize((200, 400), Image.Resampling.LANCZOS)

        label_path = img_path.parent.parent / 'labels' / (img_path.stem + '.txt')

        bounding_boxs = open(label_path, "r").readlines()
        for bbox in bounding_boxs:
            x, y, w, h = map(float, bbox.split()[1:])

            x_min = x - w / 2
            y_min = y - h / 2
            x_max = x + w / 2
            y_max = y + h / 2

            img1 = ImageDraw.Draw(img)
            img1.rectangle([(int(x_min * width), int(y_min * height)), (int(x_max * width), int(y_max * height))], outline="red", width=5)

        img.show()

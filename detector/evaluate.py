import glob
import subprocess
from PIL import Image as PILImage
import matplotlib.pyplot as plt

def predict_and_display(model_path, image_path):
    command = f"yolo mode=predict model={model_path} source={image_path}"
    subprocess.run(command, shell=True, check=True)

    for image_path in glob.glob(f"{image_path}/*.jpg"):
        img = PILImage.open(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

def predict_save_prediction(image_path, model_path, answer_path):
    list_data_test = list(Path(image_path).iterdir())
    model = YOLO(model_path)

    if Path(answer_path).exists():
        os.remove(answer_path)

    answer_path = Path(answer_path)
    answer_path.touch(exist_ok=True)

    for img_path in tqdm(list_data_test, desc="Processing images"):
        result = model([img_path])[0]

        boxes = result.boxes
        img = Image.open(img_path)
        img_draw = ImageDraw.Draw(img)
        width, height = img.size

        with open(answer_path, "a") as file:
            for bbox in boxes:
                x, y, w, h = map(float, bbox.xywh[0].tolist())
                y /= height
                w /= width
                h /= height
                file.write(f"{img_path.stem} 0 {x} {y} {w} {h}\n")


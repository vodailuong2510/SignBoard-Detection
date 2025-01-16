from detector.evaluate import predict_and_display, predict_save_prediction

if __name__ == "__main__":
    model_path = "runs/detect/train/weights/best.pt"
    image_path = "data/test/images"
    answer_path = "./results/answers.txt"

    predict_and_display(model_path, image_path)
    predict_save_prediction(image_path, model_path, answer_path)
# Sign Board Detection using YOLOv8 🚦

A computer vision project that uses **YOLOv8** to detect and localize sign boards in images with high accuracy and efficiency. This project is designed for real-time sign board detection in various environmental conditions.

## 🌟 Key Features

- **YOLOv8-based Detection**: State-of-the-art object detection using Ultralytics YOLOv8
- **High Accuracy**: Optimized for sign board detection with custom training
- **Data Augmentation**: Comprehensive augmentation pipeline for robust training
- **Easy Training**: Simple training pipeline with configurable parameters
- **Evaluation Tools**: Built-in evaluation and prediction utilities
- **Docker Support**: Containerized deployment for easy setup
- **Visualization**: Tools for visualizing detection results

## 🏗️ Project Structure

```
SignBoard-Detection/
├── train.py                 # Main training script
├── evaluate.py              # Model evaluation and prediction
├── plot.py                  # Visualization utilities
├── requirements.txt         # Python dependencies
├── dockerfile              # Docker configuration
├── detector/               # Core detection module
│   ├── model.py            # YOLO model training logic
│   ├── utils.py            # Utility functions
│   └── evaluate.py         # Evaluation functions
├── data/                   # Dataset directory
│   ├── raw/                # Raw dataset
│   │   ├── images/         # Training images
│   │   └── labels/         # YOLO format labels
│   └── test/               # Test dataset
│       └── images/         # Test images
└── results/                # Output results
    └── answers.txt         # Prediction results
```

## 🛠️ Technologies Used

### Core Technologies
- **YOLOv8**: Latest YOLO model for object detection
- **Ultralytics**: YOLO training and inference framework
- **OpenCV**: Computer vision operations
- **PIL/Pillow**: Image processing
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization

### Development & Deployment
- **Python 3.10**: Runtime environment
- **Docker**: Containerization
- **PyTorch**: Deep learning framework
- **TensorFlow**: Additional ML capabilities

## 🚀 Installation and Setup

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended for training)
- Docker (optional)

### Method 1: Local Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd SignBoard-Detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Method 2: Docker Installation

1. **Build Docker image**
```bash
docker build -t signboard-detection .
```

2. **Run container**
```bash
docker run -it --gpus all -v $(pwd):/app signboard-detection
```

## 📊 Dataset Preparation

### Data Structure
The project expects data in YOLO format:

```
data/
├── raw/
│   ├── images/          # Training images (.jpg, .png)
│   └── labels/          # YOLO format labels (.txt)
└── test/
    └── images/          # Test images
```

### Label Format
Each label file should contain bounding boxes in YOLO format:
```
class_id center_x center_y width height
```

For sign board detection:
```
0 0.5 0.5 0.3 0.2  # SignBoard class
```

## 🎯 Training

### Automatic Training
Run the main training script:
```bash
python train.py
```

This will:
1. Split data into train/validation sets
2. Create dataset configuration
3. Train YOLOv8 model with optimized parameters

### Custom Training Parameters
Modify training parameters in `train.py`:

```python
# Training configuration
dataset_info = {
    'train': './data/train',
    'val': './data/val',
    'nc': 1,                    # Number of classes
    'names': ['SignBoard'],     # Class names
}
```

### Training Arguments
Available training parameters in `detector/model.py`:
- `--epochs`: Number of training epochs (default: 20)
- `--imgsz`: Input image size (default: 1024)
- `--batch`: Batch size (default: 8)
- `--patience`: Early stopping patience (default: 100)
- `--dropout`: Dropout rate (default: 0.2)

## 🔍 Evaluation and Prediction

### Model Evaluation
```bash
python evaluate.py
```

This will:
1. Load the trained model
2. Run predictions on test images
3. Save results to `results/answers.txt`
4. Display detection visualizations

### Custom Evaluation
```python
from detector.evaluate import predict_save_prediction

# Evaluate custom model and images
predict_save_prediction(
    image_path="path/to/images",
    model_path="path/to/model.pt",
    answer_path="path/to/results.txt"
)
```

## 📈 Visualization

### View Random Images with Detections
```bash
python plot.py
```

This displays random images from the dataset with bounding boxes.

### Custom Visualization
```python
from detector.utils import plot_random_images

# Visualize custom number of images
plot_random_images(path="./data/raw/images", num_images=10)
```

## 🐳 Docker Usage

### Build and Run
```bash
# Build image
docker build -t signboard-detection .

# Run with GPU support
docker run --gpus all -it signboard-detection

# Run with volume mounting
docker run -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results signboard-detection
```

### Development with Docker
```bash
# Run with interactive shell
docker run -it signboard-detection /bin/bash

# Run specific script
docker run signboard-detection python evaluate.py
```

## 📊 Model Performance

The trained model provides:
- **High Detection Accuracy**: Optimized for sign board detection
- **Real-time Performance**: Fast inference suitable for real-time applications
- **Robust Detection**: Works in various lighting and weather conditions
- **Scalable**: Can be deployed on edge devices

## 🔧 Configuration

### Model Configuration
- **Model**: YOLOv8 (latest version)
- **Input Size**: 1024x1024 (configurable)
- **Classes**: 1 (SignBoard)
- **Augmentation**: Comprehensive augmentation pipeline

### Training Configuration
- **Optimizer**: AdamW
- **Learning Rate**: Adaptive
- **Loss Function**: YOLO loss
- **Augmentation**: Mosaic, MixUp, Copy-Paste, etc.

## 📝 Usage Examples

### Basic Training
```bash
# Start training with default parameters
python train.py
```

### Custom Training
```bash
# Train with custom parameters
python -c "
from detector.model import train_yolo, parse_args
args = parse_args()
args.epochs = 50
args.batch = 16
train_yolo(args)
"
```

### Model Prediction
```bash
# Run predictions on test images
python evaluate.py
```

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Project Link**: [https://github.com/your-username/SignBoard-Detection](https://github.com/your-username/SignBoard-Detection)
- **Email**: vodailuong2510@gmail.com

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [OpenCV](https://opencv.org/) for computer vision tools
- [PyTorch](https://pytorch.org/) for deep learning framework

## 📚 References

- [YOLOv8 Paper](https://arxiv.org/abs/2304.00501)
- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [YOLO Object Detection](https://pjreddie.com/darknet/yolo/)

from detector.utils import train_test_split
import yaml
import subprocess
from detector.model import train_yolo, parse_args
if __name__ == "__main__":
    train_path = "./data/train"
    valid_path = "./data/valid"
    data_path = "./data/raw"
    
    train_test_split(data_path, train_path, valid_path)
    
    dataset_info = {
        'train': './data/train',
        'val': './data/val',
        'nc': 1,
        'names': ['SignBoard'],
    }
    
    yamlfile_path = './data/dataset_signboard.yaml'
    
    with open(yamlfile_path, 'w') as file:
        yaml.dump(dataset_info, file, default_flow_style=None)
    
    
    if __name__ == "__main__":
        args = parse_args()
        train_yolo(args)

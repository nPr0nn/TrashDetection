
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.ops import scale_image

import os
from tqdm import tqdm 
from pathlib import Path

def run_training(dataset_folder_path, output_model_path): 
    # check GPU 
    print(ultralytics.checks())
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ.pop('TORCH_USE_CUDA_DSA', None)

    # train hyperparameters 
    YAML_PATH  = os.path.join(dataset_folder_path, 'data.yaml')
    EPOCHS     = 80
    IMG_SIZE   = 416 
    BATCHES    = 6
    PATIENCE   = 10

    model      = YOLO('yolov8n-seg.pt')

    # transfer learning with YOLOv8
    model.train( data      = YAML_PATH, 
                 epochs    = EPOCHS, 
                 imgsz     = IMG_SIZE, 
                 batch     = BATCHES, 
                 patience  = PATIENCE, 
                 optimizer ='Adam' )

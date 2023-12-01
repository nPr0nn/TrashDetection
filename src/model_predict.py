
from ultralytics import YOLO
from ultralytics.utils.ops import scale_image

from tqdm import tqdm 
from pathlib import Path

import cv2
import os
from .computer_vision import cv 
from .computer_vision import yolo_pred
# from moviepy.editor import VideoFileClip

def predict_images(images_paths, files_names, predictions_folder_path, model_path, show):
        
    for i in tqdm(range(len(images_paths))):
        path       = images_paths[i]
        file_name  = files_names[i]
        image      = cv2.imread(path)
        segmented_image, image_segments = yolo_pred.segment_image(image, model_path, (416, 416))
        
        if(show):
            cv.show(segmented_image)
                   
        # save whole image 
        segmented_image_output_path = os.path.join(predictions_folder_path, file_name)
        print(segmented_image_output_path)
        # cv2.imwrite(segmented_image_output_path, segmented_image)
                    
def run_prediction(images_folder_path, predictions_folder_path, model_path, show=False):
    images_paths, files_names = cv.read_all_images(images_folder_path) 
    predict_images(images_paths, files_names, predictions_folder_path, model_path, show) 

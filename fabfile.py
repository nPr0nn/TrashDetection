# Libraries
import os
from pathlib import Path
from fabric import task

# Source Modules
from src.computer_vision import cv
from src import model_predict
from src import model_train

drive_path = '/content/drive/Shareddrives/MC886 - Projeto Final'
dataset_path = drive_path + '/Trash Detection Dataset'
result_path = drive_path + '/Resultados'
git_path = '/content/TrashDetection'
model_path = drive_path + '/yolov8n-seg.pt'

@task
def TrainModel(c):
    dataset_folder_path = dataset_path
    output_model_path   = result_path 

    dataset_folder_path = cv.read_folder_path(dataset_folder_path)
    output_model_path   = cv.read_folder_path(output_model_path)
    
    model_train.run_training(dataset_folder_path, output_model_path) 
    
@task
def PredictImages(c):
    images_folder_path = dataset_path
    output_folder_path = result_path

    images_folder_path = cv.read_folder_path(images_folder_path)
    output_folder_path = cv.read_folder_path(output_folder_path)
    model_path         = cv.read_file_path(model_path)
    
    model_predict.run_prediction(images_folder_path, output_folder_path, model_path, show=False)

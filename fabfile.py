# Libraries
import os
from pathlib import Path
from fabric import task

# Source Modules
from src.computer_vision import cv
from src import model_predict
from src import model_train

@task
def TrainModel(dataset_folder_path=None, output_model_path=None):
    if dataset_folder_path == None:
        dataset_folder_path = "/home/lucas/Documentos/Universidade/Disciplinas/MC886/ProjetoFinal/TrashDetection/Datasets/TrashDetectionDataset"
    if output_model_path == None:
        output_model_path   = "runs" 

    dataset_folder_path = cv.read_folder_path(dataset_folder_path)
    output_model_path   = cv.read_folder_path(output_model_path)
    
    model_train.run_training(dataset_folder_path, output_model_path) 
    
@task
def PredictImages(images_folder_path=None, output_folder_path=None, model_path=None):
    if images_folder_path == None:
        images_folder_path = "/home/lucas/Documentos/Universidade/Disciplinas/MC886/ProjetoFinal/TrashDetection/Datasets/TrashDetectionDataset/test/images"
    if output_folder_path == None:
        output_folder_path = "/home/lucas/Documentos/Universidade/Disciplinas/MC886/ProjetoFinal/TrashDetection/results/test1" 
    if model_path == None:
        model_path         = "/home/lucas/Documentos/Universidade/Disciplinas/MC886/ProjetoFinal/TrashDetection/runs/segment/train5/weights/best.pt"

    images_folder_path = cv.read_folder_path(images_folder_path)
    output_folder_path = cv.read_folder_path(output_folder_path)
    model_path         = cv.read_file_path(model_path)
    
    model_predict.run_prediction(images_folder_path, output_folder_path, model_path, show=False)

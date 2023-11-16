
from ultralytics import YOLO
from ultralytics.utils.ops import scale_image
from . import cv 

import random
import numpy as np 
import cv2

# Run segmentation model on the image and return useful info
def predict_masks(model, img, conf):
    result = model(img, conf=conf)[0]

    # detection
    cls = result.boxes.cls.cpu().numpy()
    probs = result.boxes.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy() 

    # segmentation    
    if hasattr(result.masks, 'data'):
        masks = result.masks.data.cpu().numpy()
    else:
        masks = []

    return boxes, masks, cls, probs

# Draw nice colored overlay over the image 
def overlay_mask(image, mask, color, alpha):
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()
    
    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_combined, contours, -1, color, 2)  

    return image_combined

def filter_masks(masks, p=0.5):
    filtered_masks = []
    
    for mask in masks: 
        mask = mask.astype(np.uint8)
        contours, _   = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        # Filter the contours to ensure fine tunning 
        largest_area = max(cv2.contourArea(contour) for contour in contours)
        min_area_threshold = largest_area*p
        filtered_contours  = [contour for contour in contours if cv2.contourArea(contour) >= min_area_threshold] 
  
        filtered_mask = np.zeros(mask.shape)
        filtered_mask = cv2.drawContours(filtered_mask, filtered_contours, -1, 255, thickness=cv2.FILLED) 

        filtered_masks.append(filtered_mask)

    return filtered_masks

def segment_image(image, model_path, new_shape):
    model         = YOLO(model_path) 
    shape         = (image.shape[1], image.shape[0])
    model_input_size = (416, 416)
    image_resized = cv.resize_to(image, model_input_size)

    # Get important stuff
    boxes, masks, cls, probs = predict_masks(model, image_resized, conf=0.25)
    # masks = filter_masks(masks)
    masks_colors = cv.generate_palette(len(masks)) 

    image_segments = []
    image_segmented = np.copy(image_resized)
    mask_number = 0
   
    # draw masks and compute centers
    label_centers = []
    for i in range(len(masks)):
        image_segments.append(cv.apply_mask(image_segmented, masks[i]))
        image_segmented = overlay_mask(image_segmented, masks[i], color=masks_colors[i], alpha=0.3)
        cm = cv.center_of_mass(masks[i]) 
        cm = (int(cm[0]*(new_shape[0]/model_input_size[0])), int(cm[1]*(new_shape[1]/model_input_size[0])))
        label_centers.append(cm)
 
    # image_segmented = cv.resize_to(image_segmented, shape)
    image_segmented = cv.resize_to(image_segmented, new_shape)

    for i in range(len(image_segments)):
        image_segments[i] = cv.resize_to(image_segments[i], new_shape)
        image_segments[i] = cv.resize_by(image_segments[i], 50)

    # draw labels 
    for i in range(len(label_centers)):
        cm = label_centers[i]
        image_segmented = cv.draw_number(image_segmented, cm, 15, i, (255, 255, 255), 0.5, 1)
    
    return image_segmented, image_segments

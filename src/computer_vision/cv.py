
import sys
import cv2
import imutils
import numpy as np
import seaborn as sns
from pathlib import Path

#------------------------------------------------------------------------------------------------------------

# Resize the requested image using a percentage(%) factor
def resize_by(image, scale_factor):
    width, height = image.shape[1], image.shape[0]
    new_width  = int(width * scale_factor/100)
    new_height = int(height * scale_factor/100)
    dimensions = (new_width, new_height)
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

def resize_to(image, new_size):
    new_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return new_image

# Show the requested image
def show(image, title='Image'):
    cv2.namedWindow(title)
    cv2.moveWindow(title, 40,30)
    cv2.imshow(title, image)
    wait_time = 1000
    while cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
        keyCode = cv2.waitKey(wait_time)
        if(keyCode & 0xFF) == ord("q"):
            break
    cv2.destroyAllWindows()

#------------------------------------------------------------------------------------------------------------ 

def read_folder_path(folder_path):    
    if(len(folder_path) == 0):
        print("Please provide the path to a folder: ", end='')
        folder_path = input()
        folder_path = Path(folder_path)
        
        if(folder_path.is_dir() == False or folder_path.exists() == False):
            sys.exit("[ERROR - Invalid path] Please provide a valid folder path")
    
    return folder_path 

def read_file_path(file_path):    
    if(len(file_path) == 0):
        print("Please provide the path to a file: ", end='')
        file_path = input()
        file_path = Path(file_path)
        
        if(file_path.is_file() == False or file_path.exist() == False):
            sys.exit("[ERROR - Invalid path] Please provide a valid file path")
    
    return file_path 

# Read into an array all images inside the folder path
def read_all_images(path):
    path = Path(path) 
    if(path.is_dir() == False):
        sys.exit("[ERROR reading images - Invalid path] Please provide a valid folder path")      
   
    png_images  = list(path.glob('*.png'))
    jpg_images  = list(path.glob('*.jpg'))
    jpeg_images = list(path.glob('*.jpeg'))
    tiff_images = list(path.glob('*.tiff'))

    images_posix_paths = png_images + jpeg_images + jpg_images + tiff_images
    images_paths       = []
    files_names        = []
    
    for path in images_posix_paths:
        path = str(path)
        images_paths.append(path)
        files_names.append(path.split("/")[-1])

    return images_paths, files_names

def print_paths(paths):
    for path in paths:
        print(path)

#------------------------------------------------------------------------------------------------------------ 

# Filter conoturs based on their area
def filter_contours(contours, perc=0.1): 
    largest_area = max(cv2.contourArea(contour) for contour in contours)
    min_area_threshold = largest_area*perc
    filtered_contours  = [contour for contour in contours if cv2.contourArea(contour) >= min_area_threshold]  
    return filtered_contours

#------------------------------------------------------------------------------------------------------------ 

# Adjust lightness
def gamma_correction(img, gamma=1):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(img, lookUpTable)

# Blend modes
def blend_colorburn(base_image, blend_image):
    epsilon = 1e-6
    result = 1 - (1 - base_image.astype(np.float32) / 255) / (blend_image.astype(np.float32) / 255 + epsilon)
    result = np.clip(result, 0, 1) * 255
    result = result.astype(np.uint8)
    return result

def blend_overlay(base_image, blend_image, alpha=0.5):
    result = cv2.addWeighted(base_image, 1 - alpha, blend_image, alpha, 0)
    return result

def blend_vivid_light(base_image, blend_image):
    epsilon = 1e-6
    param1 = 255 - (255 - blend_image)/(2*base_image + epsilon)*255
    param2 = blend_image/(2*(255 - base_image) + epsilon)*255 
    result = np.where(base_image <= 128, param1, param2)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def blend_vivid(base_image, blend_image, alpha=0.5):
    # Normalize the images to the range [0, 1] for the blend operation
    base_image_norm  = base_image.astype(np.float32) / 255.0
    blend_image_norm = blend_image.astype(np.float32) / 255.0

    # Separate the color channels (BGR)
    base_b, base_g, base_r    = cv2.split(base_image_norm)
    blend_b, blend_g, blend_r = cv2.split(blend_image_norm)

    # Perform the Vivid blend mode operation for each channel
    result_b = np.where(blend_b <= 0.5,  2*base_b*blend_b, 1 - 2 * (1 - base_b) * (1 - blend_b))
    result_g = np.where(blend_g <= 0.5,  2*base_g*blend_g, 1 - 2 * (1 - base_g) * (1 - blend_g))
    result_r = np.where(blend_r <= 0.5,  2*base_r*blend_r, 1 - 2 * (1 - base_r) * (1 - blend_r))

    # Combine the color channels
    result = cv2.merge((result_b, result_g, result_r))

    # Apply the blend intensity
    result = alpha * result + (1 - alpha) * base_image_norm
    result = (result * 255).astype(np.uint8)
    return result

def blend_multiply(base_image, blend_image):
    # Normalize the images to the range [0, 1] for the blend operation
    base_image_norm = base_image.astype(np.float32) / 255.0
    blend_image_norm = blend_image.astype(np.float32) / 255.0

    # Perform the Multiply blend mode operation
    result = base_image_norm * blend_image_norm
    result = (result * 255).astype(np.uint8)
    return result

def remove_glare(base_image, threshold): 
    # grayscale image
    gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) 
    _, threshold = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY) 
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(opening, kernel, iterations=1) 
    
    result = cv2.inpaint(base_image, mask, 10, cv2.INPAINT_NS)
    return result

#------------------------------------------------------------------------------------------------------------ 

def crop_center(image, crop_perc_w, crop_perc_h, offset_x, offset_y):
    img_height, img_width, _ = image.shape 
    crop_width  = int(img_width * crop_perc_w)
    crop_height = int(img_height * crop_perc_h)
 
    center_x = img_width //  2 + offset_x
    center_y = img_height // 2 + offset_y
    
    left   = max(0, center_x - crop_width // 2)
    top    = max(0, center_y - crop_height // 2)
    right  = min(img_width, center_x + crop_width // 2)
    bottom = min(img_height, center_y + crop_height // 2)    
     
    cropped_image = image[top:bottom, left:right]
    return cropped_image

# TODO: Talvez precise alterar no futuro dependendo se eu quero pegar mais de um contorno na imagem :/
def apply_mask(image, mask):
    res_image = image.copy()
    res_image[np.logical_not(mask)] = [0, 0, 0]
   
    gray_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 10, 255,0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    # rect = cv2.minAreaRect(cnt)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # res_image = cv2.drawContours(res_image, [box], 0,(0,255,255),2)

    x,y,w,h = cv2.boundingRect(cnt) 
    # res_image = cv2.rectangle(res_image, (x,y),(x+w,y+h),(0,255,0),2)
    res_image = res_image[y:y+h, x:x+w] 

    return res_image

#------------------------------------------------------------------------------------------------------------ 

def generate_palette(n):
    sns.set_palette("husl", n_colors=n)
    colors = sns.color_palette()
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    return colors

def center_of_mass(binary_image):
    binary_image = binary_image.astype(np.uint8)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        
        if moments['m00'] != 0:
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00']) 
            return (center_x, center_y)

    return None

def draw_number(image, center, radius, number, color=(0,255,0), font_scale=0.5, font_thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Draw a circle on the image
    result_image = np.copy(image)
    cv2.circle(result_image, center, radius, (0,0,0), -1)

    # Draw the number inside the circle
    text_size = cv2.getTextSize(str(number), font, font_scale, font_thickness)[0]
    text_position = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)
    cv2.putText(result_image, str(number), text_position, font, font_scale, color, font_thickness)

    return result_image

def draw_text(image, center, text, color=(0, 255, 0), font_scale=0.5, font_thickness=1, rectangle_thickness=-1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Create a copy of the input image
    result_image = np.copy(image)
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    text_position = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)

    # Draw a rectangle around the text
    rectangle_start = (text_position[0] - 5, text_position[1] - text_size[1] - 5)
    rectangle_end = (text_position[0] + text_size[0] + 5, text_position[1] + 5)
    cv2.rectangle(result_image, rectangle_start, rectangle_end, (0,0,0), rectangle_thickness)

    cv2.putText(result_image, text, text_position, font, font_scale, color, font_thickness)
    return result_image

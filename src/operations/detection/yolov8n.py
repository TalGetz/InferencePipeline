import cv2
import numpy as np



def preprocess(image):
    input_img, new_h, new_w, pad_h, pad_w = resize_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    scale_h, scale_w = image.shape[0] / new_h, image.shape[1] / new_w
    input_img = input_img.astype(np.float32) / 255.0
    input_img = input_img.transpose(2, 0, 1)
    input_img = np.expand_dims(input_img, 0)

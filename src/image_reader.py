import os
import cv2
import numpy as np


def read(image_path):
    assert os.path.exists(image_path)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.uint8)
    return img

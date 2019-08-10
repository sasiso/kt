from image_reader import read
from gui import gui
from ip.find_mask import find_masks
import cv2
import numpy as np

show_ui = True

if __name__ == '__main__':
    gui = gui.Gui()
    input_image = 'a.jpeg'  # input("please input path to image")
    img = read(input_image)
    gui.add(img, "Original")
    ret, mask = find_masks(img, gui.add)
    gui.add(mask)

    gui.add(mask)
    cv2.imwrite('mask.jpeg', mask)

    if show_ui:
        gui.show()

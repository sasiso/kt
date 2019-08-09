from image_reader import read
from gui import gui
from ip.find_mask import find_masks
import cv2

if __name__ == '__main__':
    gui = gui.Gui()
    input_image = 'a.jpeg'  # input("please input path to image")
    img = read(input_image)
    gui.add(img, "Original")
    mask = find_masks(img, gui.add)
    gui.add(mask)
    overlay = cv2.add(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), mask)
    gui.add(overlay)

    gui.show()

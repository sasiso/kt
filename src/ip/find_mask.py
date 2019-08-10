import cv2
import numpy as np


def imshow_components(labels, display):
    global labeled_img
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0
    return labeled_img


def find_masks(img, display=None):
    gaussian_3 = cv2.GaussianBlur(img, (9, 9), 10.0)
    display(gaussian_3, "gaussian_3")

    unsharp_image = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0, img)
    display(unsharp_image, "unsharp_image")

    unsharp_image = cv2.medianBlur(unsharp_image, 5)
    ret, thresh5 = cv2.threshold(unsharp_image, 240, 255, cv2.THRESH_BINARY)
    display(thresh5, "thresh5")

    img = cv2.Canny(thresh5, 100, 200)
    display(img, "Canny")

    ret, labels = cv2.connectedComponents(img)

    return imshow_components(labels, display)

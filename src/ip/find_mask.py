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
    display(labeled_img, "labeled_img")
    cv2.imwrite("labeled_img.jpeg", labeled_img)
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

    for i in range(1, ret):
        try:
            sub = np.argwhere(labels == i)
            x_min = np.argmin(sub, axis=0)
            x_max = np.argmax(sub, axis=0)
            y1 = sub[x_min[0], 0]
            y2 = sub[x_max[0], 0]
            x1 = sub[x_min[1], 1]
            x2 = sub[x_max[1], 1]

            area = (x2 - x1) * (y2 - y1)
            if area > 14000 or area < 300:
                print("Skipping object of size:", area)
                continue

            cv2.rectangle(labels, (x1, y1), (x2, y2), 100)
        except Exception as ex:
            print(ex)
    imshow_components(labels, display)

    return ret, labels

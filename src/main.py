import matplotlib
import numpy as np
import cv2
from matplotlib import pyplot
import numpy as np
from PIL import Image

img = cv2.imread('../ext/kt_images/45, X, Monosomy X/ZWK99002o.jpeg')
cv2.imwrite("org.jpg", img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if img is None:
    assert False

img = img.astype(np.uint8)
org = np.copy(img)
# remove noise
img = cv2.GaussianBlur(img,ksize=(1,1), sigmaX=6)

cv2.imwrite("GaussianBlur.jpg", img)
weight = 0.9

print("Float -> ", img)
img = img * weight
output = org - (weight * img)
print("Mul -> ", img)
output = output / (1.0-weight)
print("Div -> ", img)
cv2.imwrite("after.jpg", output)

pyplot.imshow(np.asarray(output), aspect='auto', cmap='gray')
pyplot.show()

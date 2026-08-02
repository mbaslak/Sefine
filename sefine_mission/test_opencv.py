import cv2
import numpy as np

print("opencv is available. ", cv2.__version__)

img = cv2.imread('sky.jpg')

cv2.imwrite('new_sky.jpg', img)

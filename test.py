import cv2
import numpy as np
import tensorflow as tf

img = cv2.imread("1.png")
idx = ((img != 0) & (img != 255))
img[idx] = 0
img_norm = cv2.normalize(img, 0, 1, cv2.NORM_MINMAX)
idx = ((img_norm != 0) & (img_norm != 1))
print np.any(idx)

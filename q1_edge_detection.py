import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# Q1 – Canny Edge Detection


img = cv.imread('crop_field.png',0)

edges = cv.Canny(img,550,690)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img,cmap='gray')
plt.title("Original Image")

plt.subplot(1,2,2)
plt.imshow(edges,cmap='gray')
plt.title("Canny Edge Image")

plt.show()

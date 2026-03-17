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



# Q2 – Scatter Plot

indices = np.where(edges != 0)

x = indices[1]
y = indices[0]

plt.scatter(x,y,s=1)
plt.title("Scatter Plot of Edge Points")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('crop_field.png',0)

edges = cv.Canny(img,550,690)

indices = np.where(edges != 0)

x = indices[1]
y = indices[0]

m, c = np.polyfit(x,y,1)

y_fit = m*x + c

plt.scatter(x,y,s=1)
plt.plot(x,y_fit,color='red')

plt.title("Least Squares Fit")

plt.savefig("q3.png")
plt.show()

theta = np.degrees(np.arctan(m))

print("Estimated Angle:",theta)
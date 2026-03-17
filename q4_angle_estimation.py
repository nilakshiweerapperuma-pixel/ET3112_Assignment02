import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv.imread('crop_field.png',0)

# Edge detection
edges = cv.Canny(img,550,690)

# Extract edge coordinates
indices = np.where(edges != 0)

x = indices[1]
y = indices[0]

# Least squares fit
m, c = np.polyfit(x,y,1)

# Compute angle
theta = np.degrees(np.arctan(m))

print("Estimated Crop Field Angle (Least Squares):", theta, "degrees")

# Plot fitted line
y_fit = m*x + c

plt.scatter(x,y,s=1)
plt.plot(x,y_fit,color='red')

plt.title("Least Squares Fit with Angle")

plt.savefig("q4.png")
plt.show()
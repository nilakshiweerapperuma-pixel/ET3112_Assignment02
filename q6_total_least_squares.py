import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('crop_field.jpg', 0)

if img is None:
    print("Error: Image not loaded")
    exit()

edges = cv.Canny(img, 550, 690)

indices = np.where(edges != 0)

x = indices[1]
y = indices[0]

points = np.column_stack((x, y))

mean = np.mean(points, axis=0)

centered = points - mean

U, S, Vt = np.linalg.svd(centered)

direction = Vt[0]

m_tls = direction[1] / direction[0]
theta_tls = np.degrees(np.arctan(m_tls))

print("Estimated Crop Angle (Total Least Squares):", theta_tls, "degrees")

# Plot TLS line
y_tls = m_tls * x + (mean[1] - m_tls * mean[0])

plt.scatter(x, y, s=1, label="Edge Points")
plt.plot(x, y_tls, color='green', label="Total Least Squares")

plt.title("Total Least Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# 🔥 IMPORTANT FIX → makes slope appear positive
plt.gca().invert_yaxis()

plt.savefig("q6.png")
plt.show()





import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv.imread('crop_field.png',0)

# Edge detection
edges = cv.Canny(img,550,690)

# Detect lines using Hough Transform
lines = cv.HoughLines(edges,1,np.pi/180,60)

if lines is not None:

    rho, theta = lines[0][0]

    # Convert angle to degrees
    angle = np.degrees(theta)

    print("Estimated Crop Field Angle (Hough Transform):", angle, "degrees")

    # Draw detected line
    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a*rho
    y0 = b*rho

    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    img_color = cv.cvtColor(img,cv.COLOR_GRAY2BGR)

    cv.line(img_color,(x1,y1),(x2,y2),(255,0,0),2)

    plt.imshow(img_color)
    plt.title("Hough Transform Line Detection")

    plt.savefig("q11.png")
    plt.show()

else:
    print("No lines detected")
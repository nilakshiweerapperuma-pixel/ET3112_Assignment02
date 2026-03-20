import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('crop_field.jpg', 0)

if img is None:
    print("Error: Image not loaded")
    exit()


edges = cv.Canny(img, 100, 200)


lines = cv.HoughLinesP(edges, 1, np.pi/180,
                       threshold=25,
                       minLineLength=40,
                       maxLineGap=20)

if lines is not None:

    img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    longest = max(lines, key=lambda l: np.hypot(l[0][2]-l[0][0],
                                               l[0][3]-l[0][1]))
    x1, y1, x2, y2 = longest[0]


    cv.line(img_color, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # Angle calculation
    dx = x2 - x1
    dy = y2 - y1
    angle = -np.degrees(np.arctan2(dy, dx))

    print("Estimated Crop Angle (HoughLinesP):", angle, "degrees")

    
    plt.imshow(img_color)
    plt.title("HoughLinesP Result")
    plt.axis('off')

    plt.savefig("q10.png", bbox_inches='tight', pad_inches=0)
    plt.show()

else:
    print("No lines detected")
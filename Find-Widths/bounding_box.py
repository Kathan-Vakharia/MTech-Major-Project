import cv2
import numpy as np

# Load the grayscale image
img = cv2.imread('1_aoi.png', 0)

# Binarize the image
_, binary_img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

# Find the contours in the binary image
contours, _ = cv2.findContours(
    binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the bounding box for the object
x, y, w, h = cv2.boundingRect(contours[0])

# Draw the bounding box on the original image
cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Save the image with the bounding box
cv2.imwrite('image_with_bbox.png', img)

# Print the coordinates of the bounding box corners
print(f"x: {x}, y:{y}, w:{w}, h:{h}")
print(f"Top-left corner: ({x}, {y})")
print(f"Top-right corner: ({x + w}, {y})")
print(f"Bottom-left corner: ({x}, {y + h})")
print(f"Bottom-right corner: ({x + w}, {y + h})")

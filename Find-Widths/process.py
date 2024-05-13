import cv2
import numpy as np
import bounding_box

# Load the grayscale image
input_img_path = './Input-Images/1_aoi.png'
img = cv2.imread(input_img_path, 0)

# Binarize the image
_, binary_img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)


# Get the bounding box coordinates of AOI
bounding_box_coordinates = bounding_box.get_bounding_box(input_img_path)
x, y, w, h = bounding_box_coordinates  # Replace with the actual coordinates

# Initialize an empty list to store the widths
widths = []

# Iterate over the rows within the bounding box with a step size of 20
for row in range(y, y + h, 20):
    # Initialize D1 and D2 to the bounding box width
    d1 = d2 = w

    # Find the first white pixel from the left
    for col in range(x, x + w):
        if binary_img[row, col] == 255:
            d1 = col - x
            break

    # Find the first white pixel from the right
    for col in range(x + w - 1, x - 1, -1):
        if binary_img[row, col] == 255:
            d2 = x + w - 1 - col
            break

    # Calculate the width and append it to the list
    width = abs(w- d1 - d2)
    widths.append(width)

    # Draw a line representing the width
    cv2.line(img, (x + d1, row), (x + w - d2, row), (255, 0, 0), 1)

# Print the widths
for i, width in enumerate(widths):
    print(f"Width at row {y + i * 20}: {width}")

# Save the image with the lines drawn
cv2.imwrite('image_with_lines.png', img)

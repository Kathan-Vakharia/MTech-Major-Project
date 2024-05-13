import cv2
import numpy as np
import utils

# Load the grayscale image
input_img_path = 'Input-Images/1_aoi.png'
input_nerve_img_path = 'Input-Images/1_aoi_mask.png'
img = cv2.imread(input_img_path, 0)

# Binarize the image
_, binary_img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
# print(np.unique(binary_img))
# cv2.imwrite('binary_image.png', binary_img)

# Get the bounding box coordinates
bounding_box_coordinates = utils.get_bounding_box(input_img_path)
x, y, w, h = bounding_box_coordinates  
nerve_top = utils.get_bounding_box(input_nerve_img_path)[1] # top 

# Initialize an empty list to store the base-line widths
widths = []
ppl = 0.19 # per pixel length in mm
base_line_bounds = []

# Iterate over the rows within the bounding box 
for row in range(y, y + h):
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
    size = width * ppl
    if size >= 4 and len(widths) == 0:
        base_line_bounds.append({"row": row, "start":x + d1, "end": x + w - d2})
        widths.append(width)
        cv2.line(img, (x + d1, row), (x + w - d2, row), (255, 0, 0), 1)
    
    if abs(row-nerve_top+1)*ppl <= 2:
        base_line_bounds.append({"row": row, "start":x + d1, "end": x + w - d2})
        widths.append(width)
        cv2.line(img, (x + d1, row), (x + w - d2, row), (255, 0, 0), 1)
        break


img = utils.draw_line_from_midpoints(img, base_line_bounds)



# # Print the widths
print(base_line_bounds)
print(widths)

# Draw the lines between the midpoints of the base line bounds
output_img_path = 'image_with_baselines.png'
cv2.imwrite(output_img_path, img)


# Save the image with the lines drawn
cv2.imwrite('image_with_baselines.png', img)

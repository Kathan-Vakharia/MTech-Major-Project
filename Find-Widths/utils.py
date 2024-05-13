import cv2
import numpy as np
import cv2

def get_bounding_box(image_path):
    # Load the grayscale image
    img = cv2.imread(image_path, 0)

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
    # cv2.imwrite(f"{image_path}__bb.png", img)

    # Print the coordinates of the bounding box
    print(f"x: {x}, y:{y}, w:{w}, h:{h}")
    print(f"Top-left corner: ({x}, {y})")
    print(f"Top-right corner: ({x + w}, {y})")
    print(f"Bottom-left corner: ({x}, {y + h})")
    print(f"Bottom-right corner: ({x + w}, {y + h})")

    # Return the coordinates of the bounding box
    return x, y, w, h


def draw_line_from_midpoints(img, base_line_bounds):
    """
    Draw line on the image between the midpoints of the base line bounds
    img: the image to draw the line on
    base_line_bounds: list of dictionaries containing the row, start, and end of the base line
    """
    
    mid1 =(base_line_bounds[0]["start"] + base_line_bounds[0]["end"]) // 2, base_line_bounds[0]["row"]
    mid2 = (base_line_bounds[1]["start"] + base_line_bounds[1]["end"]) // 2, base_line_bounds[1]["row"]

    # Draw a line between the midpoints
    cv2.line(img, mid1, mid2, (255, 0, 0), 1)

    # Save the image with the lines
    return img

# Usage
if __name__ == "__main__":
    x, y, w, h = get_bounding_box('Input-Images/1_aoi_mask.png')




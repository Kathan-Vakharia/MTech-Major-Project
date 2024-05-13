import cv2

def multiply_images(image1_path, image2_path, output_path):
    # Read the images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Perform element-wise multiplication
    result = cv2.add(image1, image2)

    # Save the result
    cv2.imwrite(output_path, result)

# Example usage
image1_path = 'image_with_baselines.png'
image2_path = '1_aoi_mask.png'
output_path = 'output.png'

multiply_images(image1_path, image2_path, output_path)
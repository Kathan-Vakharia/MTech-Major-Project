from data import create_dir, augment_data
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


def create_aoi_dataset():
    # Set the paths for input and output folders
    images_folder = 'kmc-dental-data/images'
    masks_folder = 'kmc-dental-data/masks'
    output_folder = 'kmc-dental-data-aoi/aoi_images'
    create_dir(output_folder)  # create output data folder

    image_filenames = sorted(os.listdir(images_folder))
    mask_filenames = sorted(os.listdir(masks_folder))
    # print(image_filenames,mask_filenames, sep="\n--")

    # no. of images should be equal to no. of masks
    assert len(image_filenames) == len(mask_filenames)

    for i in range(len(image_filenames)):
        img_fname = image_filenames[i]
        msk_fname = mask_filenames[i]

        image = cv2.imread(os.path.join(
            images_folder, img_fname), cv2.IMREAD_GRAYSCALE)
        image = image.astype("float64")

        mask = cv2.imread(os.path.join(
            masks_folder, msk_fname), cv2.IMREAD_GRAYSCALE)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        mask = mask/255

        assert mask.shape == image.shape
        result = image*mask # element-wise multiplication
        
        output_name = os.path.splitext(img_fname)[0] + '_aoi.png'
        output_path = os.path.join(output_folder, output_name)
        cv2.imwrite(output_path, result)
        
        # print(img_fname, msk_fname)
        # print(np.unique(mask))


def load_data(path, split=0.1):
    # load images, masks
    X = sorted(glob(os.path.join(path, "aoi_images", "*.png")))
    Y = sorted(glob(os.path.join(path, "aoi_masks", "*.png")))

    # check if the number of images and masks are equal
    assert len(X) == len(Y)

    # split the dataset
    split_size = int(len(X) * split)

    train_x, test_x = train_test_split(
        X, test_size=split_size, random_state=50)
    train_y, test_y = train_test_split(
        Y, test_size=split_size, random_state=50)

    return (train_x, train_y), (test_x, test_y)


if __name__ == "__main__":
    np.random.seed(50)

    # Create the aoi data
    # create_aoi_dataset()


    # # Load the dataset
    data_path = "kmc-dental-data-aoi"

    (train_x, train_y), (test_x, test_y) = load_data(data_path, split=0.2)
    print("Train: ", len(train_x), len(train_y))
    print("Test: ", len(test_x), len(test_y))

    # create directories to save augmented data
    create_dir("new_data_aoi/train/images")
    create_dir("new_data_aoi/train/masks")
    create_dir("new_data_aoi/test/images")
    create_dir("new_data_aoi/test/masks")

    # Data Augmentation
    augment_data(train_x, train_y, "new_data_aoi/train/", augment=True)
    augment_data(test_x, test_y, "new_data_aoi/test/", augment=False)

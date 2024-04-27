import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, GridDistortion, OpticalDistortion, RandomSizedCrop, Compose, ChannelShuffle, HueSaturationValue, RandomBrightnessContrast, RandomGamma, ShiftScaleRotate, Blur, GaussNoise, MotionBlur, MedianBlur, ElasticTransform, IAAPiecewiseAffine, IAASharpen, Rotate, RandomContrast, RandomBrightness, RandomRotate90, RandomSizedCrop, OneOf, Resize, RandomCrop, Normalize, CoarseDropout, CenterCrop, GaussianBlur
# Creates a directory


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path, split=0.15):
    # load images, masks
    X = sorted(glob(os.path.join(path, "images", "*.png")))
    Y = sorted(glob(os.path.join(path, "masks", "*.png")))

    # split the dataset
    split_size = int(len(X) * split)

    train_x, test_x = train_test_split(
        X, test_size=split_size, random_state=50)
    train_y, test_y = train_test_split(
        Y, test_size=split_size, random_state=50)

    return (train_x, train_y), (test_x, test_y)

# Data Augmentation


def augment_data(images, masks, save_path, augment=True):
    """
    images: list of image file paths
    masks: list of mask file paths
    save_path: path to save augmented images and masks
    augment: boolean, whether to apply data augmentation or not
    """

    H = 512
    W = 512

    for x, y in tqdm(zip(images, masks), total=len(images)):
        # extract file name
        name = x.split("\\")[-1].split(".")[0]
        print(name)

        # read image and mask
        x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        y = cv2.threshold(y, 127, 255, cv2.THRESH_BINARY)[1]  # remove noise
        print(y.shape, x.shape)

        # Augmentation
        if augment:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented['image']
            y1 = augmented['mask']

            aug = CoarseDropout(p=1, min_holes=3, max_holes=8,
                                max_height=32, max_width=32)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = RandomBrightnessContrast(p=1)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = MotionBlur(p=1)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            aug = GaussNoise(p=1)
            augmented = aug(image=x, mask=y)
            x6 = augmented['image']
            y6 = augmented['mask']

            aug = GaussianBlur(p=1)
            augmented = aug(image=x, mask=y)
            x7 = augmented['image']
            y7 = augmented['mask']

            aug = ElasticTransform(
                p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            augmented = aug(image=x, mask=y)
            x8 = augmented['image']
            y8 = augmented['mask']

            aug = OpticalDistortion(p=1, distort_limit=3, shift_limit=0.5)
            augmented = aug(image=x, mask=y)
            x9 = augmented['image']
            y9 = augmented['mask']

            X = [x, x1, x2, x3, x4, x5, x6, x7, x8, x9]
            Y = [y, y1, y2, y3, y4, y5, y6, y7, y8, y9]

        else:
            X = [x]
            Y = [y]

        idx = 0
        for img, msk in zip(X, Y):
            try:
                aug = CenterCrop(p=1, height=H, width=W)
                augmented = aug(image=img, mask=msk)
                img = augmented['image']
                msk = augmented['mask']
            except:
                img = cv2.resize(img, (W, H))
                msk = cv2.resize(msk, (W, H))

            # save images
            tmp_img_name = f"{name}_{idx}.png"
            tmp_msk_name = f"{name}_{idx}.png"

            image_path = os.path.join(save_path, "images", tmp_img_name)
            mask_path = os.path.join(save_path, "masks", tmp_msk_name)

            cv2.imwrite(image_path, img)
            cv2.imwrite(mask_path, msk)
            idx += 1


if __name__ == "__main__":
    np.random.seed(50)

    # Load the dataset
    data_path = "kmc-dental-data"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)
    print("Train: ", len(train_x), len(train_y))
    print("Test: ", len(test_x), len(test_y))

    # create directories to save augmented data
    create_dir("new_data/train/images")
    create_dir("new_data/train/masks")
    create_dir("new_data/test/images")
    create_dir("new_data/test/masks")

    # Data Augmentation
    augment_data(train_x, train_y, "new_data/train/", augment=True)
    augment_data(test_x, test_y, "new_data/test/", augment=False)

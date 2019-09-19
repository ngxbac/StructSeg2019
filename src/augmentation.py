from albumentations import *

import itertools


def train_aug(image_size=224):
    # return Compose([
    #     HorizontalFlip(),
    #     Normalize()
    # ],p=1)

    return Compose([
        Resize(image_size, image_size),
        HorizontalFlip(p=0.5),
        # OneOf([
        #     RandomContrast(),
        #     RandomGamma(),
        #     RandomBrightness(),
        # ], p=0.3),
        OneOf([
            ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(),
            OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15),
        Normalize(max_pixel_value=1)
        # RandomSizedCrop(min_max_height=(156, 256), height=h, width=w, p=0.25),
        # ToFloat(max_value=1)
    ], p=1)


def valid_aug(image_size=224):
    return Compose([
        Resize(image_size, image_size),
        Normalize(max_pixel_value=1)
    ], p=1)


def test_tta(image_size):
    test_dict = {
        'normal': Compose([
            Resize(image_size, image_size)
        ]),
        # 'hflip': Compose([
        #     HorizontalFlip(p=1),
        #     Resize(image_size, image_size),
        # ], p=1),
        # 'rot90': Compose([
        #     Rotate(limit=(90, 90), p=1),
        #     Resize(image_size, image_size),
        # ], p=1),
    }

    return test_dict
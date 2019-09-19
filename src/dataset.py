import numpy as np
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import SimpleITK
import scipy.ndimage as ndimage
import SimpleITK as sitk


UPPER_BOUND = 400
LOWER_BOUND = -1000


def load_ct_images(path):
    image = SimpleITK.ReadImage(path)
    spacing = image.GetSpacing()[-1]
    image = SimpleITK.GetArrayFromImage(image).astype(np.float32)
    return image, spacing


def load_itkfilewithtrucation(filename, upper=200, lower=-200):
    """
    load mhd files,set truncted value range and normalization 0-255
    :param filename:
    :param upper:
    :param lower:
    :return:
    """
    # 1,tructed outside of liver value
    srcitkimage = sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32)
    srcitkimagearray = sitk.GetArrayFromImage(srcitkimage)
    srcitkimagearray[srcitkimagearray > upper] = upper
    srcitkimagearray[srcitkimagearray < lower] = lower
    # 2,get tructed outside of liver value image
    sitktructedimage = sitk.GetImageFromArray(srcitkimagearray)
    origin = np.array(srcitkimage.GetOrigin())
    spacing = np.array(srcitkimage.GetSpacing())
    sitktructedimage.SetSpacing(spacing)
    sitktructedimage.SetOrigin(origin)
    # 3 normalization value to 0-255
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    itkimage = rescalFilt.Execute(sitk.Cast(sitktructedimage, sitk.sitkFloat32))
    return sitk.GetArrayFromImage(itkimage)


def resize(image, mask, spacing, slice_thickness, scale_ratio):
    image = (image - LOWER_BOUND) / (UPPER_BOUND - LOWER_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    image = image.astype(np.float32)

    if slice_thickness and scale_ratio:
        image = ndimage.zoom(image, (spacing / slice_thickness, scale_ratio, scale_ratio), order=3)
        mask = ndimage.zoom(mask, (spacing / slice_thickness, scale_ratio, scale_ratio), order=0)
    return image, mask


def load_patient(imgpath, mskpath, slice_thickness=None, scale_ratio=None):
    image, spacing = load_ct_images(imgpath)

    mask, _ = load_ct_images(mskpath)
    image, mask = resize(image, mask, spacing, slice_thickness, scale_ratio)
    return image, mask


def pad_if_need(image, mask, patch):
    assert image.shape == mask.shape

    n_slices, x, y = image.shape
    if n_slices < patch:
        padding = patch - n_slices
        offset = padding // 2
        image = np.pad(image, (offset, patch - n_slices - offset), 'edge')
        mask = np.pad(mask, (offset, patch - n_slices - offset), 'edge')

    return image, mask


def slice_window(image, mask, slice, patch):
    image, mask = pad_if_need(image, mask, patch)
    n_slices, x, y = image.shape
    idx = 0

    image_patches = []
    mask_patches = []

    while idx + patch <= n_slices:
        image_patch = image[idx:idx + patch]
        mask_patch = mask[idx:idx + patch]

        # Save patch
        image_patches.append(image_patch)
        mask_patches.append(mask_patch)

        idx += slice

    return image_patches, mask_patches


def slice_builder(imgpath, mskpath, slice_thichness, scale_ratio, slice, patch, save_dir):
    image, mask = load_patient(imgpath, mskpath, slice_thichness, scale_ratio)
    image_patches, mask_patches = slice_window(image, mask, slice, patch)
    patient_id = imgpath.split("/")[-2]
    save_dir = os.path.join(save_dir, patient_id)
    os.makedirs(save_dir, exist_ok=True)

    image_paths = []
    mask_paths = []
    for i, (image_patch, mask_patch) in enumerate(zip(image_patches, mask_patches)):
        image_path = os.path.join(save_dir, f'image.{i}.npy')
        mask_path = os.path.join(save_dir, f'mask.{i}.npy')

        image_paths.append(image_path)
        mask_paths.append(mask_path)

        np.save(image_path, image_patch)
        np.save(mask_path, mask_patch)

    df = pd.DataFrame({
        'image': image_paths,
        'mask': mask_paths
    })

    df['patient_id'] = patient_id
    return df


def slice_builder_2d(imgpath, mskpath, save_dir):
    image, mask = load_patient(imgpath, mskpath)
    patient_id = imgpath.split("/")[-2]
    save_dir = os.path.join(save_dir, patient_id)
    os.makedirs(save_dir, exist_ok=True)

    image_paths = []
    mask_paths = []
    for i, (image_slice, mask_slice) in enumerate(zip(image, mask)):
        # if np.any(mask_slice):
        image_path = os.path.join(save_dir, f'image.{i}.npy')
        mask_path = os.path.join(save_dir, f'mask.{i}.npy')

        image_paths.append(image_path)
        mask_paths.append(mask_path)

        np.save(image_path, image_slice)
        np.save(mask_path, mask_slice)

    df = pd.DataFrame({
        'image': image_paths,
        'mask': mask_paths
    })

    df['patient_id'] = patient_id
    return df


def random_crop(image, mask, patch):
    n_slices = image.shape[0]
    start = 0
    end = int(n_slices - patch)

    rnd_idx = np.random.randint(start, end)
    return image[rnd_idx:rnd_idx + patch, :, :], mask[rnd_idx:rnd_idx + patch, :, :]


def center_crop(image, mask, patch):
    n_slices = image.shape[0]
    mid = n_slices // 2
    start = int(mid - patch // 2)
    end = int(mid + patch // 2)

    return image[start:end, :, :], mask[start:end, :, :]


class StructSegTrain2D(Dataset):

    def __init__(self,
                 csv_file,
                 transform
                 ):
        df = pd.read_csv(csv_file)
        self.transform = transform
        self.images = df['image'].values
        self.masks = df['mask'].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]
        mask = self.masks[idx]

        image = np.load(image)
        mask = np.load(mask)

        image = np.stack((image, image, image), axis=-1).astype(np.float32)

        if self.transform:
            transform = self.transform(image=image, mask=mask)
            image = transform['image']
            mask = transform['mask']

        # image = np.stack((image, image, image), axis=0).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        # mask = np.transpose(mask, (2, 0, 1))

        # image = np.expand_dims(image, axis=0)
        mask = mask.astype(np.int)

        return {
            'images': image,
            'targets': mask
        }


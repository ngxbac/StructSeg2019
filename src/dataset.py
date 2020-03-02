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


def cut_edge(data, keep_margin):
    '''
    function that cuts zero edge
    '''
    D, H, W = data.shape
    D_s, D_e = 0, D - 1
    H_s, H_e = 0, H - 1
    W_s, W_e = 0, W - 1

    while D_s < D:
        if data[D_s].sum() != 0:
            break
        D_s += 1
    while D_e > D_s:
        if data[D_e].sum() != 0:
            break
        D_e -= 1
    while H_s < H:
        if data[:, H_s].sum() != 0:
            break
        H_s += 1
    while H_e > H_s:
        if data[:, H_e].sum() != 0:
            break
        H_e -= 1
    while W_s < W:
        if data[:, :, W_s].sum() != 0:
            break
        W_s += 1
    while W_e > W_s:
        if data[:, :, W_e].sum() != 0:
            break
        W_e -= 1

    if keep_margin != 0:
        D_s = max(0, D_s - keep_margin)
        D_e = min(D - 1, D_e + keep_margin)
        H_s = max(0, H_s - keep_margin)
        H_e = min(H - 1, H_e + keep_margin)
        W_s = max(0, W_s - keep_margin)
        W_e = min(W - 1, W_e + keep_margin)

    return int(D_s), int(D_e), int(H_s), int(H_e), int(W_s), int(W_e)



import random
class StructSegTrain3D(Dataset):

    def __init__(self,
                 csv_file,
                 transform,
                 mode='train'
                 ):
        df = pd.read_csv(csv_file)
        self.transform = transform
        self.patients = df['patient_id'].unique()
        self.root = "/data/Thoracic_OAR/"
        self.crop_size = (16, 256, 256)
        self.mode = mode

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):

        patient_id = self.patients[idx]
        data_path = os.path.join(self.root, str(patient_id), 'data.nii.gz')
        label_path = os.path.join(self.root, str(patient_id), 'label.nii.gz')

        image, mask = load_patient(data_path, label_path)
        # image[image < 0.5] = 0
        # image[image >= 0.5] = 1
        # # image2 = image > 0
        # margin = 32
        # min_D_s, max_D_e, min_H_s, max_H_e, min_W_s, max_W_e = cut_edge(image, margin)
        #
        # image = image[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1]
        # mask = mask[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1]

        # print(image.shape)

        D, H, W = image.shape

        if self.mode == 'train':
            rd = random.randint(0, D - self.crop_size[0])
            rh = random.randint(0, H - self.crop_size[1])
            rw = random.randint(0, W - self.crop_size[2])

            # rd = (D - self.crop_size[0]) // 2
            # rh = (H - self.crop_size[1]) // 2
            # rw = (W - self.crop_size[2]) // 2
        else:
            rd = (D - self.crop_size[0]) // 2
            rh = (H - self.crop_size[1]) // 2
            rw = (W - self.crop_size[2]) // 2

        image = image[rd: rd + self.crop_size[0], rh: rh + self.crop_size[1], rw: rw + self.crop_size[2]]
        mask = mask[rd: rd + self.crop_size[0], rh: rh + self.crop_size[1], rw: rw + self.crop_size[2]]

        image = np.expand_dims(image, axis=0).astype(np.float32)
        mask = mask.astype(np.int)
        # mask = np.expand_dims(mask, axis=0).astype(np.int)

        return {
            'images': image,
            'targets': mask
        }
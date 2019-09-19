import numpy as np
import glob
import os
import cv2
import pandas as pd
import SimpleITK
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import SimpleITK
from augmentation import valid_aug
from models import UNet3D
# from segmentation_models_pytorch.unet import Unet


import scipy.ndimage as ndimage


UPPER_BOUND = 400
LOWER_BOUND = -1000

expand_slice = 20  # 轴向上向外扩张的slice数量
size = 16  # 取样的slice数量
stride = 3  # 取样的步长
down_scale = 0.5
slice_thickness = 2


device = torch.device('cuda')


def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for dct in loader:
            images = dct['images'].to(device)
            pred = model(images)
            pred = F.sigmoid(pred)
            # pred = pred.squeeze()
            pred = pred.detach().cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    return preds


class TestDataset(Dataset):
    def __init__(self, image_slices, transform):
        self.image_slices = image_slices
        self.transform = transform

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        image = self.image_slices[idx]
        # image = np.stack((image, image, image), axis=-1).astype(np.float32)

        if self.transform:
            transform = self.transform(image=image)
            image = transform['image']

        image = np.expand_dims(image, axis=0).astype(np.float32)

        return {
            'images': image
        }


def extract_slice(file):
    ct_image = SimpleITK.ReadImage(file)
    image = SimpleITK.GetArrayFromImage(ct_image)

    image[image > UPPER_BOUND] = UPPER_BOUND
    image[image < LOWER_BOUND] = LOWER_BOUND

    image = ndimage.zoom(image, (ct_image.GetSpacing()[-1] / slice_thickness, down_scale, down_scale), order=3)
    print("Image shape: ", image.shape)
    n_slices = image.shape[0]

    # flag = False
    # start_slice = 0
    # end_slice = start_slice + size - 1
    # ct_array_list = []
    image_slices = []
    idx = 0
    while idx < n_slices - size - 1:
        image_slices.append(image[idx:idx + size, :, :])
        idx += size

    return image_slices, n_slices, ct_image


def predict_valid():
    inputdir = "../Lung_GTV/"
    outdir = "../Lung_GTV_val_pred/190917/Unet3D-bs4-0/"

    transform = valid_aug(image_size=512)

    # nii_files = glob.glob(inputdir + "/*/data.nii.gz")
    threshold = 0.5

    folds = [0]

    for fold in folds:
        log_dir = f"../logs/190918/Unet3D-bs4-fold-{fold}"
        model = UNet3D(
            in_channels=1,
            out_channels=1,
            f_maps=64
        )

        ckp = os.path.join(log_dir, "checkpoints/best.pth")
        checkpoint = torch.load(ckp)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = nn.DataParallel(model)
        model = model.to(device)

        df = pd.read_csv(f'./csv/5folds/valid_{fold}.csv')
        patient_ids = df.patient_id.values
        for patient_id in patient_ids:
            print(patient_id)
            nii_file = f"{inputdir}/{patient_id}/data.nii.gz"

            image_slices, n_slices, ct_image = extract_slice(nii_file)

            # import pdb
            # pdb.set_trace()

            dataset = TestDataset(image_slices, None)
            dataloader = DataLoader(
                dataset=dataset,
                num_workers=4,
                batch_size=2,
                drop_last=False
            )

            pred_mask = predict(model, dataloader)

            # pred_mask = torch.FloatTensor(pred_mask)
            # pred_mask = F.upsample(pred_mask, (size, 512, 512), mode='trilinear').detach().cpu().numpy()
            pred_mask = (pred_mask > threshold).astype(np.int16)
            # pred_mask = pred_mask.reshpae(-1, 512, 512)
            pred_mask = np.transpose(pred_mask, (1, 0, 2, 3, 4))
            pred_mask = pred_mask[0]
            pred_mask = pred_mask.reshape(-1, 256, 256)
            count = n_slices - pred_mask.shape[0]
            if count > 0:
                pred_mask = np.concatenate([pred_mask, pred_mask[-count:, :, :]], axis=0)

            pred_mask = ndimage.zoom(pred_mask, (slice_thickness / ct_image.GetSpacing()[-1], 1/down_scale, 1/down_scale), order=3)

            pred_mask = SimpleITK.GetImageFromArray(pred_mask)
            pred_mask.SetDirection(ct_image.GetDirection())
            pred_mask.SetOrigin(ct_image.GetOrigin())
            pred_mask.SetSpacing(ct_image.GetSpacing())

            # patient_id = nii_file.split("/")[-2]
            patient_dir = f"{outdir}/{patient_id}"
            os.makedirs(patient_dir, exist_ok=True)
            patient_pred = f"{patient_dir}/predict.nii.gz"
            SimpleITK.WriteImage(
                pred_mask, patient_pred
            )


if __name__ == '__main__':
    predict_valid()

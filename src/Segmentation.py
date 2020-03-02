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
from segmentation_models_pytorch.unet import Unet
from segmentation_models_pytorch import FPN
from models import VNet


import scipy.ndimage as ndimage


UPPER_BOUND = 400
LOWER_BOUND = -1000


device = torch.device('cuda')


def predict(model, loader):
    model.eval()
    preds = []
    pred_logits = []
    with torch.no_grad():
        for dct in loader:
            images = dct['images'].to(device)
            pred = model(images)
            pred_sofmax = F.softmax(pred, dim=1)
            pred_sofmax = pred_sofmax.detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()
            preds.append(pred_sofmax)
            pred_logits.append(pred)

    preds = np.concatenate(preds, axis=0)
    pred_logits = np.concatenate(pred_logits, axis=0)
    return preds, pred_logits


class TestDataset(Dataset):
    def __init__(self, image_slices, transform):
        self.image_slices = image_slices
        self.transform = transform

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        image = self.image_slices[idx]
        image = np.stack((image, image, image), axis=-1).astype(np.float32)

        if self.transform:
            transform = self.transform(image=image)
            image = transform['image']

        image = np.transpose(image, (2, 0, 1))

        return {
            'images': image
        }


class TestDatasetNB(Dataset):
    def __init__(self, image_slices, transform):
        self.image_slices = image_slices
        self.transform = transform

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        image = self.image_slices[idx]
        if idx == 0:
            image_prev = image
        else:
            image_prev = self.image_slices[idx-1]

        if idx == len(self.image_slices) - 1:
            image_next = image
        else:
            image_next = self.image_slices[idx + 1]

        image = np.stack((image_prev, image, image_next), axis=-1).astype(np.float32)

        if self.transform:
            transform = self.transform(image=image)
            image = transform['image']

        image = np.transpose(image, (2, 0, 1))

        return {
            'images': image
        }


def extract_slice(file):
    ct_image = SimpleITK.ReadImage(file)
    image = SimpleITK.GetArrayFromImage(ct_image).astype(np.float32)

    image = (image - LOWER_BOUND) / (UPPER_BOUND - LOWER_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    image = image.astype(np.float32)

    image_slices = []
    for i, image_slice in enumerate(image):
        image_slices.append(image_slice)

    return image_slices, ct_image


def predict_valid():
    inputdir = "/data/Thoracic_OAR/"

    transform = valid_aug(image_size=512)

    # nii_files = glob.glob(inputdir + "/*/data.nii.gz")

    folds = [0, 1, 2, 3, 4]

    for fold in folds:
        print(fold)
        outdir = f"/data/Thoracic_OAR_predict/FPN-seresnext50/"
        log_dir = f"/logs/ss_miccai/FPN-se_resnext50_32x4d-fold-{fold}"
        # model = VNet(
        #     encoder_name='se_resnext50_32x4d',
        #     encoder_weights=None,
        #     classes=7,
        #     # activation='sigmoid',
        #     group_norm=False,
        #     center='none',
        #     attention_type='scse',
        #     reslink=True,
        #     multi_task=False
        # )

        model = FPN(
            encoder_name='se_resnext50_32x4d',
            encoder_weights=None,
            classes=7
        )

        ckp = os.path.join(log_dir, "checkpoints/best.pth")
        checkpoint = torch.load(ckp)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = nn.DataParallel(model)
        model = model.to(device)

        df = pd.read_csv(f'./csv/5folds/valid_{fold}.csv')
        patient_ids = df.patient_id.unique()
        for patient_id in patient_ids:
            print(patient_id)
            nii_file = f"{inputdir}/{patient_id}/data.nii.gz"

            image_slices, ct_image = extract_slice(nii_file)
            dataset = TestDataset(image_slices, transform)
            dataloader = DataLoader(
                dataset=dataset,
                num_workers=4,
                batch_size=8,
                drop_last=False
            )

            pred_mask, pred_logits = predict(model, dataloader)
            # import pdb
            # pdb.set_trace()
            pred_mask = np.argmax(pred_mask, axis=1).astype(np.uint8)
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
            # np.save(f"{patient_dir}/predic_logits.npy", pred_logits)



if __name__ == '__main__':
    predict_valid()

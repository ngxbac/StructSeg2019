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
from models import ResidualUNet3D

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


def load_ct_images(path):
    image = SimpleITK.ReadImage(path)
    spacing = image.GetSpacing()[-1]
    image_arr = SimpleITK.GetArrayFromImage(image).astype(np.float32)
    return image_arr, image



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


def predict_valid():
    inputdir = "/data/Thoracic_OAR/"

    transform = valid_aug(image_size=512)

    # nii_files = glob.glob(inputdir + "/*/data.nii.gz")

    folds = [0]

    crop_size = (32, 256, 256)
    xstep = 1
    ystep = 256
    zstep = 256
    num_classes = 7

    for fold in folds:
        print(fold)
        outdir = f"/data/Thoracic_OAR_predict/Unet3D/"
        log_dir = f"/logs/ss_miccai/Unet3D-fold-{fold}"
        model = ResidualUNet3D(
            in_channels=1,
            out_channels=num_classes
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

            image, ct_image = load_ct_images(nii_file)

            image = (image - LOWER_BOUND) / (UPPER_BOUND - LOWER_BOUND)
            image[image > 1] = 1.
            image[image < 0] = 0.
            image = image.astype(np.float32)
            C, H, W = image.shape

            deep_slices = np.arange(0, C - crop_size[0] + xstep, xstep)
            height_slices = np.arange(0, H - crop_size[1] + ystep, ystep)
            width_slices = np.arange(0, W - crop_size[2] + zstep, zstep)

            whole_pred = np.zeros((num_classes, C, H, W))
            count_used = np.zeros((C, H, W)) + 1e-5

            # no update parameter gradients during testing
            with torch.no_grad():
                for i in tqdm(range(len(deep_slices))):
                    for j in range(len(height_slices)):
                        for k in range(len(width_slices)):
                            deep = deep_slices[i]
                            height = height_slices[j]
                            width = width_slices[k]
                            image_crop = image[deep: deep + crop_size[0],
                                         height: height + crop_size[1],
                                         width: width + crop_size[2]]
                            image_crop = np.expand_dims(image_crop, axis=0)
                            image_crop = np.expand_dims(image_crop, axis=0)
                            image_crop = torch.from_numpy(image_crop).to(device)
                            # import pdb
                            # pdb.set_trace()
                            outputs = model(image_crop)
                            outputs = F.softmax(outputs, dim=1)
                            # ----------------Average-------------------------------
                            whole_pred[:,
                                deep: deep + crop_size[0],
                                height: height + crop_size[1],
                                width: width + crop_size[2]
                            ] += outputs.data.cpu().numpy()[0]

                            count_used[deep: deep + crop_size[0],
                            height: height + crop_size[1],
                            width: width + crop_size[2]] += 1

            whole_pred = whole_pred / count_used
            pred_mask = np.argmax(whole_pred, axis=0).astype(np.uint8)

            # pred_mask, pred_logits = predict(model, dataloader)
            # # import pdb
            # # pdb.set_trace()
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

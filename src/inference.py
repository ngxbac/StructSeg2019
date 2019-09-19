import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as Ftorch
from torch.utils.data import DataLoader
import os
import glob
import click
from tqdm import *
import cv2

from models import *
from segmentation_models_pytorch.unet import Unet
from augmentation import *
from dataset import *
from utils import *


device = torch.device('cuda')


def predict(model, loader):
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for dct in tqdm(loader, total=len(loader)):
            images = dct['images'].to(device)
            pred = model(images)
            pred = Ftorch.sigmoid(pred)
            pred = pred.detach().cpu().numpy()
            preds.append(pred)
            mask = dct['targets'].numpy()
            gts.append(mask)

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    return preds, gts


data_csv = "../Lung_GTV_2d/data.csv"
log_dir = f"../logs/Unet-SEResnext50-0/"


def predict_valid():
    test_csv = './csv/valid_0.csv'

    model = Unet(
        encoder_name='se_resnext50_32x4d',
        classes=1,
        activation='sigmoid'
    )
    ckp = os.path.join(log_dir, "checkpoints/best.pth")
    checkpoint = torch.load(ckp)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = nn.DataParallel(model)
    model = model.to(device)

    print("*" * 50)
    print(f"checkpoint: {ckp}")
    # Dataset
    dataset = StructSegTrain2D(
        csv_file=test_csv,
        data_csv=data_csv,
        transform=valid_aug(image_size=512),
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )

    preds, gts = predict(model, loader)

    os.makedirs("./prediction/", exist_ok=True)
    np.save(f"./prediction/valid.npy", preds)
    np.save(f"./prediction/gts.npy", gts)


if __name__ == '__main__':
    # predict_test()
    predict_valid()

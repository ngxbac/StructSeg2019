import click
import pandas as pd
import os
import numpy as np
from dataset import slice_builder, slice_builder_2d
from sklearn.model_selection import KFold, GroupKFold


@click.group()
def cli():
    print("Extract slices")


@cli.command()
@click.option('--csv_file', type=str)
@click.option('--root', type=str)
@click.option('--save_dir', type=str)
@click.option('--slice_thichness', type=int)
@click.option('--scale_ratio', type=float)
@click.option('--slice', type=int)
@click.option('--patch', type=int)
def extract(
    csv_file,
    root,
    save_dir,
    slice_thichness,
    scale_ratio,
    slice=16,
    patch=32
):
    df = pd.read_csv(csv_file)
    all_patient_df = []
    for imgpath, mskpath in zip(df.path, df.pathmsk):
        imgpath = os.path.join(root, imgpath)
        mskpath = os.path.join(root, mskpath)
        patient_df = slice_builder(imgpath, mskpath, slice_thichness, scale_ratio, slice, patch, save_dir)
        all_patient_df.append(patient_df)
    all_patient_df = pd.concat(all_patient_df, axis=0).reset_index(drop=True)
    all_patient_df.to_csv(os.path.join(save_dir, 'data.csv'))


@cli.command()
@click.option('--root', type=str)
@click.option('--save_dir', type=str)
def extract_2d(
    root,
    save_dir,
):
    # df = pd.read_csv(csv_file)
    all_patient_df = []
    import glob
    paths = glob.glob(root + "/*/*data*")
    masks = glob.glob(root + "/*/*label*")

    for imgpath, mskpath in zip(paths, masks):
        patient_df = slice_builder_2d(imgpath, mskpath, save_dir)
        all_patient_df.append(patient_df)
    all_patient_df = pd.concat(all_patient_df, axis=0).reset_index(drop=True)
    all_patient_df.to_csv(os.path.join(save_dir, 'data.csv'))


@cli.command()
@click.option('--csv_file', type=str)
@click.option('--n_folds', type=int)
@click.option('--save_dir', type=str)
def split_kfold(
    csv_file,
    n_folds,
    save_dir,
):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    patient_ids = df['patient_id'].values
    kf = GroupKFold(n_splits=n_folds)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(df, groups=patient_ids)):
        # train_patient = patient_ids[train_idx]
        # valid_patient = patient_ids[valid_idx]
        # train_df = df[df['patient_id'].isin(train_patient)].reset_index(drop=True)
        # valid_df = df[df['patient_id'].isin(valid_patient)].reset_index(drop=True)
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)

        train_df.to_csv(os.path.join(save_dir, f'train_{fold}.csv'), index=False)
        valid_df.to_csv(os.path.join(save_dir, f'valid_{fold}.csv'), index=False)


@cli.command()
@click.option('--csv_file', type=str)
@click.option('--n_folds', type=int)
@click.option('--save_dir', type=str)
def split_kfold_semi(
    csv_file,
    n_folds,
    save_dir,
):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    all_patients = df['patient_id'].unique()
    unlabeled_patients = np.random.choice(all_patients, size=10, replace=False)
    unlabeled_df = df[df['patient_id'].isin(unlabeled_patients)]
    labeled_df = df[~df['patient_id'].isin(unlabeled_patients)].reset_index(drop=True)
    patient_ids = labeled_df['patient_id'].values
    kf = GroupKFold(n_splits=n_folds)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(labeled_df, groups=patient_ids)):
        train_df = labeled_df.iloc[train_idx].reset_index(drop=True)
        valid_df = labeled_df.iloc[valid_idx].reset_index(drop=True)

        train_df.to_csv(os.path.join(save_dir, f'train_{fold}.csv'), index=False)
        valid_df.to_csv(os.path.join(save_dir, f'valid_{fold}.csv'), index=False)

    unlabeled_df.to_csv(os.path.join(save_dir, 'unlabeled_patients.csv'), index=False)


if __name__ == '__main__':
    cli()

import click
import pandas as pd
import os
import numpy as np
from dataset import slice_builder


@click.group()
def cli():
    print("Extract slices")


@cli.command()
@click.option('--csv_file', type=str)
@click.option('--root', type=str)
@click.option('--save_dir', type=str)
@click.option('--slice', type=int)
@click.option('--patch', type=int)
def extract(
    csv_file,
    root,
    save_dir=None,
    slice=16,
    patch=32
):
    df = pd.read_csv(csv_file)
    all_patient_df = []
    for imgpath, mskpath in zip(df.path, df.pathmsk):
        imgpath = os.path.join(root, imgpath)
        mskpath = os.path.join(root, mskpath)
        patient_df = slice_builder(imgpath, mskpath, slice, patch, save_dir)
        all_patient_df.append(patient_df)
    all_patient_df = pd.concat(all_patient_df, axis=0).reset_index(drop=True)
    all_patient_df.to_csv(os.path.join(save_dir, 'data.csv'))


if __name__ == '__main__':
    cli()

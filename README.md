# StructSeg2019
3rd of Task3 and 5th of Task4 of StructSeg2019 competition - MICCAI 2019

# Overview  

This repository is the solution of 3rd place of Task3 and 5th place of Task4 of [StructSeg2019](https://structseg2019.grand-challenge.org/) competition which is a part of MICCAI 2019. 


# Requirements 
- catalyst==19.9.1 
- albumentations==0.3.2 
- segmentation-models-pytorch==0.0.2 


# Note 
You may see my model named as `VNet`, you may be confused to this [paper](https://arxiv.org/abs/1606.04797). 
Actually, it is not, the model is still `UNet`. I named as `V` because of my personal purpose (The full name is `VUONGNet`). 

# How to run 

## Extract 2d slices 
Change the input and output path in [extract_slices.sh](bin/extract_slices.sh#L22). 

```bash
bash bin/extract_slices.sh
``` 

The output should contain numpy array of each slice and a csv file (data.csv)

## Split kfold 

```bash 
python src/preprocessing.py split-kfold --csv_file <path to your data.csv file> --n_folds 5 --out_dir <path to your output> 
```

## Train 

I train 2D images for 2 tasks.
All the settings are placed at: [config_2d.yml](configs/config_2d.yml)

```bash 
bash bin/train_2d.sh 
```

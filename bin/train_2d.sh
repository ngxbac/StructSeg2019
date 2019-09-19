#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3
RUN_CONFIG=config_2d.yml

for model in resnet34; do
    for fold in 0; do
        log_name=Vnet-$model-weighted-cedice-fold-${fold}
#        tag="["Unet","$model","$loss","fold-$fold"]"
        #stage 1
        LOGDIR=/logs/ss_task3/${log_name}/
        catalyst-dl run \
            --config=./configs/${RUN_CONFIG} \
            --logdir=$LOGDIR \
            --out_dir=$LOGDIR:str \
            --model_params/encoder_name=$model:str \
            --monitoring_params/name=${log_name}:str \
            --stages/data_params/train_csv=./csv/5folds/train_$fold.csv:str \
            --stages/data_params/valid_csv=./csv/5folds/valid_$fold.csv:str \
            --verbose
    done
done
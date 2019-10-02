#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3
RUN_CONFIG=config_task1.yml

for model in se_resnext50_32x4d; do
    for fold in 0; do
        log_name=Vnet-$model-weighted3-cedice19-cbam-fold-${fold}
#        tag="["Unet","$model","$loss","fold-$fold"]"
        #stage 1
        LOGDIR=/logs/ss_task1/${log_name}/
        catalyst-dl run \
            --config=./configs/${RUN_CONFIG} \
            --logdir=$LOGDIR \
            --out_dir=$LOGDIR:str \
            --model_params/encoder_name=$model:str \
            --monitoring_params/name=${log_name}:str \
            --stages/data_params/train_csv=./csv/task1_5folds/train_$fold.csv:str \
            --stages/data_params/valid_csv=./csv/task1_5folds/valid_$fold.csv:str \
            --verbose
    done
done
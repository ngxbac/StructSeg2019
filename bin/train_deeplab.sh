#!/usr/bin/env bash

 export CUDA_VISIBLE_DEVICES=2,3
RUN_CONFIG=config_deeplab.yml



for fold in 0; do
    log_name=Deeplab-${fold}
    #stage 1
    LOGDIR=../logs/${log_name}/
    catalyst-dl run \
        --config=./configs/${RUN_CONFIG} \
        --logdir=$LOGDIR \
        --out_dir=$LOGDIR:str \
        --monitoring_params/name=${log_name}:str \
        --stages/data_params/train_csv=./csv/train_$fold.csv:str \
        --stages/data_params/valid_csv=./csv/valid_$fold.csv:str \
        --verbose
done
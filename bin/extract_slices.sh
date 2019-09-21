#!/usr/bin/env bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8


slice_thichness=1
scale_ratio=0.125
slice=16
patch=32

#python src/preprocessing.py extract     --csv_file ./data/Lung_GTV/idx-train.csv \
#                                        --root ./data/Lung_GTV/ \
#                                        --save_dir ./data/Lung_GTV_st${slice_thichness}_sr${scale_ratio}_s${slice}_p${patch} \
#                                        --slice_thichness $slice_thichness \
#                                        --scale_ratio $scale_ratio \
#                                        --slice $slice \
#                                        --patch $patch



python src/preprocessing.py extract-2d      --root /data/HaN_OAR/ \
                                            --save_dir /data/HaN_OAR_2d/

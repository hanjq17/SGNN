#!/bin/bash
export HDF5_USE_FILE_LOCKING=FALSE
CUDA_VISIBLE_DEVICES=$2 python train_new.py  --env TDWdominoes  --model_name SGNN --log_per_iter 1000 --training_fpt 3 --ckp_per_iter 5000 --floor_cheat 1  --dataf "$1," --outf "$1_SGNN"

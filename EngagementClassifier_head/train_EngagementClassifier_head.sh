#!/bin/bash
CUDA_VISIBLE_DEVICES=4 python3 run_pplm_engagement_train.py --pretrained_model ../DialoGPT/model-medium  --save_model --epochs 5 --output_fp output_master --dataset_fp_train data/train_user_system_conv.csv --dataset_fp_valid data/valid_user_system_conv.csv --dataset_fp_test data/test_user_system_conv.csv  

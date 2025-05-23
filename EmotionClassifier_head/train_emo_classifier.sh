#!/bin/bash
#blenderbot
CUDA_VISIBLE_DEVICES=5 python3 train_pplm_emo_classifier.py --config_name strat --load_checkpoint ../GenerationModel/DATA/strat_pp.strat/2024-04-21230454.3e-05.16.1gpu/epoch-2.bin --save_model --epochs 10 --output_fp output_blenderbot --dataset_fp_train data/GO/train.csv --dataset_fp_valid data/GO/valid.csv --dataset_fp_test data/GO/test.csv 
#dialogpt
#CUDA_VISIBLE_DEVICES=4 python3 train_pplm_emo_classifier.py --config_name strat_dialogpt --load_checkpoint ../GenerationModel/DATA/strat_pp.strat_dialogpt/2024-04-22115423.5e-05.16.1gpu/epoch-3.bin --save_model --epochs 10 --output_fp output_dialogpt --dataset_fp_train data/GO/train.csv --dataset_fp_valid data/GO/valid.csv --dataset_fp_test data/GO/test.csv 

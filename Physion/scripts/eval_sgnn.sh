#!/bin/bash

ScenarioArray=("Dominoes" "Collide" "Support" "Link" "Contain" "Roll" "Drop" "Drape")
ScenarioArray=($4)
for val in ${ScenarioArray[*]}; do
	echo $val
    CUDA_VISIBLE_DEVICES=$5  python eval_new.py --subsample 1000 --env TDWdominoes --ransac_on_pred 1 --epoch $2 --iter $3 --model_name SGNN --training_fpt 3 --mode "test" --floor_cheat 1 --test_training_data_processing 1 --modelf "$1_SGNN"  --dataf "$val"
done


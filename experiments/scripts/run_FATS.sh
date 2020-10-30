#!/bin/bash

#example or use:
#bash run_FATS.sh save/FSNe/survey-FSNe_bands-gr_classes-3_kfid-0_kfid-0.slcd

filedir=$1
pkill -f calculate_FATS.py
ps ax | grep calculate_FATS.py
#python calculate_FATS.py -filedir $filedir
nohup python calculate_FATS.py -filedir $filedir > temp/fats_info.log &
#nohup python calculate_FATS.py -filedir $filedir > /dev/null &

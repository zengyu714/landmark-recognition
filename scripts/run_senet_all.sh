#!/bin/bash

SCRIPTS_ROOT="/home/kimmy/landmark-recognition/scripts"
cd $(dirname $SCRIPTS_ROOT)

nohup /opt/anaconda3/bin/python -u run.py \
--cuda-device 3 -p \
--use-stage 0 \
--model-name senet_all \
--nickname senet_all \
--modelname se_resnet50 \
--lr 1e-3 \
>> logs/log_finetune_senet_all.txt 2>&1 &
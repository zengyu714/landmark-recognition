#!/bin/bash

SCRIPTS_ROOT="/home/kimmy/landmark-recognition/scripts"
cd $(dirname $SCRIPTS_ROOT)

nohup /opt/anaconda3/bin/python -u run.py \
--cuda-device 0 -p \
--use-stage 0 \
--nickname resnet_all \
--modelname resnet50 \
--lr 1e-3 \
>> logs/log_finetune_all.txt 2>&1 &
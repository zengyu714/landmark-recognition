#!/bin/bash

SCRIPTS_ROOT="/home/kimmy/landmark-recognition/scripts"
cd $(dirname $SCRIPTS_ROOT)

nohup /opt/anaconda3/bin/python -u run.py \
--cuda-device 1 -p \
--use-stage 1 \
--nickname resnet_stage \
--modelname resnet50 \
--lr 1e-3 \
>> logs/log_finetune_stage.txt 2>&1 &
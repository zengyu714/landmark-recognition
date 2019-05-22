#!/bin/bash

SCRIPTS_ROOT="/home/kimmy/landmark-recognition/scripts"
cd $(dirname $SCRIPTS_ROOT)

nohup /opt/anaconda3/bin/python -u run.py \
--cuda-device 0 -p \
--use-stage 0 \
--nickname seatt_resnet50 \
--modelname seatt_resnet50 \
--lr 1e-3 \
--step-size 5 \
--tot-epochs 15 \
--input-size 96 \
--batch-size 128 \
>> logs/log_seatt_resnet50.txt 2>&1 &
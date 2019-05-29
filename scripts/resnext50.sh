#!/bin/bash

SCRIPTS_ROOT="/home/kimmy/landmark-recognition/scripts"
cd $(dirname $SCRIPTS_ROOT)

nohup /opt/anaconda3/bin/python -u run.py \
--cuda-device 2 -p \
--use-stage 0 \
--nickname resnext50 \
--modelname resnext50_32x4d \
--lr 1e-4 \
--step-size -1 \
--tot-epochs 40 \
--input-size 96 \
--batch-size 150 \
--optim-params '{"name": "adam", "weight_decay": 1e-4}' \
>> logs/resnext50.txt 2>&1 &
#!/bin/bash

SCRIPTS_ROOT="/home/kimmy/landmark-recognition/scripts"
cd $(dirname $SCRIPTS_ROOT)

nohup /opt/anaconda3/bin/python -u run.py \
--cuda-device 3 -p \
--use-stage 0 \
--nickname seatt_123 \
--modelname seatt_resnext50_32x4d \
--lr 5e-4 \
--step-size -1 \
--tot-epochs 30 \
--input-size 96 \
--batch-size 140 \
--optim-params '{"name": "adam", "weight_decay": 1e-4}' \
>> logs/seatt_123.txt 2>&1 &
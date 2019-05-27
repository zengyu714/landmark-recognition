#!/bin/bash

SCRIPTS_ROOT="/home/kimmy/landmark-recognition/scripts"
cd $(dirname $SCRIPTS_ROOT)

nohup /opt/anaconda3/bin/python -u run.py \
--cuda-device 2 -p \
--use-stage 0 \
--nickname resnext50_32x4d \
--modelname resnext50_32x4d \
--lr 1e-3 \
--step-size 5 \
--tot-epochs 20 \
--input-size 96 \
--batch-size 512 \
--optim-params '{"name": "adam", "weight_decay": 1e-4}' \
>> logs/resnext50_32x4d.txt 2>&1 &
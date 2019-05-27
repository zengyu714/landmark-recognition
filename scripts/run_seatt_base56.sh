#!/bin/bash

SCRIPTS_ROOT="/home/kimmy/landmark-recognition/scripts"
cd $(dirname $SCRIPTS_ROOT)

nohup /opt/anaconda3/bin/python -u run.py \
--cuda-device 2 \
--use-stage 0 \
--nickname seatt_base56 \
--modelname seatt_base56 \
--lr 1e-3 \
--step-size 5 \
--tot-epochs 20 \
--input-size 96 \
--batch-size 256 \
--optim-params '{"name": "adam", "weight_decay": 1e-4}' \
>> logs/seatt_base56.txt 2>&1 &
#!/bin/bash

SCRIPTS_ROOT="/home/kimmy/landmark-recognition/scripts"
cd $(dirname $SCRIPTS_ROOT)

nohup /opt/anaconda3/bin/python -u run.py \
--cuda-device 1 -p \
--use-stage 0 \
--nickname seatt_111 \
--modelname seatt_resnext50_base \
--lr 1e-5 \
--step-size -1 \
--tot-epochs 30 \
--input-size 96 \
--batch-size 150 \
--optim-params '{"name": "adam", "weight_decay": 1e-4}' \
>> logs/seatt_111.txt 2>&1 &
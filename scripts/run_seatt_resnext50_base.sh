#!/bin/bash

SCRIPTS_ROOT="/home/kimmy/landmark-recognition/scripts"
cd $(dirname $SCRIPTS_ROOT)

nohup /opt/anaconda3/bin/python -u run.py \
--cuda-device 1 -p \
--use-stage 0 \
--nickname seatt_resnext50_base \
--modelname seatt_resnext50_base \
--lr 1e-3 \
--step-size 5 \
--tot-epochs 20 \
--input-size 96 \
--batch-size 180 \
--optim-params '{"name": "adam", "weight_decay": 1e-4}' \
>> logs/seatt_resnext50_base.txt 2>&1 &
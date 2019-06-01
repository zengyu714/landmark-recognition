#!/bin/bash

SCRIPTS_ROOT="/home/kimmy/landmark-recognition/scripts"
cd $(dirname $SCRIPTS_ROOT)

nohup /opt/anaconda3/bin/python -u run.py \
--cuda-device 3 -p \
--use-stage 0 \
--nickname seatt_in224 \
--modelname seatt_resnext50_in224 \
--lr 1e-1 \
--step-size -1 \
--tot-epochs 40 \
--input-size 224 \
--batch-size 36 \
--optim-params '{"name": "sgd", "weight_decay": 1e-4, "momentum": 0.9}' \
>> logs/seatt_in224.txt 2>&1 &
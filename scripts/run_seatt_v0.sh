#!/bin/bash

SCRIPTS_ROOT="/home/kimmy/landmark-recognition/scripts"
cd $(dirname $SCRIPTS_ROOT)

nohup /opt/anaconda3/bin/python -u run.py \
--cuda-device 3 -p \
--use-stage 0 \
--nickname seatt_v0 \
--modelname seatt154 \
--lr 1e-3 \
--step-size 5 \
--tot-epochs 15 \
--input-size 96 \
--batch-size 74 \
>> logs/log_seatt_v0.txt 2>&1 &
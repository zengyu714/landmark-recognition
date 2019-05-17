#!/bin/bash

nohup /opt/anaconda3/bin/python -u run.py \
--cuda-device 3 -f \
--use-stage 0 \
--model-name senet_all \
--lr 1e-3 \
>> log_finetune_senet_all.txt 2>&1 &
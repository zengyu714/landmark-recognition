#!/bin/bash

nohup /opt/anaconda3/bin/python -u run.py \
--cuda-device 0 -f \
--use-stage 0 \
--model-name finetune_all \
--lr 1e-3 \
>> log_finetune_all.txt 2>&1 &
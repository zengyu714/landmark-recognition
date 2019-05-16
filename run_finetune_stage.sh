#!/bin/bash

nohup /opt/anaconda3/bin/python -u run.py \
--cuda-device 1 -f \
--use-stage 1 \
--model-name finetune_stage \
--lr 1e-3 \
>> log_finetune_stage.txt 2>&1 &
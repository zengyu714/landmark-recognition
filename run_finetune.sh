#!/bin/bash

nohup /opt/anaconda3/bin/python -u run.py \
--cuda-device 1 -f \
--tot-epoch 20 \
--finetune-name finetune_resnet50 \
--lr 1e-3 \
>> log_finetune_lr_1e-3.txt 2>&1 &
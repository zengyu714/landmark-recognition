#!/bin/bash

nohup /opt/anaconda3/bin/python -u run.py --cuda-device 1 -f  --tot-epoch 20 >> log_finetune_resnet50.txt 2>&1 &
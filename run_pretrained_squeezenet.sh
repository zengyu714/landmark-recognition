#!/bin/bash

nohup /opt/anaconda3/bin/python -u run.py \
--use-stage 0 \
--model-name squeezenet \
--lr 1e-3 \
>> log_squeezenet_all.txt 2>&1 &

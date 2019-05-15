#!/bin/bash

nohup /opt/anaconda3/bin/python -u run.py --cuda-device 0 >> log_bare_resnet50.txt 2>&1 &
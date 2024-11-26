#!/bin/bash

cd /data/wcx/yuka
cp /data/wcx/yuka/assets/Arial.ttf ~/.config/Ultralytics
source .venv/bin/activate
python -u train.py
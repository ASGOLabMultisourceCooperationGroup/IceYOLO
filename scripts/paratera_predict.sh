#!/bin/bash

# Fix font issue
cd ~/run/yuka
cp ~/run/yuka/assets/Arial.ttf ~/.config/Ultralytics

module unload cuda/10.0
module unload cuda/10.1
module unload cuda/10.1u1
module unload cuda/10.1u2
module unload cuda/10.2
module unload cuda/11.0
module unload cuda/11.0u1
module unload cuda/11.1
module unload cuda/11.2
module unload cuda/11.3
module unload cuda/11.4
module unload cuda/11.5
module unload cuda/11.6
module unload cuda/11.7
module unload cuda/11.8
module unload cuda/12.0
module unload cuda/12.1
module unload cuda/12.2
module unload cuda/9.0
module unload cuda/9.2


module load cuda/12.2

source .venv/bin/activate
python -u predict.py
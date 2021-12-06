#!/bin/sh
export FLASK_APP=./resnet_gpu/index.py
flask run -h 0.0.0.0 -p 50119

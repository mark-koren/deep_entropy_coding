#!/bin/bash

source /home/mark/miniconda2/bin/activate AST
cd ..
export PYTHONPATH=$(pwd):$(pwd)/garage:$PYTHONPATH

python deep_entropy_coding/garage_huffman.py

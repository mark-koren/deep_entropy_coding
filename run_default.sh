#!/bin/bash

source activate AST
#source /home/mkoren/miniconda2/bin/activate AST
cd ..
export PYTHONPATH=$(pwd):$(pwd)/AdaptiveStressTestingToolbox/Toolbox/garage:$PYTHONPATH

python deep_entropy_coding/garage_huffman.py

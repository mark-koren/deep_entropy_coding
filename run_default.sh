#!/bin/bash

source activate AST
cd ..
export PYTHONPATH=$(pwd):$(pwd)/AdaptiveStressTestingToolbox/Toolbox/garage:$PYTHONPATH

python deep_entropy_coding/garage_huffman.py

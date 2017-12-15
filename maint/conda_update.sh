#!/usr/bin/env bash
set -e
conda env remove -n ramutils -y
conda create -n ramutils python=3
source activate ramutils
conda install -y -c pennmem --file=requirements.txt
python setup.py install

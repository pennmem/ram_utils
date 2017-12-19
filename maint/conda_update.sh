#!/usr/bin/env bash
conda env remove -n ramutils -y
set -e
conda create -y -n ramutils python=3
source activate ramutils
conda install -y -c pennmem --file=requirements.txt
python setup.py install

#!/usr/bin/env bash
conda env remove -n ramutils -y
set -e
conda create -y -n ramutils python=2.7
source activate ramutils
conda install -y -c pennmem -c conda-forge --file=requirements.txt
python setup.py install

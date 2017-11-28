#!/usr/bin/env bash
set -e
conda env remove -n ramutils -y
conda env create --file test_env.yaml
source activate ramutils
pip install git+https://github.com/pennmem/classiflib.git
pip install git+https://github.com/pennmem/bptools.git
python setup.py install

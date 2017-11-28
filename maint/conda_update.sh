#!/usr/bin/env bash
set -e
conda env remove -n ramutils -y
conda create --file test_env.yaml -y
source activate ramutils
pip install git+https://github.com/pennmem/classiflib.git
pip install git+https://github.com/pennmem/bptools.git
pip install .

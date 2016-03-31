#!/bin/bash
export PATH="/home1/mswat/miniconda/bin:$PATH"
export LD_LIBRARY_PATH=~/libs/lib:$LD_LIBRARY_PATH

source activate rampy
cd /home1/mswat/RAM_UTILS_GIT/tests/fr1_report
python /home1/mswat/RAM_UTILS_GIT/tests/fr1_report/fr1_report_all.py  --task=RAM_CatFR1 --workspace-dir=/scratch/mswat/CatFR1_check_1 --mount-point="" --python-path=/home1/mswat/RAM_UTILS_GIT --python-path=/home1/mswat/PTSA_NEW_GIT

#--exit-on-no-change

#python fr1_report.py --subject=$1 --task=RAM_FR1 --workspace-dir=/scratch/pwanda/FR1_reports --mount-point='' --python-path=/home2/losu/ram_utils --python-path=/home2/losu/python/ptsa_latest

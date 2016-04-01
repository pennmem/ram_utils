#!/bin/bash
export PATH="/home1/mswat/miniconda/bin:$PATH"
export LD_LIBRARY_PATH=~/libs/lib:$LD_LIBRARY_PATH

source activate rampy

export PYTHONPATH="/home1/mswat/RAM_UTILS_GIT":$PYTHONPATH
export PYTHONPATH="/home1/mswat/PTSA_NEW_GIT":$PYTHONPATH

current_directory=$(pwd)

# PAL1
cd /home1/mswat/RAM_UTILS_GIT/tests/pal1_report
python /home1/mswat/RAM_UTILS_GIT/tests/pal1_report/pal1_report_all.py  --task=RAM_PAL1 --workspace-dir=/scratch/mswat/automated_reports/PAL1_reports

#--exit-on-no-change
cd ${current_directory}

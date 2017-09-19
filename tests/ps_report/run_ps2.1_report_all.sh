#!/bin/bash
export PATH="/home1/mswat/miniconda/bin":$PATH
export LD_LIBRARY_PATH=~/libs/lib:$LD_LIBRARY_PATH

source activate rampy

export PYTHONPATH="/home1/mswat/RAM_UTILS_GIT":$PYTHONPATH
export PYTHONPATH="/home1/mswat/PTSA_NEW_GIT":$PYTHONPATH
export PYTHONPATH="/home1/mswat/extra_libs":$PYTHONPATH

current_directory=$(pwd)



# PS2
cd /home1/mswat/RAM_UTILS_GIT/tests/ps_report
python /home1/mswat/RAM_UTILS_GIT/tests/ps_report/ps_report_all.py  --experiment=PS2.1 --workspace-dir=/scratch/mswat/automated_reports/PS2.1_reports



cd ${current_directory}


#--exit-on-no-change




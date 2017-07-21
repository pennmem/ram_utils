
#!/bin/bash
export PATH="/home1/mswat/miniconda/bin":$PATH
export LD_LIBRARY_PATH=~/libs/lib:$LD_LIBRARY_PATH

source activate rampy

export PYTHONPATH="/home1/mswat/RAM_UTILS_GIT":$PYTHONPATH
export PYTHONPATH="/home1/mswat/PTSA_NEW_GIT":$PYTHONPATH
export PYTHONPATH="/home1/mswat/extra_libs":$PYTHONPATH

#--python-path=/home1/mswat/extra_libs
current_directory=$(pwd)

# THR
cd /home1/mswat/RAM_UTILS_GIT/tests/thr_report
python /home1/mswat/RAM_UTILS_GIT/tests/thr_report/thr_report_all.py  --task=THR1 --workspace-dir=/scratch/mswat/automated_reports/THR_reports




cd ${current_directory}

#--exit-on-no-change

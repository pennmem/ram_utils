
#!/bin/bash
export PATH="/home1/mswat/miniconda/bin":$PATH
export LD_LIBRARY_PATH=~/libs/lib:$LD_LIBRARY_PATH

source activate rampy

export PYTHONPATH="/home1/mswat/RAM_UTILS_GIT":$PYTHONPATH
export PYTHONPATH="/home1/mswat/PTSA_NEW_GIT":$PYTHONPATH
export PYTHONPATH="/home1/mswat/extra_libs":$PYTHONPATH

#--python-path=/home1/mswat/extra_libs
current_directory=$(pwd)

# FR1
cd /home1/mswat/RAM_UTILS_GIT/tests/fr1_reportz
python /home1/mswat/RAM_UTILS_GIT/tests/fr1_report/fr1_report_all.py  --task=RAM_FR1 --workspace-dir=/scratch/mswat/automated_reports/FR1_reports




cd ${current_directory}

#--exit-on-no-change



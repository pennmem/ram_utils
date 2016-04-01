
#!/bin/bash
export PATH="/home1/mswat/miniconda/bin":$PATH
export LD_LIBRARY_PATH=~/libs/lib:$LD_LIBRARY_PATH

source activate rampy

export PYTHONPATH="/home1/mswat/RAM_UTILS_GIT":$PYTHONPATH
export PYTHONPATH="/home1/mswat/PTSA_NEW_GIT":$PYTHONPATH

current_directory=$(pwd)


# FR1 CatFR1 joint
cd /home1/mswat/RAM_UTILS_GIT/tests/fr1_catfr1_joint_report
python /home1/mswat/RAM_UTILS_GIT/tests/fr1_catfr1_joint_report/fr1_catfr1_joint_report_all.py   --workspace-dir=/scratch/mswat/automated_reports/FR1_CatFr1_reports


cd ${current_directory}

#--exit-on-no-change


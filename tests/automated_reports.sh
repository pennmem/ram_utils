
#!/bin/bash
export PATH="/home1/mswat/miniconda/bin":$PATH
export LD_LIBRARY_PATH=~/libs/lib:$LD_LIBRARY_PATH

source activate rampy

export PYTHONPATH="/home1/mswat/RAM_UTILS_GIT":$PYTHONPATH
export PYTHONPATH="/home1/mswat/PTSA_NEW_GIT":$PYTHONPATH

current_directory=$(pwd)

# FR1
cd /home1/mswat/RAM_UTILS_GIT/tests/fr1_report
python /home1/mswat/RAM_UTILS_GIT/tests/fr1_report/fr1_report_all.py  --task=RAM_FR1 --workspace-dir=/scratch/mswat/automated_reports/FR1_reports

#CatFR1
cd /home1/mswat/RAM_UTILS_GIT/tests/fr1_report
python /home1/mswat/RAM_UTILS_GIT/tests/fr1_report/fr1_report_all.py  --task=RAM_CatFR1 --workspace-dir=/scratch/mswat/automated_reports/CatFR1_reports

# FR1 CatFR1 joint
cd /home1/mswat/RAM_UTILS_GIT/tests/fr1_catfr1_joint_report
python /home1/mswat/RAM_UTILS_GIT/tests/fr1_catfr1_joint_report/fr1_catfr1_joint_report_all.py   --workspace-dir=/scratch/mswat/automated_reports/FR1_CatFr1_reports

# PAL1
cd /home1/mswat/RAM_UTILS_GIT/tests/pal1_report
python /home1/mswat/RAM_UTILS_GIT/tests/pal1_report/pal1_report_all.py  --task=RAM_PAL1 --workspace-dir=/scratch/mswat/automated_reports/PAL1_reports


# PS2
cd /home1/mswat/RAM_UTILS_GIT/tests/ps_report
python /home1/mswat/RAM_UTILS_GIT/tests/ps_report/ps_report_all.py  --experiment=PS2 --workspace-dir=/scratch/mswat/automated_reports/PS2_reports


# PS1
cd /home1/mswat/RAM_UTILS_GIT/tests/ps_report
python /home1/mswat/RAM_UTILS_GIT/tests/ps_report/ps_report_all.py  --experiment=PS1 --workspace-dir=/scratch/mswat/automated_reports/PS1_reports


# PS3
cd /home1/mswat/RAM_UTILS_GIT/tests/ps_report
python /home1/mswat/RAM_UTILS_GIT/tests/ps_report/ps_report_all.py  --experiment=PS3 --workspace-dir=/scratch/mswat/automated_reports/PS3_reports



cd ${current_directory}

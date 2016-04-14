
#!/bin/bash
export PATH="/home1/mswat/miniconda/bin":$PATH
export LD_LIBRARY_PATH=~/libs/lib:$LD_LIBRARY_PATH

source activate rampy

export PYTHONPATH="/home1/mswat/RAM_UTILS_GIT":$PYTHONPATH
export PYTHONPATH="/home1/mswat/PTSA_NEW_GIT":$PYTHONPATH

export PYTHONPATH="/home1/mswat/extra_libs":$PYTHONPATH

current_directory=$(pwd)

datetime=$(date '+%Y_%m_%d_%H_%M_%S')

echo ${datetime}

status_output_dirs=()


# FR1
cd /home1/mswat/RAM_UTILS_GIT/tests/fr1_report

workspace_dir=/scratch/mswat/automated_reports/FR1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})


python /home1/mswat/RAM_UTILS_GIT/tests/fr1_report/fr1_report_all.py  --task=RAM_FR1 \
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir}

#CatFR1
cd /home1/mswat/RAM_UTILS_GIT/tests/fr1_report

workspace_dir=/scratch/mswat/automated_reports/CatFR1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

python /home1/mswat/RAM_UTILS_GIT/tests/fr1_report/fr1_report_all.py  --task=RAM_CatFR1 \
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir}

# FR1 CatFR1 joint
cd /home1/mswat/RAM_UTILS_GIT/tests/fr1_catfr1_joint_report

workspace_dir=/scratch/mswat/automated_reports/FR1_CatFr1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

python /home1/mswat/RAM_UTILS_GIT/tests/fr1_catfr1_joint_report/fr1_catfr1_joint_report_all.py --task=RAM_FR1_CatFR1_joint\
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir}

# PAL1
cd /home1/mswat/RAM_UTILS_GIT/tests/pal1_report

workspace_dir=/scratch/mswat/automated_reports/PAL1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

python /home1/mswat/RAM_UTILS_GIT/tests/pal1_report/pal1_report_all.py  --task=RAM_PAL1 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir}

# PS1
cd /home1/mswat/RAM_UTILS_GIT/tests/ps_report

workspace_dir=/scratch/mswat/automated_reports/PS1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

python /home1/mswat/RAM_UTILS_GIT/tests/ps_report/ps_report_all.py  --experiment=PS1 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir}


# PS2
cd /home1/mswat/RAM_UTILS_GIT/tests/ps_report

workspace_dir=/scratch/mswat/automated_reports/PS2_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

python /home1/mswat/RAM_UTILS_GIT/tests/ps_report/ps_report_all.py  --experiment=PS2 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir}

# PS2.1
cd /home1/mswat/RAM_UTILS_GIT/tests/ps_report

workspace_dir=/scratch/mswat/automated_reports/PS2.1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

python /home1/mswat/RAM_UTILS_GIT/tests/ps_report/ps_report_all.py  --experiment=PS2.1 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir}

# PS3
cd /home1/mswat/RAM_UTILS_GIT/tests/ps_report

workspace_dir=/scratch/mswat/automated_reports/PS3_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

python /home1/mswat/RAM_UTILS_GIT/tests/ps_report/ps_report_all.py  --experiment=PS3 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir}


#PS1-2 aggregator
cd /home1/mswat/RAM_UTILS_GIT/tests/ps_aggregator
python /home1/mswat/RAM_UTILS_GIT/tests/ps_aggregator/ps_aggregator.py --workspace-dir=/scratch/mswat/automated_reports

#PS3 aggregator
cd /home1/mswat/RAM_UTILS_GIT/tests/ps3_aggregator
python /home1/mswat/RAM_UTILS_GIT/tests/ps3_aggregator/ps3_aggregator.py --workspace-dir=/scratch/mswat/automated_reports

#ttest significance generator
cd /home1/mswat/RAM_UTILS_GIT/tests/ps_ttest_table
python /home1/mswat/RAM_UTILS_GIT/tests/ps_ttest_table/ttest_table_with_params.py --workspace-dir=/scratch/mswat/automated_reports

python /Users/m/RAM_UTILS_GIT/ReportUtils/ReportMailer.py\
 --status-output-dirs ${status_output_dirs[@]}


cd ${current_directory}

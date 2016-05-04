#!/bin/bash

source /home1/mswat/.bashrc
module load Tex/2014

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

echo ${status_output_dir}

automated_reports_dir=/scratch/mswat/automated_reports

exit_on_no_change_flag=--exit-on-no-change
exit_on_no_change_flag=

echo ${exit_on_no_change}




## PAL1
cd /home1/mswat/RAM_UTILS_GIT/tests/pal1_report

workspace_dir=${automated_reports_dir}/PAL1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})


python /home1/mswat/RAM_UTILS_GIT/tests/pal1_report/pal1_report_all.py  --task=RAM_PAL1 --workspace-dir=/scratch/mswat/automated_reports/PAL1_reports ${exit_on_no_change_flag}


python /home1/mswat/RAM_UTILS_GIT/ReportUtils/ReportMailer.py\
 --status-output-dirs ${status_output_dirs[@]} --error-log-file=${automated_reports_dir}/error_logs/${datetime}.error.txt

cd ${current_directory}

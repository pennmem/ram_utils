
#!/bin/bash

source /home1/mswat/.bashrc
module load Tex/2014


export PATH="/home1/mswat/miniconda/bin":$PATH
export LD_LIBRARY_PATH=~/libs/lib:$LD_LIBRARY_PATH

source activate rampy

export PYTHONPATH="/home1/mswat/RAM_UTILS_GIT":$PYTHONPATH
export PYTHONPATH="/home1/mswat/PTSA_NEW_GIT":$PYTHONPATH

export PYTHONPATH="/home1/mswat/extra_libs":$PYTHONPATH

#source /home1/mswat/.bashrc
#module load Tex/2014

datetime=$(date '+%Y_%m_%d_%H_%M_%S')
echo ${datetime}

# variable definitions
status_output_dirs=()

automated_reports_dir=/scratch/mswat/automated_reports

exit_on_no_change_flag=--exit-on-no-change
#exit_on_no_change_flag=




current_directory=$(pwd)

# FR1
cd /home1/mswat/RAM_UTILS_GIT/tests/fr1_report

workspace_dir=${automated_reports_dir}/FR1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})


python /home1/mswat/RAM_UTILS_GIT/tests/fr1_report/fr1_report_all.py  --task=RAM_FR1\
 --workspace-dir=/scratch/mswat/automated_reports/FR1_reports --recompute-on-no-status --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}

python /home1/mswat/RAM_UTILS_GIT/ReportUtils/ReportMailer.py\
  --status-output-dirs ${status_output_dirs[@]} --error-log-file=${automated_reports_dir}/error_logs/${datetime}.error.txt

cd ${current_directory}

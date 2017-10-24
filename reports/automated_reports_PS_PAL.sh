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



function remove_old_status_dirs {
    # this function removes all status output dirs that begin with yesterdays date
    # date -v -1d '+%Y_%m_%d'|  xargs bash -c   'ls -d $0*' | xargs rm -rf
#    date -v -1d '+%Y_%m_%d' |  xargs bash -c   'find . -name "$0*" -mindepth 1 -maxdepth 1 -type d' | xargs rm -rf

    find $1 -name "$(date --date="yesterday" +%Y_%m_%d)*" -type d | xargs rm -rf

}


function remove_old_error_logs {
    # this function removes old error logs whose name begin with yesterday's date
#    date -v -1d '+%Y_%m_%d'|  xargs bash -c   'ls $0*' | xargs rm -rf
#    date --date="yesterday" +%Y_%m_%d|  xargs bash -c   'ls $0*' | xargs rm -rf
    find . -name "$(date --date="yesterday" +%Y_%m_%d)*" -type f | xargs rm -rf
}



current_directory=$(pwd)

datetime=$(date '+%Y_%m_%d_%H_%M_%S')

echo ${datetime}

status_output_dirs=()

automated_reports_dir=/scratch/mswat/automated_reports

exit_on_no_change_flag=--exit-on-no-change
exit_on_no_change_flag=





# PAL_PS1
report_code_dir=/home1/mswat/RAM_UTILS_GIT/tests/ps_pal_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/PS1_PAL_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/ps_pal_report_all.py  --experiment=PS1 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}


# PAL_PS2
report_code_dir=/home1/mswat/RAM_UTILS_GIT/tests/ps_pal_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/PS2_PAL_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/ps_pal_report_all.py  --experiment=PS2 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}


# PAL_PS2.1
report_code_dir=/home1/mswat/RAM_UTILS_GIT/tests/ps_pal_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/PS2.1_PAL_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/ps_pal_report_all.py  --experiment=PS2.1 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}


# PAL_PS3
report_code_dir=/home1/mswat/RAM_UTILS_GIT/tests/ps_pal_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/PS3_PAL_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/ps_pal_report_all.py  --experiment=PS3 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}



python /home1/mswat/RAM_UTILS_GIT/ReportUtils/ReportMailer.py\
 --status-output-dirs ${status_output_dirs[@]} --error-log-file=${automated_reports_dir}/error_logs/${datetime}.error.txt


#removing old error logs
cd ${automated_reports_dir}/error_logs
remove_old_error_logs

#
cd ${current_directory}

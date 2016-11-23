#!/bin/bash

#$ -q RAM.q
#$ -S /bin/bash
#$ -cwd
#$ -N REPORT_CONNECTIVITY
#$ -j y
#$ -o /scratch/RAM_maint/automated_connectivity_reports/automated_connectivity_reports.log

#$ -pe python-shared 1
#$ -l h_rt=72:00:00
#$ -l h_vmem=64G

source /home2/RAM_maint/.cron_init_script
module load Tex/2014

#export PATH="/home2/RAM_maint/miniconda/bin":$PATH
#export LD_LIBRARY_PATH=~/libs/lib:$LD_LIBRARY_PATH

export PYTHONPATH=/home2/RAM_maint/RAM_UTILS_GIT:$PYTHONPATH
#export PYTHONPATH="/home2/RAM_maint/PTSA_NEW_GIT":$PYTHONPATH

#export PYTHONPATH="/home2/RAM_maint/extra_libs":$PYTHONPATH

which python

#source /home2/RAM_maint/.bashrc
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

automated_reports_dir=/scratch/RAM_maint/automated_connectivity_reports

if [ ! -d "$automated_reports_dir" ]; then
  mkdir -p ${automated_reports_dir}
fi

exit_on_no_change_flag=--exit-on-no-change
#exit_on_no_change_flag=

LOCKFILE=${automated_reports_dir}/automated_connectivity_reports.lock
# making sure only one copy of automated reports runs
if [ -f ${LOCKFILE} ]
then
    echo 'Lock file ${automated_reports_dir}/automated_reports.lock is present indicating another script is running'
    exit 1
else
    touch ${LOCKFILE}
fi
#lockfile -r 0 ${automated_reports_dir}/automated_reports.lock || exit 1



# FR1/catFR1
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/fr_connectivity_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/FR1_connectivity_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/fr_connectivity_report_all.py   \
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\

#removing old error logs
cd ${automated_reports_dir}/error_logs
remove_old_error_logs

cd ${current_directory}

# removing lockfile
rm -f ${LOCKFILE}

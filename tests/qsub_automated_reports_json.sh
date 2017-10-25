#!/bin/bash

source /home2/RAM_maint/.bashrc

automated_reports_dir=/scratch/RAM_maint/automated_reports_json


LOCKFILE=${automated_reports_dir}/automated_reports.lock
# making sure only one copy of automated reports runs
if [ -f ${LOCKFILE} ]
then
    echo 'Lock file ${automated_reports_dir}/automated_reports.lock is present indicating another script is running'
    exit 1

fi


qsub /home2/RAM_maint/RAM_UTILS_GIT/tests/automated_reports_ram_maint_json.sh

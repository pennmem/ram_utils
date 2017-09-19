#!/bin/bash

#$ -q RAM.q
#$ -S /bin/bash
#$ -cwd
#$ -N REPORT_JSON
#$ -j y
#$ -o /scratch/RAM_maint/automated_reports_json/automated_reports.log

#$ -pe python-shared 10
#$ -l h_rt=72:00:00
#$ -l h_vmem=12G

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

report_code_base=/home2/RAM_maint/RAM_UTILS_GIT

current_directory=$(pwd)

datetime=$(date '+%Y_%m_%d_%H_%M_%S')

echo ${datetime}

status_output_dirs=()

automated_reports_dir=/scratch/RAM_maint/automated_reports_json

if [ ! -d "$automated_reports_dir" ]; then
  mkdir -p ${automated_reports_dir}
fi

exit_on_no_change_flag=--exit-on-no-change
#exit_on_no_change_flag=

LOCKFILE=${automated_reports_dir}/automated_reports.lock
# making sure only one copy of automated reports runs
if [ -f ${LOCKFILE} ]
then
    echo 'Lock file ${automated_reports_dir}/automated_reports.lock is present indicating another script is running'
    exit 1
else
    touch ${LOCKFILE}
fi
#lockfile -r 0 ${automated_reports_dir}/automated_reports.lock || exit 1


function reports {
   # syntax: reports report_code_dir workspace_dir report_code_script ${@}
   # report_code_dir is relative to RAM_UTILS_GIT/tests
   report_code_dir=${report_code_base}/$1
   cd ${report_code_dir}

   workspace_dir=$2

   status_output_dir=$2/${datetime}
   status_output_dirs+=(${status_output_dir})

   remove_old_status_dirs ${workspace_dir}
   python ${report_code_dir}/$3.py --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir}\
   ${exit_on_no_change_flag} --recompute-on-no-status "${@:4}"
}



## THR1
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/thr1_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/THR1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/thr1_report_all.py  --task=THR1\
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\

python ${report_code_dir}/thr1_report_all.py  --task=THR\
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\

#PS4

report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/ps4_report
cd ${report_code_dir}

## PS4_FR5
task=FR5

workspace_dir=${automated_reports_dir}/PS4_${task}_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/ps4_report_all.py  --task=${task} \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}

## PS4_PAL5
task=PAL5

workspace_dir=${automated_reports_dir}/PS4_${task}_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/ps4_report_all.py  --task=${task} \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}

## PS4_catFR5
task = catFR5
workspace_dir=${automated_reports_dir}/PS4_${task}_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/ps4_report_all.py  --task=${task} \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}



# FR5
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/fr5_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/FR5_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/fr5_report_all.py  --task=FR5 \
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}


# FR3
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/fr_stim_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/FR3_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/fr_stim_report_all.py  --task=FR3 \
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\
 --skip-subjects R1093J_1 R1235E

 # catFR3
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/fr_stim_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/catFR3_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}


python ${report_code_dir}/fr_stim_report_all.py  --task=catFR3 \
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\
 --skip-subjects R1093J_1 R1243T

#joint FR3 catFr3
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/fr_catfr_joint_stim_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/joint_fr3_catFR3_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/fr_catfr_joint_stim_report_all.py  --task=FR3 \
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\
 --skip-subjects R1093J_1




# PAL3
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/pal_stim_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/PAL3_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/pal_stim_report_all.py  --task=PAL3 \
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}


# PAL5
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/pal5_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/PAL5_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/pal5_report_all.py  --task=PAL5 --classifier=pal \
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}

#TH3
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/th_stim_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/TH3_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/th_stim_report_all.py  --task=TH3 \
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}


# FR4
#report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/fr_stim_report
#cd ${report_code_dir}
#
#workspace_dir=${automated_reports_dir}/FR4_reports
#status_output_dir=${workspace_dir}/${datetime}
#status_output_dirs+=(${status_output_dir})
#
#remove_old_status_dirs ${workspace_dir}
#
#python ${report_code_dir}/fr_stim_report_all.py  --task=FR4 \
# --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}
#


# FR1
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/fr1_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/FR1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/fr1_report_all.py  --task=FR1 \
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\
 --skip-subjects R1055J R1061T R1090C R1092J_2 R1093J_1 R2009P_1 R2012P_1 R2015P_1

#CatFR1
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/fr1_report
cd ${report_code_dir}


workspace_dir=${automated_reports_dir}/CatFR1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/fr1_report_all.py  --task=catFR1 \
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\
 --skip-subjects R1029W


# FR1 CatFR1 joint
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/fr1_catfr1_joint_report
cd ${report_code_dir}


workspace_dir=${automated_reports_dir}/FR1_CatFr1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/fr1_catfr1_joint_report_all.py --task=FR1_CatFR1_joint\
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\
 --skip-subjects R1061T


# PAL1
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/pal1_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/PAL1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/pal1_report_all.py  --task=PAL1 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\
  --skip-subjects R1050M R1136N


# PAL1 -- system 3

report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/pal1_sys3_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/PAL1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/pal1_report_all.py  --task=PAL1 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\
  --skip-subjects R1050M R1136N


# PS1
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/ps_report
cd ${report_code_dir}


workspace_dir=${automated_reports_dir}/PS1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/ps_report_all.py  --task=PS1 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\
  --skip-subjects R1061T


# PS2
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/ps_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/PS2_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

#python ${report_code_dir}/ps_report_all.py  --task=PS2 \
#  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\
#  --skip-subjects R1100D

# PS2.1
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/ps_report
cd ${report_code_dir}



workspace_dir=${automated_reports_dir}/PS2.1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

#python ${report_code_dir}/ps_report_all.py  --task=PS2.1 \
#  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}

# PS3
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/ps_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/PS3_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/ps_report_all.py  --task=PS3 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}



# PAL_PS1
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/ps_pal_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/PS1_PAL_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/ps_pal_report_all.py  --task=PS1 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}


# PAL_PS2
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/ps_pal_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/PS2_PAL_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/ps_pal_report_all.py  --task=PS2 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\
  --skip-subjects R1100D


# PAL_PS2.1
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/ps_pal_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/PS2.1_PAL_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/ps_pal_report_all.py  --task=PS2.1 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}


# PAL_PS3
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/ps_pal_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/PS3_PAL_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/ps_pal_report_all.py  --task=PS3 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}


# TH_PS1
#report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/ps_th_report
#cd ${report_code_dir}
#
#workspace_dir=${automated_reports_dir}/PS1_TH_reports
#status_output_dir=${workspace_dir}/${datetime}
#status_output_dirs+=(${status_output_dir})
#
#remove_old_status_dirs ${workspace_dir}
#
#python ${report_code_dir}/ps_pal_report_all.py  --task=PS1 \
#  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}

# TH_PS2
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/ps_th_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/PS2_TH_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/ps_th_report_all.py  --task=PS2 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}

# TH_PS2.1
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/ps_th_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/PS2.1_TH_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/ps_th_report_all.py  --task=PS2.1 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}



# TH_PS3
#report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/ps_th_report
#cd ${report_code_dir}
#
#workspace_dir=${automated_reports_dir}/PS3_TH_reports
#status_output_dir=${workspace_dir}/${datetime}
#status_output_dirs+=(${status_output_dir})
#
#remove_old_status_dirs ${workspace_dir}
#
#python ${report_code_dir}/ps_th_report_all.py  --task=PS3 \
#  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}


## TH1
report_code_dir=/home2/RAM_maint/RAM_UTILS_GIT/tests/th1_report
cd ${report_code_dir}

workspace_dir=${automated_reports_dir}/TH1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python ${report_code_dir}/th1_report_all.py  --task=TH1\
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\
 --skip-subjects R1132C R1201P_1




#PS1-2 aggregator
#cd /home2/RAM_maint/RAM_UTILS_GIT/tests/ps_aggregator
#python /home2/RAM_maint/RAM_UTILS_GIT/tests/ps_aggregator/ps_aggregator.py --workspace-dir=${automated_reports_dir}

#PS3 aggregator
#cd /home2/RAM_maint/RAM_UTILS_GIT/tests/ps3_aggregator
#python /home2/RAM_maint/RAM_UTILS_GIT/tests/ps3_aggregator/ps3_aggregator.py --workspace-dir=${automated_reports_dir}

#ttest significance generator
#cd /home2/RAM_maint/RAM_UTILS_GIT/tests/ps_ttest_table
#python /home2/RAM_maint/RAM_UTILS_GIT/tests/ps_ttest_table/ttest_table_with_params.py --workspace-dir=${automated_reports_dir}


python /home2/RAM_maint/RAM_UTILS_GIT/ReportUtils/ReportMailer.py\
 --status-output-dirs ${status_output_dirs[@]} --error-log-file=${automated_reports_dir}/error_logs/${datetime}.error.txt


#removing old error logs
cd ${automated_reports_dir}/error_logs
remove_old_error_logs


cd ${current_directory}

# removing lockfile
rm -f ${LOCKFILE}

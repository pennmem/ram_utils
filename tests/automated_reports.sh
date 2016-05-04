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
#exit_on_no_change_flag=



# FR1
cd /home1/mswat/RAM_UTILS_GIT/tests/fr1_report

workspace_dir=/scratch/mswat/automated_reports/FR1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python /home1/mswat/RAM_UTILS_GIT/tests/fr1_report/fr1_report_all.py  --task=RAM_FR1 \
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\
 --skip-subjects R1055J R1061T R1090C R1092J_2 R1093J_1 R2009P_1 R2012P_1 R2015P_1

# FR3
cd /home1/mswat/RAM_UTILS_GIT/tests/fr_stim_report

workspace_dir=${automated_reports_dir}/FR3_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python /home1/mswat/RAM_UTILS_GIT/tests/fr_stim_report/fr_stim_report_all.py  --task=RAM_FR3 \
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\
 --skip-subjects R1093J_1

# FR4
cd /home1/mswat/RAM_UTILS_GIT/tests/fr_stim_report

workspace_dir=${automated_reports_dir}/FR4_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python /home1/mswat/RAM_UTILS_GIT/tests/fr_stim_report/fr_stim_report_all.py  --task=RAM_FR4 \
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}


#CatFR1
cd /home1/mswat/RAM_UTILS_GIT/tests/fr1_report

workspace_dir=${automated_reports_dir}/CatFR1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python /home1/mswat/RAM_UTILS_GIT/tests/fr1_report/fr1_report_all.py  --task=RAM_CatFR1 \
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\
 --skip-subjects R1029W

# FR1 CatFR1 joint
cd /home1/mswat/RAM_UTILS_GIT/tests/fr1_catfr1_joint_report

workspace_dir=${automated_reports_dir}/FR1_CatFr1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python /home1/mswat/RAM_UTILS_GIT/tests/fr1_catfr1_joint_report/fr1_catfr1_joint_report_all.py --task=RAM_FR1_CatFR1_joint\
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\
 --skip-subjects R1061T


# PAL1
cd /home1/mswat/RAM_UTILS_GIT/tests/pal1_report

workspace_dir=${automated_reports_dir}/PAL1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python /home1/mswat/RAM_UTILS_GIT/tests/pal1_report/pal1_report_all.py  --task=RAM_PAL1 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\
  --skip-subjects R1050M R1136N

# PS1
cd /home1/mswat/RAM_UTILS_GIT/tests/ps_report

workspace_dir=${automated_reports_dir}/PS1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python /home1/mswat/RAM_UTILS_GIT/tests/ps_report/ps_report_all.py  --experiment=PS1 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\
  --skip-subjects R1061T


# PS2
cd /home1/mswat/RAM_UTILS_GIT/tests/ps_report

workspace_dir=${automated_reports_dir}/PS2_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python /home1/mswat/RAM_UTILS_GIT/tests/ps_report/ps_report_all.py  --experiment=PS2 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}\
  --skip-subjects R1100D

# PS2.1
cd /home1/mswat/RAM_UTILS_GIT/tests/ps_report

workspace_dir=${automated_reports_dir}/PS2.1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python /home1/mswat/RAM_UTILS_GIT/tests/ps_report/ps_report_all.py  --experiment=PS2.1 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}

# PS3
cd /home1/mswat/RAM_UTILS_GIT/tests/ps_report

workspace_dir=${automated_reports_dir}/PS3_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python /home1/mswat/RAM_UTILS_GIT/tests/ps_report/ps_report_all.py  --experiment=PS3 \
  --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}


# FR3
cd /home1/mswat/RAM_UTILS_GIT/tests/fr_stim_report

workspace_dir=${automated_reports_dir}/FR3_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python /home1/mswat/RAM_UTILS_GIT/tests/fr1_report/fr_stim_report_all.py  --task=RAM_FR3 \
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}


# FR4
cd /home1/mswat/RAM_UTILS_GIT/tests/fr_stim_report

workspace_dir=${automated_reports_dir}/FR4_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python /home1/mswat/RAM_UTILS_GIT/tests/fr1_report/fr_stim_report_all.py  --task=RAM_FR4 \
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}



## TH1
cd /home1/mswat/RAM_UTILS_GIT/tests/th1_report

workspace_dir=${automated_reports_dir}/TH1_reports
status_output_dir=${workspace_dir}/${datetime}
status_output_dirs+=(${status_output_dir})

remove_old_status_dirs ${workspace_dir}

python /home1/mswat/RAM_UTILS_GIT/tests/th1_report/th1_report_all.py  --task=RAM_TH1\
 --recompute-on-no-status --workspace-dir=${workspace_dir} --status-output-dir=${status_output_dir} ${exit_on_no_change_flag}



##PS1-2 aggregator
#cd /home1/mswat/RAM_UTILS_GIT/tests/ps_aggregator
#python /home1/mswat/RAM_UTILS_GIT/tests/ps_aggregator/ps_aggregator.py --workspace-dir=${automated_reports_dir}
#
##PS3 aggregator
#cd /home1/mswat/RAM_UTILS_GIT/tests/ps3_aggregator
#python /home1/mswat/RAM_UTILS_GIT/tests/ps3_aggregator/ps3_aggregator.py --workspace-dir=${automated_reports_dir}
#
##ttest significance generator
#cd /home1/mswat/RAM_UTILS_GIT/tests/ps_ttest_table
#python /home1/mswat/RAM_UTILS_GIT/tests/ps_ttest_table/ttest_table_with_params.py --workspace-dir=${automated_reports_dir}



python /home1/mswat/RAM_UTILS_GIT/ReportUtils/ReportMailer.py\
 --status-output-dirs ${status_output_dirs[@]} --error-log-file=${automated_reports_dir}/error_logs/${datetime}.error.txt


#removing old error logs
cd ${automated_reports_dir}/error_logs
remove_old_error_logs

#
cd ${current_directory}

#!/bin/bash

#$ -q RAM.q
#$ -S /bin/bash
#$ -cwd
#$ -N CONNECTIVITY_REPORT
#$ -j y
#$ -o /scratch/pwanda/connectivity_reports.log

#$ -pe python-shared 1
#$ -l h_rt=72:00:00
#$ -l h_vmem=64G

source $HOME/.cron_init_script

cd $HOME/ram_utils/tests/fr_connectivity_report
python fr_connectivity_report.py --subject=${subject} --workspace-dir=/scratch/pwanda/FR1_CatFR1_connectivity_reports --mount-point=''

cd -

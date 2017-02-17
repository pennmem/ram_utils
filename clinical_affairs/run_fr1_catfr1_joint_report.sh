#!/usr/bin/env bash
cd ../tests/fr1_catfr1_joint_report
python fr1_catfr1_joint_report.py --subject=$1 --workspace-dir=/scratch/pwanda/FR_reports --mount-point=''
cd -

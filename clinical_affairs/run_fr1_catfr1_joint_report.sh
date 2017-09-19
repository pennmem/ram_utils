#!/usr/bin/env bash
cd ../tests/fr1_catfr1_joint_report
python fr1_catfr1_joint_report.py --subject=$1 --workspace-dir=/scratch/RAM_reports/FR1_catFR1_joint_reports --mount-point='' "${@:2}"
cd -

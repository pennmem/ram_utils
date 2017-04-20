#!/usr/bin/env bash

cd ../tests/fr5_report
python fr5_report.py --subject=$1 --task=FR5 --workspace-dir=/scratch/pwanda/FR_reports --mount-point='' "${@:2}"
cd -

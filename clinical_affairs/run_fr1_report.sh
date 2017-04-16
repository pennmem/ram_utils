#!/usr/bin/env bash

cd ../tests/fr1_report
python fr1_report.py --subject=$1 --task=FR1 --workspace-dir=/scratch/leond/FR_reports --mount-point='' "${@:2}"
cd -

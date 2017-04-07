#!/usr/bin/env bash

cd ../tests/fr_stim_report
python fr_stim_report.py --subject=$1 --task=FR5 --workspace-dir=/scratch/pwanda/FR_reports --mount-point='' "${@:2}"
cd -

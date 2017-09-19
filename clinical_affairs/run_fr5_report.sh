#!/usr/bin/env bash

cd ../tests/fr5_report
python fr5_report.py --subject=$1 --task=FR5 --workspace-dir=/scratch/RAM_reports/FR5_reports --mount-point='' "${@:2}"
cd -

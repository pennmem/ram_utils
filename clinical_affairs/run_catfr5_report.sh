#!/usr/bin/env bash

cd ../tests/fr5_report
python fr5_report.py --subject=$1 --task=catFR5 --workspace-dir=/scratch/RAM_reports/catFR5_reports --mount-point='' "${@:2}"
cd -

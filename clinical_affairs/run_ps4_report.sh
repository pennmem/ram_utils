#!/usr/bin/env bash

cd ../tests/ps4_report
python ps4_report.py --subject $1 --task $2 --workspace-dir /scratch/RAM_reports/PS4_reports/ --mount-point="" "${@:3}"
cd -
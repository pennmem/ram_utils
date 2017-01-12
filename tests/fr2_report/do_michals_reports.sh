#!/usr/bin/env bash

SUBJECTS=(R1050M R1111M R1176M R1177M)

for subject in "${SUBJECTS[@]}"
    do
    echo ${subject}
    python fr2_report.py --subject=${subject} --task=FR2 --workspace-dir=/scratch/leond/FR2_reports/ram_freqs/stim --stim=True
    python fr2_report.py --subject=${subject} --task=FR2 --workspace-dir=/scratch/leond/FR2_reports/ram_freqs/no_stim --stim=False
    done

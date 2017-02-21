cd ../tests/pal_stim_report
python pal_stim_report.py --subject=$1 --task=PAL3 --workspace-dir=/scratch/pwanda/PAL_reports --mount-point='' "${@:2}"
cd -

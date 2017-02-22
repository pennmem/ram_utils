cd ../tests/pal1_report
python pal1_report.py --subject=$1 --task=PAL1 --workspace-dir=/scratch/pwanda/PAL_reports --mount-point='' "${@:2}"
cd -

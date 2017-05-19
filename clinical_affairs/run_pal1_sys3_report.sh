cd ../tests/pal1_sys3_report
python pal1_report.py --subject=$1 --task=PAL1 --workspace-dir=/scratch/RAM_reports/PAL_reports_sys3 --mount-point='' "${@:2}"
cd -

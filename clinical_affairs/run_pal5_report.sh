cd ../tests/pal5_report
python pal5_report.py --subject=$1 --task=PAL5 --workspace-dir=/scratch/RAM_reports/PAL5_reports --mount-point='' "${@:2}"
cd -

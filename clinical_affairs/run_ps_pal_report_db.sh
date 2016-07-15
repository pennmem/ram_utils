cd ../tests/ps_pal_report_db
python ps_pal_report.py --subject=$1 --experiment=$2 --workspace-dir=/scratch/pwanda/$2 --mount-point=""
cd -

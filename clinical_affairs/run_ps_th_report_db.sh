cd ../tests/ps_th_report_db
python ps_th_report.py --subject=$1 --experiment=$2 --workspace-dir=/scratch/pwanda/$2 --mount-point=""
cd -

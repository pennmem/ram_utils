cd ../tests/ps_pal_report
python ps_pal_report.py --subject=$1 --experiment=$2 --workspace-dir=/scratch/pwanda/PAL_reports --mount-point="" "${@:3}"
cd -

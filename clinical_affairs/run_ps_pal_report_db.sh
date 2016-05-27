cd ../tests/ps_pal_report_db
python ps_pal_report.py --subject=$1 --experiment=$2 --workspace-dir=/scratch/pwanda/$2 --mount-point="" --python-path=/home2/losu/ram_utils --python-path=/home2/losu/python/ptsa_latest --python-path=/home1/mswat/extra_libs
cd -

if [ -z $RAM_UTILS_DEBUG ]
then
workspace_dir=/scratch/RAM_reports/${1}
else
workspace_dir=/scratch/leond/reports
fi

read -p 'SUBJECT    :' subject
read -p 'EXPERIMENT :' experiment


cd ..
python -m ram_utils.reports --task=$experiment --subject=$subject --workspace-dir=${workspace_dir} --mount-point='' "${@:3}"
cd -
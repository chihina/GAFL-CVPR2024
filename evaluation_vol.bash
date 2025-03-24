IFS_BACKUP=$IFS
IFS=$'\n'

model_array=(
    # Please fill in the model path
    # 'model_name'
    )

for model in ${model_array[@]}; do
  echo $model
  python scripts/eval_volleyball_stage2_gr.py 0 $model
done
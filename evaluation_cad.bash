IFS_BACKUP=$IFS
IFS=$'\n'

model_array=(
    # Please fill in the model path
    # 'model_name'
  )

for model in ${model_array[@]}; do
  echo $model
  python scripts/eval_collective_stage2_gr.py 1 $model
done
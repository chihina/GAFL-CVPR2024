IFS_BACKUP=$IFS
IFS=$'\n'

model_array=(
    # Please fill in the model path
    '[GAFL-PAF CAD_stage2]<2025-03-24_13-44-20>'
  )

for model in ${model_array[@]}; do
  echo $model
  python scripts/eval_collective_stage2_gr.py 1 $model
done
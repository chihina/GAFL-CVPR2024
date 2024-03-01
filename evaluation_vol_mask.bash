IFS_BACKUP=$IFS
IFS=$'\n'

# model='[GR ours rand mask 5_stage2]<2023-10-16_22-26-54>'
model='[GR ours_stage2]<2023-07-08_08-59-54>'

eval_mask_num_array=(
  11
  10
  9
  8
  7  
  6
  5
  4
  3
  2
  1
  0
)

echo $model
for eval_mask_num in ${eval_mask_num_array[@]}; do
  echo $eval_mask_num
  python scripts/eval_volleyball_stage2_gr.py 3 $model -eval_mask_num $eval_mask_num
done
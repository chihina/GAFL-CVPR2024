IFS_BACKUP=$IFS
IFS=$'\n'

model_array=(
    # Please fill in the model path
    # '[GR ours rand mask 5_stage2]<2023-10-16_22-26-54>'
    '[GR ours_stage2]<2023-07-08_08-59-54>'
    # '[GR ours recon feat random mask 6_stage2]<2023-10-18_09-34-15>'
    # '[GR ours recon feat random mask 0 w temp cond_stage2]<2023-11-10_13-26-06>'
    )

for model in ${model_array[@]}; do
  echo $model
  python scripts/clustering_volleyball.py $1 0 $model
done
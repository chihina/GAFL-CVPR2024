IFS_BACKUP=$IFS
IFS=$'\n'

model_array=(
    # Please fill in the model path
    # '[CAD GR prev ImageNet pretrain autoencoder crop_stage2]<2023-07-29_18-37-59>'
    # '[CAD GR prev ImageNet pretrain VGG crop_stage2]<2023-07-29_09-51-27>'
    # '[CAD GR prev ImageNet pretrain HRN crop_stage2]<2023-07-29_18-37-23>'
    '[CAD GR ours DIN PAF_stage2]<2024-01-25_09-08-40>'
    '[CAD GR ours HIGCIN PAF_stage2]<2024-01-25_11-38-17>'
    # '[CAD GA ours recon rand mask 6_stage2]<2023-10-26_16-18-47>'
    # '[Higcin_collective_stage2]<2023-11-10_16-06-02>'
    # '[Dynamic_collective_stage2]<2023-11-04_12-40-36>'
    '[CAD GR ours DIN PAC_stage2]<2024-01-25_07-55-29>'
    '[CAD GR ours HIGCIN PAC_stage2]<2024-01-25_07-52-11>'
    # '[CAD GA ours finetune_stage2]<2023-10-20_10-05-47>'
    # '[CAD GA ours rand mask 0_stage2]<2023-10-23_13-28-02>'
  )

for model in ${model_array[@]}; do
  echo $model
  python scripts/eval_collective_stage2_gr.py 1 $model
done
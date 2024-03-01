IFS_BACKUP=$IFS
IFS=$'\n'

model_array=(
    # Please fill in the model path
    # '[PAC_DC_w_act_loss_use_gt_act_stage2]<2024-01-23_16-35-25>'
    # '[PAC_DC_wo_act_loss_use_gt_act_stage2]<2024-01-23_16-53-09>'
    # '[PAC_DC_w_act_loss_use_pred_act_stage2]<2024-01-23_21-22-46>'
    # '[PAC_DC_wo_act_loss_use_pred_act_stage2]<2024-01-23_21-25-29>'
    # '[GR prev ImageNet pretrain autoencoder crop_stage2]<2023-07-12_11-25-04>'
    # '[GR prev ImageNet pretrain VGG crop_stage2]<2023-07-12_11-12-53>'
    # '[GR prev ImageNet pretrain HRN crop sigmoid_stage2]<2023-07-13_08-09-45>'
    # '[GR ours HIGCIN PAF_stage2]<2024-01-25_09-01-07>'
    # '[GR ours DIN PAF_stage2]<2024-01-25_09-03-08>'
    # '[GR ours recon feat random mask 6_stage2]<2023-10-18_09-34-15>'
    # '[Higcin_volleyball_stage2]<2023-11-10_16-03-32>'
    # '[Dynamic_volleyball_stage2]<2023-11-06_13-33-37>'
    # '[GA ours finetune_stage2]<2023-07-07_09-43-38>'
    # '[GR ours HIGCIN PAC_stage2]<2024-01-24_10-51-29>'
    # '[GR ours HIGCIN PAC_stage2]<2024-01-25_08-49-08>'
    # '[GR ours DIN PAC_stage2]<2024-01-24_10-54-45>'
    # '[GR ours rand mask 5_stage2]<2023-10-16_22-26-54>'
    # '[GR ours rand mask 0 w REC act_stage2]<2024-01-25_16-59-55>'
    # '[GR ours HIGCIN PAC rand mask 5_stage2]<2024-01-26_09-12-38>'
    # '[GR ours DIN PAC rand mask 5_stage2]<2024-01-26_09-11-37>'
    # '[GR ours rand mask 5 w same enc dual path_stage2]<2024-01-26_22-57-24>'
    # '[GR ours rand mask 5 wo backbone pretrain_stage2]<2024-01-26_22-52-05>'
    # '[GR ours rand mask 5 w REC act 10_stage2]<2024-01-27_00-10-40>'
    # '[GR ours rand mask 5 w REC act 20_stage2]<2024-01-27_00-11-47>'
    # '[GR ours rand mask 5 w REC act 50_stage2]<2024-01-27_00-13-15>'
    # '[GR ours rand mask 5 w REC act_stage2]<2024-01-26_08-15-35>'
    # '[GR ours rand mask 5 w GT act 10_stage2]<2024-01-27_19-56-02>'
    # '[GR ours rand mask 5 w GT act 20_stage2]<2024-01-27_19-56-23>'
    # '[GR ours rand mask 5 w GT act 50_stage2]<2024-01-27_19-56-32>'
    # '[GR ours rand mask 5 w finetune jae_stage3]<2024-01-29_15-44-44>'
    # '[GR ours rand mask 5 wo finetune jae_stage3]<2024-01-29_15-45-03>'
    # '[GR ours rand mask 5 wo backbone pretrain wo imagenet pretrain_stage2]<2024-01-29_23-27-35>'
    # '[GR ours rand mask 5 wo finetune jae w backbone pretrain_stage3]<2024-01-29_22-27-16>'
    # '[GR dual_ai wo backbone pretrain wo imagenet pretrain_stage2]<2024-01-29_23-29-50>'
    # '[GR ours recon feat random mask 6 wo finetune jae_stage3]<2024-01-30_08-27-25>'
    # '[GR ours recon feat random mask 6 w finetune jae_stage3]<2024-01-30_08-26-04>'
    # '[VOL_GR PAC_DC act loss False pseudo act gt True 1e3 w backbone pretrain_stage2]<2024-02-05_08-57-56>'
    # '[VOL_GR PAC_DC act loss False pseudo act gt True 1e4 w backbone pretrain_stage2]<2024-02-05_08-57-18>'
    # '[VOL_GR PAC_DC act loss False pseudo act gt True 1e5 w backbone pretrain_stage2]<2024-02-05_08-57-44>'
    # '[VOL_GR PAC_DC act loss True pseudo act gt True single_branch_transformer_stage2]<2024-02-06_19-19-15>'
    # '[VOL_GR PAC_DC act loss False pseudo act gt False person_action_recognizor finetune False_stage2]<2024-02-07_16-56-35>'
    '[VOL_GR PAC_DC act loss True pseudo act gt False person_action_recognizor finetune False_stage2]<2024-02-07_16-56-20>'
    )

for model in ${model_array[@]}; do
  echo $model
  python scripts/eval_volleyball_stage2_gr.py 7 $model
done
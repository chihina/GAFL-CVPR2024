import json
import os
import sys
import pandas as pd
import numpy as np
sys.path.append(".")
from analysis_utils import write_excel, refine_excel

saved_result_dir = os.path.join('result')
saved_result_analysis_dir = os.path.join('result_analysis')

analyze_name_list = []
analyze_name_list.append('[GR ours recon feat without pretrain and backbone train_stage2]<2023-07-09_21-59-22>')
# analyze_name_list.append('[GR ours recon feat random mask 2_stage2]<2023-08-24_13-50-18>')
analyze_name_list.append('[GR ours recon kinetics feat_stage2]<2023-09-10_21-35-09>')
analyze_name_list.append('[GR ours recon vgg16 ucf101 feat crop single_stage2]<2023-09-11_11-26-53>')
analyze_name_list.append('[GR ours recon vgg16 ucf101 feat crop single official norm_stage2]<2023-09-20_09-29-56>')
# analyze_name_list.append('[GR ours recon vgg16 ucf101 feat crop single rand mask 2_stage2]<2023-09-14_07-57-42>')
# analyze_name_list.append('[GR ours recon vgg16 ucf101 feat crop single rand mask 2 offical norm_stage2]<2023-09-20_09-30-17>')


# define model names
model_name_list = []
model_name_list.append(r'Ours (ImageNet)')
# model_name_list.append(r'Ours (ImageNet) rand mask 2')
model_name_list.append(r'Ours (Kinetics)')
model_name_list.append(r'Ours (UCF101)')
model_name_list.append(r'Ours (UCF101) off.')
# model_name_list.append(r'Ours (UCF101) rand mask 2')
# model_name_list.append(r'Ours (UCF101) rand mask 2 off.')

# write excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)
refine_excel(save_excel_file_path, is_paper=True)

save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}_wo_tex.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)
refine_excel(save_excel_file_path, is_paper=False)
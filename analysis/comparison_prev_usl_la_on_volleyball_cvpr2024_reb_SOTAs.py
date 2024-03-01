import json
import os
import sys
import pandas as pd
import numpy as np
sys.path.append(".")
from analysis_utils import write_excel, refine_excel, collect_visualization_results

saved_result_dir = os.path.join('result')
saved_result_analysis_dir = os.path.join('result_analysis')

# define analyze model type
analyze_name_list = []
# analyze_name_list.append('[GR prev ImageNet pretrain autoencoder_stage2]<2023-07-11_07-18-03>')
# analyze_name_list.append('[GR prev ImageNet pretrain autoencoder crop_stage2]<2023-07-12_11-25-04>')
# analyze_name_list.append('[GR prev ImageNet pretrain VGG_stage2]<2023-07-10_14-03-57>')
# analyze_name_list.append('[GR prev ImageNet pretrain VGG crop_stage2]<2023-07-12_11-12-53>')
# analyze_name_list.append('[GR prev ImageNet pretrain HRN_stage2]<2023-07-11_16-55-27>')
# analyze_name_list.append('[GR prev ImageNet pretrain HRN crop sigmoid_stage2]<2023-07-13_08-09-45>')

# analyze_name_list.append('[GR ours HIGCIN PAF_stage2]<2024-01-25_09-01-07>')
analyze_name_list.append('[GR ours HIGCIN PAF_stage2]<2024-01-25_09-01-07>')
# analyze_name_list.append('[GR ours DIN PAF_stage2]<2024-01-25_09-03-08>')
analyze_name_list.append('[GR ours DIN PAF_stage2]<2024-01-25_09-03-08>')
# analyze_name_list.append('[GR ours recon feat random mask 6_stage2]<2023-10-18_09-34-15>')
analyze_name_list.append('[GR ours recon feat random mask 6_stage2]<2023-10-18_09-34-15>')

# define model names (use crop (C))
model_name_list = []
# model_name_list.append(r'Compact128-R')
# model_name_list.append(r'Compact128-C')
# model_name_list.append(r'VGG19-R')
# model_name_list.append(r'VGG19-C')
# model_name_list.append(r'HRN-R')
# model_name_list.append(r'HRN-C')
# model_name_list.append(r'Ours-ind (HiGCIN)')
model_name_list.append(r'Ours-grp (HiGCIN)')
# model_name_list.append(r'Ours-ind (DIN)')
model_name_list.append(r'Ours-grp (DIN)')
# model_name_list.append(r'Ours-ind (Dual-AI)')
model_name_list.append(r'Ours-grp (Dual-AI)')

# write and refine excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)
refine_excel(save_excel_file_path, is_paper=True)

# collect visualization results
save_vis_dir = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}')
collect_visualization_results(save_vis_dir, analyze_name_list, model_name_list, saved_result_dir)
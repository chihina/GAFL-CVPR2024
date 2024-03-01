import json
import os
import sys
import pandas as pd
import numpy as np
sys.path.append(".")
from analysis_utils import write_excel, refine_excel, collect_visualization_results

saved_result_dir = os.path.join('result')
saved_result_analysis_dir = os.path.join('result_analysis')

analyze_name_list = []
# analyze_name_list.append('[CAD GR prev ImageNet pretrain autoencoder_stage2]<2023-07-31_10-07-31>')
analyze_name_list.append('[CAD GR prev ImageNet pretrain autoencoder crop_stage2]<2023-07-29_18-37-59>')
# analyze_name_list.append('[CAD GR prev ImageNet pretrain VGG_stage2]<2023-07-31_09-37-52>')
analyze_name_list.append('[CAD GR prev ImageNet pretrain VGG crop_stage2]<2023-07-29_09-51-27>')
# analyze_name_list.append('[CAD GR prev ImageNet pretrain HRN_stage2]<2023-07-31_10-08-36>')
analyze_name_list.append('[CAD GR prev ImageNet pretrain HRN crop_stage2]<2023-07-29_18-37-23>')
analyze_name_list.append('[CAD GA ours recon rand mask 6_stage2]<2023-10-26_16-18-47>')
analyze_name_list.append('[CAD GA ours recon rand mask 6_stage2]<2023-10-26_16-18-47>')


# define model names
model_name_list = []
# model_name_list.append(r'Compact128-R')
model_name_list.append(r'Compact128-C')
# model_name_list.append(r'VGG19-R')
model_name_list.append(r'VGG19-C')
# model_name_list.append(r'HRN-R')
model_name_list.append(r'HRN-C')
model_name_list.append(r'Ours-ind')
model_name_list.append(r'Ours-grp')

# write and refine excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)
refine_excel(save_excel_file_path, is_paper=True)

# collect visualization results
save_vis_dir = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}')
collect_visualization_results(save_vis_dir, analyze_name_list, model_name_list, saved_result_dir)
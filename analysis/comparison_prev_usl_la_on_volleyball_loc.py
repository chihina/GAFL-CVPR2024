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
analyze_name_list.append('[GR prev ImageNet pretrain autoencoder crop_stage2]<2023-07-12_11-25-04>')
analyze_name_list.append('[GR prev ImageNet pretrain autoencoder crop with loc_stage2]<2023-11-24_10-05-23>')
analyze_name_list.append('[GR prev ImageNet pretrain VGG crop_stage2]<2023-07-12_11-12-53>')
analyze_name_list.append('[GR prev ImageNet pretrain VGG crop with loc_stage2]<2023-11-24_10-02-00>')
analyze_name_list.append('[GR prev ImageNet pretrain HRN crop sigmoid_stage2]<2023-07-13_08-09-45>')
analyze_name_list.append('[GR prev ImageNet pretrain HRN crop sigmoid with loc_stage2]<2023-11-24_10-06-11>')
analyze_name_list.append('[GR ours recon feat random mask 6_stage2]<2023-10-18_09-34-15>')
analyze_name_list.append('[GR ours recon feat random mask 6_stage2]<2023-10-18_09-34-15>')

# define model names
model_name_list = []
model_name_list.append(r'Compact128-C')
model_name_list.append(r'Compact128-C with location')
model_name_list.append(r'VGG19-C')
model_name_list.append(r'VGG19-C with location')
model_name_list.append(r'HRN-C')
model_name_list.append(r'HRN-C with location')
model_name_list.append(r'Ours-ind')
model_name_list.append(r'Ours-grp')

# write and refine excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)
refine_excel(save_excel_file_path, is_paper=True)

# collect visualization results
save_vis_dir = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}')
collect_visualization_results(save_vis_dir, analyze_name_list, model_name_list, saved_result_dir)
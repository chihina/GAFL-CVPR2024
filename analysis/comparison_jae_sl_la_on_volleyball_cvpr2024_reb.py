import json
import os
import sys
import pandas as pd
import numpy as np
sys.path.append(".")
from analysis_utils import write_excel, refine_excel
from analysis_utils import collect_visualization_results, collect_nn_results

saved_result_dir = os.path.join('result')
saved_result_analysis_dir = os.path.join('result_analysis')

analyze_name_list = []
analyze_name_list.append('[GR ours rand mask 5 wo finetune jae_stage3]<2024-01-29_15-45-03>')
analyze_name_list.append('[GR ours rand mask 5 wo finetune jae w backbone pretrain_stage3]<2024-01-29_22-27-16>')
analyze_name_list.append('[GR ours rand mask 5 w finetune jae_stage3]<2024-01-29_15-44-44>')
analyze_name_list.append('[GR ours recon feat random mask 6 wo finetune jae_stage3]<2024-01-30_08-27-25>')
analyze_name_list.append('[GR ours recon feat random mask 6 w finetune jae_stage3]<2024-01-30_08-26-04>')

# define model names
model_name_list = []
model_name_list.append(r'Ours GAFL-PAC (full scratch)')
model_name_list.append(r'Ours GAFL-PAC (w backbone pretrain)')
model_name_list.append(r'Ours GAFL-PAC (w finetune)')
model_name_list.append(r'Ours GAFL-PAF (full scratch)')
model_name_list.append(r'Ours GAFL-PAF (w finetune)')

# write and refine excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True, is_clustering=False, is_jae=True)
refine_excel(save_excel_file_path, is_paper=True, is_clustering=False, is_jae=True)

# collect visualization results
save_vis_dir = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}')
collect_visualization_results(save_vis_dir, analyze_name_list, model_name_list, saved_result_dir)

# collect nn results
save_nn_dir = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}')
collect_nn_results(save_nn_dir, analyze_name_list, model_name_list, saved_result_dir)
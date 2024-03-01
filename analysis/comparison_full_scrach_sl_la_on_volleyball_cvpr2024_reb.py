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
analyze_name_list.append('[GR dual_ai wo backbone pretrain wo imagenet pretrain_stage2]<2024-01-29_23-29-50>')
analyze_name_list.append('[GR ours rand mask 5 wo backbone pretrain wo imagenet pretrain_stage2]<2024-01-29_23-27-35>')
analyze_name_list.append('[GR ours rand mask 5_stage2]<2023-10-16_22-26-54>')

# define model names
model_name_list = []
model_name_list.append(r'Dual-ai (full scratch)')
model_name_list.append(r'Ours-grp (full scratch)')
model_name_list.append(r'Ours-grp')

# write and refine excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True, is_clustering=False)
refine_excel(save_excel_file_path, is_paper=True, is_clustering=False)

# collect visualization results
save_vis_dir = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}')
collect_visualization_results(save_vis_dir, analyze_name_list, model_name_list, saved_result_dir)

# collect nn results
save_nn_dir = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}')
collect_nn_results(save_nn_dir, analyze_name_list, model_name_list, saved_result_dir)
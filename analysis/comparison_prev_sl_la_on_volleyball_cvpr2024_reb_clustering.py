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
analyze_name_list.append('[Higcin_volleyball_stage2]<2023-11-10_16-03-32>')
analyze_name_list.append('[Dynamic_volleyball_stage2]<2023-11-06_13-33-37>')
analyze_name_list.append('[GA ours finetune_stage2]<2023-07-07_09-43-38>')
analyze_name_list.append('[GR ours rand mask 5_stage2]<2023-10-16_22-26-54>')
analyze_name_list.append('[GR ours rand mask 5_stage2]<2023-10-16_22-26-54>')

# define model names
model_name_list = []
model_name_list.append(r'HiGCIN')
model_name_list.append(r'DIN')
model_name_list.append(r'Dual-AI')
model_name_list.append(r'Ours-ind')
model_name_list.append(r'Ours-grp')

# write and refine excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True, is_clustering=True)
refine_excel(save_excel_file_path, is_paper=True, is_clustering=True)

# collect visualization results
save_vis_dir = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}')
collect_visualization_results(save_vis_dir, analyze_name_list, model_name_list, saved_result_dir)

# collect nn results
save_nn_dir = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}')
collect_nn_results(save_nn_dir, analyze_name_list, model_name_list, saved_result_dir)
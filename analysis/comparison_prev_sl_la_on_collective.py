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
analyze_name_list.append('[Higcin_collective_stage2]<2023-11-10_16-06-02>')
analyze_name_list.append('[Dynamic_collective_stage2]<2023-11-04_12-40-36>')
analyze_name_list.append('[CAD GA ours finetune_stage2]<2023-10-20_10-05-47>')
analyze_name_list.append('[CAD GA ours rand mask 0_stage2]<2023-10-23_13-28-02>')
analyze_name_list.append('[CAD GA ours rand mask 0_stage2]<2023-10-23_13-28-02>')

# define model names
model_name_list = []
model_name_list.append(r'HiGCIN')
model_name_list.append(r'DIN')
model_name_list.append(r'Dual-AI')
model_name_list.append(r'Ours-ind')
model_name_list.append(r'Ours-grp')

# write and refine excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)
refine_excel(save_excel_file_path, is_paper=True)

# collect visualization results
save_vis_dir = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}')
collect_visualization_results(save_vis_dir, analyze_name_list, model_name_list, saved_result_dir)
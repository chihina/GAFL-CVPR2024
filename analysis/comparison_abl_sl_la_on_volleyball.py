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
# analyze_name_list.append('[GR ours rand mask 5 wo temp_stage2]<2023-10-19_16-30-23>')
analyze_name_list.append('[GR ours rand mask 5 wo pos_cond_stage2]<2023-10-19_16-30-59>')
analyze_name_list.append('[GR ours rand mask 5_stage2]<2023-10-16_22-26-54>')

# define model names
model_name_list = []
model_name_list.append(r'Ours w/o $\bm{F}_{loc}$')
# model_name_list.append(r'Ours w/o trans.')
# model_name_list.append(r'Ours w/o $\bm{F}_{tmp}$')
# model_name_list.append(r'Ours w/o temp.')
model_name_list.append(r'Ours-grp')

# write excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)

# refine excel
refine_excel(save_excel_file_path, is_paper=True)

# collect visualization results
save_vis_dir = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}')
collect_visualization_results(save_vis_dir, analyze_name_list, model_name_list, saved_result_dir)
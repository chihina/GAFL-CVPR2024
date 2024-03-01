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
analyze_name_list.append('[GR ours rand mask 5 w REC act 10_stage2]<2024-01-27_00-10-40>')
analyze_name_list.append('[GR ours rand mask 5 w REC act 20_stage2]<2024-01-27_00-11-47>')
analyze_name_list.append('[GR ours rand mask 5 w REC act 50_stage2]<2024-01-27_00-13-15>')
analyze_name_list.append('[GR ours rand mask 5 w REC act_stage2]<2024-01-26_08-15-35>')
analyze_name_list.append('[GR ours rand mask 5 w GT act 10_stage2]<2024-01-27_19-56-02>')
analyze_name_list.append('[GR ours rand mask 5 w GT act 20_stage2]<2024-01-27_19-56-23>')
analyze_name_list.append('[GR ours rand mask 5 w GT act 50_stage2]<2024-01-27_19-56-32>')
analyze_name_list.append('[GR ours rand mask 5_stage2]<2023-10-16_22-26-54>')
analyze_name_list.append('[GR ours recon feat random mask 6_stage2]<2023-10-18_09-34-15>')

# define model names
model_name_list = []
model_name_list.append(r'Ours-grp (10%) for REC')
model_name_list.append(r'Ours-grp (20%) for REC')
model_name_list.append(r'Ours-grp (50%) for REC')
model_name_list.append(r'Ours-grp (100%) for REC')
model_name_list.append(r'Ours-grp (GAFL-PAC 10%)')
model_name_list.append(r'Ours-grp (GAFL-PAC 20%)')
model_name_list.append(r'Ours-grp (GAFL-PAC 50%)')
model_name_list.append(r'Ours-grp (GAFL-PAC)')
model_name_list.append(r'Ours-grp (GAFL-PAF)')

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
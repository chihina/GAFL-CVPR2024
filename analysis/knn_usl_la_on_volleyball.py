import json
import os
import sys
import pandas as pd
import numpy as np
sys.path.append(".")
from analysis_utils import write_excel, refine_excel

saved_result_dir = os.path.join('result')
saved_result_analysis_dir = os.path.join('result_analysis')
analyze_name = '[GR ours recon feat random mask 6_stage2]<2023-10-18_09-34-15>'

# write excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')

json_file_path_gar = os.path.join(saved_result_dir, analyze_name, 'eval_gar_metrics.json')
with open(json_file_path_gar, 'r') as f:
    eval_results_dic_gar = json.load(f)

eval_results_list = []
for mode in ['people', 'scene']:
    eval_results_dic_update = {}
    for k in range(1, 21):
        eval_results_dic_update[f'{k}NN'] = eval_results_dic_gar[f'GAR accuracy {k}NN (group activity) ({mode})'] * 100
    eval_results_list.append(list(eval_results_dic_update.values()))
    eval_metrics_list = list(eval_results_dic_update.keys())

eval_results_array = np.array(eval_results_list)
df_eval_results = pd.DataFrame(eval_results_array, ['people', 'scene'], eval_metrics_list)
df_eval_results.to_excel(save_excel_file_path, sheet_name='all')

# refine excel
refine_excel(save_excel_file_path)
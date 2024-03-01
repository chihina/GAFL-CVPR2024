import json
import os
import sys
import pandas as pd
import numpy as np
sys.path.append(".")
from analysis_utils import write_excel, refine_excel

saved_result_dir = os.path.join('result')
saved_result_analysis_dir = os.path.join('result_analysis')

analyze_name_list = []
# analyze_name_list.append('[GR ours loc_stage2]<2023-07-10_11-26-26>')
analyze_name_list.append('[GR ours loc active mask inv gen iar 0.0_stage2]<2023-09-19_11-32-54>')
# analyze_name_list.append('[GR ours app_stage2]<2023-07-10_11-42-35>')
analyze_name_list.append('[GR ours app active mask inv gen iar 0.0_stage2]<2023-09-19_11-33-22>')
# analyze_name_list.append('[GR ours_stage2]<2023-07-08_08-59-54>')
analyze_name_list.append('[GR ours active mask inv_stage2]<2023-08-25_08-29-32>')

# define model names
model_name_list = []
model_name_list.append(r'Ours ($\bm{F}_{set}=\bm{F}_{loc}$)')
model_name_list.append(r'Ours ($\bm{F}_{set}=\bm{F}_{app}$)')
model_name_list.append(r'Ours ($\bm{F}_{set}=\bm{F}_{loc}+\bm{F}_{app}$)')

# write excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)

# refine excel
refine_excel(save_excel_file_path, is_paper=True)
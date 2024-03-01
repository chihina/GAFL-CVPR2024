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
analyze_name_list.append('[GR ours_stage2]<2023-07-08_08-59-54>')
analyze_name_list.append('[GR ours gen iar ratio 0_stage2]<2023-09-11_21-23-24>')
analyze_name_list.append('[GR ours gen iar ratio 0.1_stage2]<2023-09-11_21-23-41>')
analyze_name_list.append('[GR ours gen iar ratio 0.2_stage2]<2023-09-11_21-24-02>')
analyze_name_list.append('[GR ours gen iar ratio 0.3_stage2]<2023-09-12_21-27-31>')
analyze_name_list.append('[GR ours gen iar ratio 0.4_stage2]<2023-09-12_21-28-34>')
analyze_name_list.append('[GR ours gen iar ratio 0.5_stage2]<2023-08-31_08-42-23>')
analyze_name_list.append('[GR ours gen iar ratio 0.6_stage2]<2023-09-12_21-28-53>')

# define model names
model_name_list = []
model_name_list.append(r'Ours')
model_name_list.append(r'Ours with gen (0)')
model_name_list.append(r'Ours with gen (0.1)')
model_name_list.append(r'Ours with gen (0.2)')
model_name_list.append(r'Ours with gen (0.3)')
model_name_list.append(r'Ours with gen (0.4)')
model_name_list.append(r'Ours with gen (0.5)')
model_name_list.append(r'Ours with gen (0.6)')

# write excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)

# refine excel
refine_excel(save_excel_file_path, is_paper=True)
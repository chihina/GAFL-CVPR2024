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
analyze_name_list.append('[GR ours loc_stage2]<2023-07-10_11-26-26>')
analyze_name_list.append('[GR ours loc active mask inv gen iar 0.0_stage2]<2023-09-19_11-32-54>')
analyze_name_list.append('[GR ours loc active mask inv gen iar 0.2_stage2]<2023-09-17_10-21-33>')
analyze_name_list.append('[GR ours loc active mask inv gen iar 0.4_stage2]<2023-09-17_10-21-56>')
analyze_name_list.append('[GR ours loc active mask inv gen iar 0.6_stage2]<2023-09-17_10-22-44>')
analyze_name_list.append('[GR ours loc active mask inv gen iar 0.8_stage2]<2023-09-17_10-23-01>')
analyze_name_list.append('[GR ours loc active mask inv gen iar 1.0_stage2]<2023-09-17_10-23-58>')

# define model names
model_name_list = []
model_name_list.append(r'Ours')
model_name_list.append(r'Ours with gen (0.0)')
model_name_list.append(r'Ours with gen (0.2)')
model_name_list.append(r'Ours with gen (0.4)')
model_name_list.append(r'Ours with gen (0.6)')
model_name_list.append(r'Ours with gen (0.8)')
model_name_list.append(r'Ours with gen (1.0)')

# write excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)

# refine excel
refine_excel(save_excel_file_path, is_paper=True)
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
analyze_name_list.append('[GR ours active mask_stage2]<2023-08-24_09-51-09>')
analyze_name_list.append('[GR ours active mask inv 0_stage2]<2023-09-20_17-42-13>')
analyze_name_list.append('[GR ours active mask inv 1_stage2]<2023-09-20_17-42-02>')
analyze_name_list.append('[GR ours active mask inv 2_stage2]<2023-09-20_17-41-51>')
analyze_name_list.append('[GR ours active mask inv 3_stage2]<2023-09-20_17-41-39>')
analyze_name_list.append('[GR ours active mask inv 4_stage2]<2023-09-20_17-41-26>')
analyze_name_list.append('[GR ours active mask inv 5_stage2]<2023-09-20_17-41-14>')
analyze_name_list.append('[GR ours active mask inv 6_stage2]<2023-09-20_17-41-01>')
analyze_name_list.append('[GR ours active mask inv_stage2]<2023-08-25_08-29-32>')

# define model names
model_name_list = []
model_name_list.append(r'Ours')
model_name_list.append(r'Ours (7)')
model_name_list.append(r'Ours (inv. 0)')
model_name_list.append(r'Ours (inv. 1)')
model_name_list.append(r'Ours (inv. 2)')
model_name_list.append(r'Ours (inv. 3)')
model_name_list.append(r'Ours (inv. 4)')
model_name_list.append(r'Ours (inv. 5)')
model_name_list.append(r'Ours (inv. 6)')
model_name_list.append(r'Ours (inv. 7)')

# write excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)

# refine excel
refine_excel(save_excel_file_path, is_paper=True)
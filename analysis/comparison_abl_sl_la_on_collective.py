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
# analyze_name_list.append('[CAD GR ours rand mask 0 wo trans_stage2]<2023-10-26_23-56-26>')
analyze_name_list.append('[CAD GR ours rand mask 0 wo pos_cond_stage2]<2023-10-26_23-55-33>')
# analyze_name_list.append('[CAD GR ours rand mask 0 wo temp_stage2]<2023-10-26_23-56-00>')
analyze_name_list.append('[CAD GA ours rand mask 0_stage2]<2023-10-23_13-28-02>')
analyze_name_list.append('[CAD GA ours rand mask 0 w temp cond_stage2]<2023-11-08_20-56-24>')

# define model names
model_name_list = []
# model_name_list.append(r'Ours w/o trans.')
# model_name_list.append(r'Ours w/o temp.')
model_name_list.append(r'Ours w/o $\bm{F}_{loc}$')
model_name_list.append(r'Ours')
model_name_list.append(r'Ours')

# write excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)

# refine excel
refine_excel(save_excel_file_path, is_paper=True)
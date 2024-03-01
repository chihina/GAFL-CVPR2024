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

# analyze_name_list.append('[GR ours recon feat random mask 6 wo temp_stage2]<2023-10-19_08-56-43>')
# analyze_name_list.append('[GR ours recon feat random mask 6 wo pos_cond_stage2]<2023-10-19_08-57-19>')
# analyze_name_list.append('[GR ours recon feat random mask 6_stage2]<2023-10-18_09-34-15>')
# analyze_name_list.append('[GR ours recon feat random mask 6 w temp cond_stage2]<2023-11-08_20-52-31>')

analyze_name_list.append('[GR ours recon feat random mask 8 w temp cond wo pos_cond_stage2]<2023-11-12_11-48-39>')
analyze_name_list.append('[GR ours recon feat random mask 8 w temp cond wo temp_cond_stage2]<2023-11-12_19-04-24>')
analyze_name_list.append('[GR ours recon feat random mask 8 w temp cond wo trans_stage2]<2023-11-12_14-48-02>')
analyze_name_list.append('[GR ours recon feat random mask 8 w temp cond wo temp_info_stage2]<2023-11-13_08-10-27>')
analyze_name_list.append('[GR ours recon feat random mask 8 w temp cond_stage2]<2023-11-10_13-32-14>')

# define model names
model_name_list = []
model_name_list.append(r'Ours w/o $\bm{F}_{loc}$')
model_name_list.append(r'Ours w/o $\bm{F}_{tmp}$')
model_name_list.append(r'Ours w/o trans.')
model_name_list.append(r'Ours w/o temp.')
model_name_list.append(r'Ours-grp')

# write excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)

# refine excel
refine_excel(save_excel_file_path, is_paper=True)
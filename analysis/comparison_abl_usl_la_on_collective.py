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
# analyze_name_list.append('[CAD GR ours recon rand mask 0 wo pos_cond_stage2]<2023-10-26_23-49-18>')
# analyze_name_list.append('[CAD GR ours recon rand mask 0_stage2]<2023-10-19_22-26-11>')
# analyze_name_list.append('[CAD GR ours recon rand mask 4 wo pos_cond_stage2]<2023-10-19_22-36-53>')
# analyze_name_list.append('[CAD GR ours recon rand mask 4_stage2]<2023-08-30_10-15-44>')
analyze_name_list.append('[CAD GR ours recon rand mask 6 wo pos_cond_stage2]<2023-11-04_15-35-36>')
analyze_name_list.append('[CAD GA ours recon rand mask 6_stage2]<2023-10-26_16-18-47>')
analyze_name_list.append('[CAD GR ours recon rand mask 6 w temp cond_stage2]<2023-11-08_20-58-31>')

# define model names
model_name_list = []
# model_name_list.append(r'Ours-ind w/o $\bm{F}_{loc}$')
# model_name_list.append(r'Ours-ind')
model_name_list.append(r'Ours-ind w/o $\bm{F}_{loc}$')
model_name_list.append(r'Ours-ind')
model_name_list.append(r'Ours-ind')

# write excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)

# refine excel
refine_excel(save_excel_file_path, is_paper=True)
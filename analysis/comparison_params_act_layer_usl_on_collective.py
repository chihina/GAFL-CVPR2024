import json
import os
import sys
import pandas as pd
import numpy as np
import pickle
sys.path.append(".")
from analysis_utils import write_excel, refine_excel

sys.path.append('../DIN-Group-Activity-Recognition-Benchmark')
from infer_model import GroupRelation_volleyball
from analysis_utils import refine_config, count_parameters

saved_result_dir = os.path.join('result')
saved_result_analysis_dir = os.path.join('result_analysis')

analyze_name_list = []

analyze_name_list.append('[CAD GR ours recon rand mask 6 act 0_stage2]<2023-11-04_16-11-00>')
analyze_name_list.append('[CAD GR ours recon rand mask 6 act 1_stage2]<2023-11-04_16-12-11>')
analyze_name_list.append('[CAD GA ours recon rand mask 6_stage2]<2023-10-26_16-18-47>')
analyze_name_list.append('[CAD GR ours recon rand mask 6 act 3_stage2]<2023-11-04_16-12-45>')
analyze_name_list.append('[CAD GR ours recon rand mask 6 act 4_stage2]<2023-11-04_16-13-18>')

# define model names
model_name_list = []
model_name_list.append(r'Ours-ind act 0')
model_name_list.append(r'Ours-ind act 1')
model_name_list.append(r'Ours-ind act 2')
model_name_list.append(r'Ours-ind act 3')
model_name_list.append(r'Ours-ind act 4')

# write excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)

# refine excel
refine_excel(save_excel_file_path, is_paper=True)
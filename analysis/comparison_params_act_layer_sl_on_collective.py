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

analyze_name_list.append('[CAD GA ours rand mask 0 act 0_stage2]<2023-10-25_17-03-23>')
analyze_name_list.append('[CAD GA ours rand mask 0 act 1_stage2]<2023-10-25_17-03-38>')
analyze_name_list.append('[CAD GA ours rand mask 0_stage2]<2023-10-23_13-28-02>')
analyze_name_list.append('[CAD GA ours rand mask 0 act 3_stage2]<2023-10-25_17-05-38>')
analyze_name_list.append('[CAD GA ours rand mask 0 act 4_stage2]<2023-11-05_20-47-21>')

# define model names
model_name_list = []
model_name_list.append(r'Ours-grp act 0')
model_name_list.append(r'Ours-grp act 1')
model_name_list.append(r'Ours-grp act 2')
model_name_list.append(r'Ours-grp act 3')
model_name_list.append(r'Ours-grp act 4')

# write excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)

# refine excel
refine_excel(save_excel_file_path, is_paper=True)
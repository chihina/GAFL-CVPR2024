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
analyze_name_list.append('[GR ours rand mask 5 grf 16_stage2]<2023-10-28_22-42-28>')
analyze_name_list.append('[GR ours rand mask 5 grf 32_stage2]<2023-10-28_22-41-53>')
analyze_name_list.append('[GR ours rand mask 5 grf 64_stage2]<2023-10-28_16-31-38>')
analyze_name_list.append('[GR ours rand mask 5 grf 128_stage2]<2023-10-28_22-41-26>')
analyze_name_list.append('[GR ours rand mask 5 grf 256_stage2]<2023-10-28_16-36-52>')
analyze_name_list.append('[GR ours rand mask 5 grf 512_stage2]<2023-10-28_16-39-25>')
analyze_name_list.append('[GR ours rand mask 5_stage2]<2023-10-16_22-26-54>')

# define model names
model_name_list = []
model_name_list.append(r'Ours gr dim 16')
model_name_list.append(r'Ours gr dim 32')
model_name_list.append(r'Ours gr dim 64')
model_name_list.append(r'Ours gr dim 128')
model_name_list.append(r'Ours gr dim 256')
model_name_list.append(r'Ours gr dim 512')
model_name_list.append(r'Ours gr dim 1024')

# write excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)

# refine excel
refine_excel(save_excel_file_path, is_paper=True)
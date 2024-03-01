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
analyze_name_list.append('[GR ours rand mask 5_stage2]<2023-10-16_22-26-54>')
analyze_name_list.append('[GR ours rand mask 5 head 2 layer 1_stage2]<2023-10-22_10-55-21>')
analyze_name_list.append('[GR ours rand mask 5 head 4 layer 1_stage2]<2023-10-22_10-55-53>')
analyze_name_list.append('[GR ours rand mask 5 head 8 layer 1_stage2]<2023-10-22_10-59-19>')
analyze_name_list.append('[GR ours rand mask 5 head 16 layer 1_stage2]<2023-10-22_10-57-31>')
analyze_name_list.append('[GR ours rand mask 5 head 32 layer 1_stage2]<2023-10-25_08-01-14>')
analyze_name_list.append('[GR ours rand mask 5 head 64 layer 1_stage2]<2023-10-25_08-01-24>')

# define model names
model_name_list = []
model_name_list.append(r'Ours trans head 1')
model_name_list.append(r'Ours trans head 2')
model_name_list.append(r'Ours trans head 4')
model_name_list.append(r'Ours trans head 8')
model_name_list.append(r'Ours trans head 16')
model_name_list.append(r'Ours trans head 32')
model_name_list.append(r'Ours trans head 64')

# write excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)

# refine excel
refine_excel(save_excel_file_path, is_paper=True)
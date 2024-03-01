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
analyze_name_list.append('[GR ours recon feat random mask 6_stage2]<2023-10-18_09-34-15>')
analyze_name_list.append('[GR ours recon feat random mask 6 head 1 layer 2_stage2]<2023-10-20_22-22-52>')
analyze_name_list.append('[GR ours recon feat random mask 6 head 1 layer 3_stage2]<2023-10-20_22-23-23>')
analyze_name_list.append('[GR ours recon feat random mask 6 head 1 layer 4_stage2]<2023-10-20_22-24-20>')

# define model names
model_name_list = []
model_name_list.append(r'Ours trans layer 1')
model_name_list.append(r'Ours trans layer 2')
model_name_list.append(r'Ours trans layer 3')
model_name_list.append(r'Ours trans layer 4')

# write excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)

# refine excel
refine_excel(save_excel_file_path, is_paper=True)
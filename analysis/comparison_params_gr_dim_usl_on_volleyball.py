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

# analyze_name_list.append('[GR ours recon feat random mask 6 grf 64_stage2]<2023-10-29_20-29-25>')
# analyze_name_list.append('[GR ours recon feat random mask 6 grf 256_stage2]<2023-10-29_20-29-08>')
# analyze_name_list.append('[GR ours recon feat random mask 6_stage2]<2023-10-18_09-34-15>')
# analyze_name_list.append('[GR ours recon feat random mask 6 grf 4224_stage2]<2023-10-29_20-28-22>')

analyze_name_list.append('[GR ours recon feat random mask 0 grf 2112_stage2]<2023-10-30_09-37-38>')
analyze_name_list.append('[GR ours recon feat random mask 1 grf 2112_stage2]<2023-10-31_08-35-55>')
analyze_name_list.append('[GR ours recon feat random mask 2 grf 2112_stage2]<2023-10-30_09-37-53>')
analyze_name_list.append('[GR ours recon feat random mask 3 grf 2112_stage2]<2023-10-30_22-43-04>')
analyze_name_list.append('[GR ours recon feat random mask 4 grf 2112_stage2]<2023-10-30_09-38-05>')
analyze_name_list.append('[GR ours recon feat random mask 5 grf 2112_stage2]<2023-10-30_22-42-48>')
analyze_name_list.append('[GR ours recon feat random mask 6 grf 2112_stage2]<2023-10-29_20-26-17>')
analyze_name_list.append('[GR ours recon feat random mask 7 grf 2112_stage2]<2023-10-30_22-42-33>')
analyze_name_list.append('[GR ours recon feat random mask 8 grf 2112_stage2]<2023-10-30_09-38-19>')
analyze_name_list.append('[GR ours recon feat random mask 9 grf 2112_stage2]<2023-10-30_22-42-17>')
analyze_name_list.append('[GR ours recon feat random mask 10 grf 2112_stage2]<2023-10-31_08-36-11>')
analyze_name_list.append('[GR ours recon feat random mask 11 grf 2112_stage2]<2023-10-31_08-36-21>')

# define model names
model_name_list = []
# model_name_list.append(r'Ours gr dim 64')
# model_name_list.append(r'Ours gr dim 256')
# model_name_list.append(r'Ours gr dim 1024')
# model_name_list.append(r'Ours gr dim 4224')
model_name_list.append(r'Ours gr dim 2112 (0)')
model_name_list.append(r'Ours gr dim 2112 (1)')
model_name_list.append(r'Ours gr dim 2112 (2)')
model_name_list.append(r'Ours gr dim 2112 (3)')
model_name_list.append(r'Ours gr dim 2112 (4)')
model_name_list.append(r'Ours gr dim 2112 (5)')
model_name_list.append(r'Ours gr dim 2112 (6)')
model_name_list.append(r'Ours gr dim 2112 (7)')
model_name_list.append(r'Ours gr dim 2112 (8)')
model_name_list.append(r'Ours gr dim 2112 (9)')
model_name_list.append(r'Ours gr dim 2112 (10)')
model_name_list.append(r'Ours gr dim 2112 (11)')

# write excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)

# refine excel
refine_excel(save_excel_file_path, is_paper=True)
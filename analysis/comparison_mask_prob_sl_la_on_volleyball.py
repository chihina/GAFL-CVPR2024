import json
import os
import sys
import glob
import pandas as pd
import numpy as np
sys.path.append(".")
from analysis_utils import write_excel, refine_excel

saved_result_dir = os.path.join('result')
saved_result_analysis_dir = os.path.join('result_analysis')

analyze_name_list = []
# analyze_name_list.append('[GR ours_stage2]<2023-07-08_08-59-54>')
# analyze_name_list.append('[GR ours rand mask 1_stage2]<2023-10-16_22-26-23>')
# analyze_name_list.append('[GR ours rand mask 2_stage2]<2023-10-17_08-59-56>')
# analyze_name_list.append('[GR ours rand mask 3_stage2]<2023-10-16_22-26-41>')
# analyze_name_list.append('[GR ours rand mask 4_stage2]<2023-10-17_09-00-12>')
# analyze_name_list.append('[GR ours rand mask 5_stage2]<2023-10-16_22-26-54>')
# analyze_name_list.append('[GR ours rand mask 6_stage2]<2023-10-17_09-00-23>')
# analyze_name_list.append('[GR ours rand mask 7_stage2]<2023-10-16_22-27-14>')
# analyze_name_list.append('[GR ours rand mask 8_stage2]<2023-10-17_09-01-33>')
# analyze_name_list.append('[GR ours rand mask 9_stage2]<2023-10-16_22-27-28>')
# analyze_name_list.append('[GR ours rand mask 10_stage2]<2023-10-17_09-01-56>')
# analyze_name_list.append('[GR ours rand mask 11_stage2]<2023-10-16_22-27-42>')

# define model names
model_name_list = []
# model_name_list.append(r'Ours w/o MPM')
# model_name_list.append(r'Ours w/ MPM ($det=1$)')
# model_name_list.append(r'Ours w/ MPM ($det=2$)')
# model_name_list.append(r'Ours w/ MPM ($det=3$)')
# model_name_list.append(r'Ours w/ MPM ($det=4$)')
# model_name_list.append(r'Ours w/ MPM ($det=5$)')
# model_name_list.append(r'Ours w/ MPM ($det=6$)')
# model_name_list.append(r'Ours w/ MPM ($det=7$)')
# model_name_list.append(r'Ours w/ MPM ($det=8$)')
# model_name_list.append(r'Ours w/ MPM ($det=9$)')
# model_name_list.append(r'Ours w/ MPM ($det=10$)')
# model_name_list.append(r'Ours w/ MPM ($det=11$)')

search_key = f'[GR ours rand mask*w temp cond_stage2*'
search_name = os.path.join(saved_result_dir, search_key)
analyze_name_list_glob = [os.path.basename(x) for x in glob.glob(search_name)]
analyze_name_list_glob_dic = {}
model_name_list_glob_dic = {}

for analyze_name in analyze_name_list_glob:
    mask_num = int(analyze_name.split('_')[0].split(' ')[4])
    analyze_name_list_glob_dic[mask_num] = analyze_name
    model_name_list_glob_dic[mask_num] = rf'Ours w/ MPM ($det={mask_num}$)'

analyze_name_list_glob_dic = sorted(analyze_name_list_glob_dic.items(), key=lambda i: i[0])
analyze_name_list_glob = [x[1] for x in analyze_name_list_glob_dic]
analyze_name_list = analyze_name_list + analyze_name_list_glob

model_name_list_glob_dic = sorted(model_name_list_glob_dic.items(), key=lambda i: i[0])
model_name_list = [x[1] for x in model_name_list_glob_dic]

# write excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)

# refine excel
refine_excel(save_excel_file_path, is_paper=True)
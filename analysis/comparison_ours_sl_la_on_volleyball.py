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
analyze_name_list.append('[GR ours_stage2]<2023-07-08_08-59-54>')

# analyze_name_list.append('[GR ours rand mask 2_stage2]<2023-07-26_10-41-00>')
# analyze_name_list.append('[GR ours rand mask 4_stage2]<2023-07-21_08-54-41>')
# analyze_name_list.append('[GR ours rand mask 6_stage2]<2023-07-20_16-00-31>')
# analyze_name_list.append('[GR ours rand mask 8_stage2]<2023-07-21_08-54-57>')
# analyze_name_list.append('[GR ours rand mask 10_stage2]<2023-07-26_10-42-35>')
analyze_name_list.append('[GR ours active mask inv_stage2]<2023-08-25_08-29-32>')
analyze_name_list.append('[GR ours active mask inv head32_stage2]<2023-09-14_14-25-05>')
analyze_name_list.append('[GR ours active mask inv head32 layer2_stage2]<2023-09-14_14-25-28>')

# analyze_name_list.append('[GR ours gen iar ratio 0_stage2]<2023-09-11_21-23-24>')
# analyze_name_list.append('[GR ours gen iar ratio 0.1_stage2]<2023-09-11_21-23-41>')
# analyze_name_list.append('[GR ours gen iar ratio 0.2_stage2]<2023-09-11_21-24-02>')
# analyze_name_list.append('[GR ours gen iar ratio 0.3_stage2]<2023-09-12_21-27-31>')
# analyze_name_list.append('[GR ours gen iar ratio 0.4_stage2]<2023-09-12_21-28-34>')
# analyze_name_list.append('[GR ours gen iar ratio 0.5_stage2]<2023-08-31_08-42-23>')
# analyze_name_list.append('[GR ours gen iar ratio 0.6_stage2]<2023-09-12_21-28-53>')

# analyze_name_list.append('[GR ours head2 layer1_stage2]<2023-07-14_07-51-07>')
# analyze_name_list.append('[GR ours head4 layer1_stage2]<2023-07-14_07-56-17>')
# analyze_name_list.append('[GR ours head8 layer1_stage2]<2023-07-14_07-56-55>')
# analyze_name_list.append('[GR ours head16 layer1_stage2]<2023-07-18_08-34-06>')
# analyze_name_list.append('[GR ours head32 layer1_stage2]<2023-07-18_08-34-42>')
# analyze_name_list.append('[GR ours head64 layer1_stage2]<2023-07-19_13-19-04>')
# analyze_name_list.append('[GR ours head128 layer1_stage2]<2023-07-20_15-51-19>')

# define model names
model_name_list = []
model_name_list.append(r'Ours')

# model_name_list.append(r'Ours rand mask 2')
# model_name_list.append(r'Ours rand mask 4')
# model_name_list.append(r'Ours rand mask 6')
# model_name_list.append(r'Ours rand mask 8')
# model_name_list.append(r'Ours rand mask 10')
model_name_list.append(r'Ours active mask head1 layer1')
model_name_list.append(r'Ours active mask head32 layer1')
model_name_list.append(r'Ours active mask head32 layer2')

# model_name_list.append(r'Ours gen iar ratio 0')
# model_name_list.append(r'Ours gen iar ratio 0.1')
# model_name_list.append(r'Ours gen iar ratio 0.2')
# model_name_list.append(r'Ours gen iar ratio 0.3')
# model_name_list.append(r'Ours gen iar ratio 0.4')
# model_name_list.append(r'Ours gen iar ratio 0.5')
# model_name_list.append(r'Ours gen iar ratio 0.6')

# model_name_list.append(r'Ours head 2')
# model_name_list.append(r'Ours head 4')
# model_name_list.append(r'Ours head 8')
# model_name_list.append(r'Ours head 16')
# model_name_list.append(r'Ours head 32')
# model_name_list.append(r'Ours head 64')
# model_name_list.append(r'Ours head 128')

# write excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path)

# refine excel
refine_excel(save_excel_file_path)
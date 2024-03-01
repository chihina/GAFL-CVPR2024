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
# analyze_name_list.append('[GR ours recon feat random mask 0_stage2]<2023-10-18_22-30-29>')
# analyze_name_list.append('[GR ours recon feat random mask 1_stage2]<2023-10-18_09-33-29>')
# analyze_name_list.append('[GR ours recon feat random mask 2_stage2]<2023-10-18_09-33-42>')
# analyze_name_list.append('[GR ours recon feat random mask 2_stage2]<2023-10-25_22-25-39>')
# analyze_name_list.append('[GR ours recon feat random mask 3_stage2]<2023-10-18_09-37-34>')
# analyze_name_list.append('[GR ours recon feat random mask 4_stage2]<2023-10-18_09-33-58>')
# analyze_name_list.append('[GR ours recon feat random mask 4_stage2]<2023-10-25_22-25-53>')
# analyze_name_list.append('[GR ours recon feat random mask 5_stage2]<2023-10-18_09-38-15>')
# analyze_name_list.append('[GR ours recon feat random mask 6_stage2]<2023-10-18_09-34-15>')
# analyze_name_list.append('[GR ours recon feat random mask 6_stage2]<2023-10-25_22-26-38>')
# analyze_name_list.append('[GR ours recon feat random mask 7_stage2]<2023-10-18_09-39-32>')
# analyze_name_list.append('[GR ours recon feat random mask 8_stage2]<2023-10-18_09-34-25>')
# analyze_name_list.append('[GR ours recon feat random mask 8_stage2]<2023-10-25_22-29-01>')
# analyze_name_list.append('[GR ours recon feat random mask 9_stage2]<2023-10-18_09-39-58>')
# analyze_name_list.append('[GR ours recon feat random mask 10_stage2]<2023-10-18_09-34-45>')
# analyze_name_list.append('[GR ours recon feat random mask 10_stage2]<2023-10-25_22-29-17>')
# analyze_name_list.append('[GR ours recon feat random mask 11_stage2]<2023-10-18_09-35-16>')
# analyze_name_list.append('[GR ours recon feat random mask 11_stage2]<2023-10-25_22-31-45>')

# analyze_name_list.append('[GR ours recon feat mean random mask 0_stage2]<2023-11-02_07-54-29>')
# analyze_name_list.append('[GR ours recon feat mean random mask 1_stage2]<2023-11-03_08-12-55>')
# analyze_name_list.append('[GR ours recon feat mean random mask 2_stage2]<2023-11-02_07-54-37>')
# analyze_name_list.append('[GR ours recon feat mean random mask 3_stage2]<2023-11-03_08-13-20>')
# analyze_name_list.append('[GR ours recon feat mean random mask 4_stage2]<2023-11-02_07-54-50>')
# analyze_name_list.append('[GR ours recon feat mean random mask 5_stage2]<2023-11-03_08-13-53>')
# analyze_name_list.append('[GR ours recon feat mean random mask 6_stage2]<2023-11-01_08-36-06>')
# analyze_name_list.append('[GR ours recon feat mean random mask 7_stage2]<2023-11-03_08-14-17>')
# analyze_name_list.append('[GR ours recon feat mean random mask 8_stage2]<2023-11-02_07-55-11>')
# analyze_name_list.append('[GR ours recon feat mean random mask 9_stage2]<2023-11-03_08-15-04>')
# analyze_name_list.append('[GR ours recon feat mean random mask 10_stage2]<2023-11-02_07-55-47>')
# analyze_name_list.append('[GR ours recon feat mean random mask 11_stage2]<2023-11-03_08-15-14>')

analyze_name_list.append('[GR ours recon feat random mask 0 w temp cond_stage2]<2023-11-10_13-26-06>')
analyze_name_list.append('[GR ours recon feat random mask 1 w temp cond_stage2]<2023-11-10_13-26-58>')
analyze_name_list.append('[GR ours recon feat random mask 2 w temp cond_stage2]<2023-11-10_13-27-11>')
analyze_name_list.append('[GR ours recon feat random mask 3 w temp cond_stage2]<2023-11-10_13-27-26>')
analyze_name_list.append('[GR ours recon feat random mask 4 w temp cond_stage2]<2023-11-10_13-28-16>')
analyze_name_list.append('[GR ours recon feat random mask 5 w temp cond_stage2]<2023-11-10_13-28-37>')
analyze_name_list.append('[GR ours recon feat random mask 6 w temp cond_stage2]<2023-11-10_13-29-45>')
analyze_name_list.append('[GR ours recon feat random mask 7 w temp cond_stage2]<2023-11-10_13-30-09>')
analyze_name_list.append('[GR ours recon feat random mask 8 w temp cond_stage2]<2023-11-10_13-32-14>')
analyze_name_list.append('[GR ours recon feat random mask 9 w temp cond_stage2]<2023-11-10_13-32-30>')
analyze_name_list.append('[GR ours recon feat random mask 10 w temp cond_stage2]<2023-11-10_13-33-03>')
analyze_name_list.append('[GR ours recon feat random mask 11 w temp cond_stage2]<2023-11-10_13-33-09>')


# define model names
model_name_list = []
# model_name_list.append(r'Ours w/o MPM')
model_name_list.append(r'Ours w/o MPM')
model_name_list.append(r'Ours w/ MPM ($det=1$)')
model_name_list.append(r'Ours w/ MPM ($det=2$)')
model_name_list.append(r'Ours w/ MPM ($det=3$)')
model_name_list.append(r'Ours w/ MPM ($det=4$)')
model_name_list.append(r'Ours w/ MPM ($det=5$)')
model_name_list.append(r'Ours w/ MPM ($det=6$)')
model_name_list.append(r'Ours w/ MPM ($det=7$)')
model_name_list.append(r'Ours w/ MPM ($det=8$)')
model_name_list.append(r'Ours w/ MPM ($det=9$)')
model_name_list.append(r'Ours w/ MPM ($det=10$)')
model_name_list.append(r'Ours w/ MPM ($det=11$)')
# model_name_list.append(r'Ours w/ MPM ($det=all$)')

# write excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)

# refine excel
refine_excel(save_excel_file_path, is_paper = True)
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
# analyze_name_list.append('[CAD GA ours rand mask 0_stage2]<2023-10-23_13-28-02>')
# analyze_name_list.append('[CAD GA ours rand mask 1_stage2]<2023-10-26_14-19-55>')
# analyze_name_list.append('[CAD GA ours rand mask 2_stage2]<2023-10-26_14-19-35>')
# analyze_name_list.append('[CAD GA ours rand mask 3_stage2]<2023-10-26_14-19-26>')
# analyze_name_list.append('[CAD GA ours rand mask 4_stage2]<2023-10-26_14-18-55>')
# analyze_name_list.append('[CAD GA ours rand mask 6_stage2]<2023-10-26_14-17-57>')
# analyze_name_list.append('[CAD GA ours rand mask 5_stage2]<2023-10-26_14-18-44>')
# analyze_name_list.append('[CAD GA ours rand mask 7_stage2]<2023-10-26_10-46-23>')
# analyze_name_list.append('[CAD GA ours rand mask 8_stage2]<2023-10-26_10-46-12>')
# analyze_name_list.append('[CAD GA ours rand mask 9_stage2]<2023-10-26_10-45-58>')
# analyze_name_list.append('[CAD GA ours rand mask 10_stage2]<2023-10-26_10-45-00>')
# analyze_name_list.append('[CAD GA ours rand mask 11_stage2]<2023-10-26_10-44-47>')
# analyze_name_list.append('[CAD GA ours rand mask 12_stage2]<2023-10-26_10-43-46>')

analyze_name_list.append('[CAD GR ours rand mask 0 w temp cond_stage2]<2023-11-11_22-13-53>')
analyze_name_list.append('[CAD GR ours rand mask 1 w temp cond_stage2]<2023-11-11_22-15-09>')
analyze_name_list.append('[CAD GR ours rand mask 2 w temp cond_stage2]<2023-11-11_22-15-20>')
analyze_name_list.append('[CAD GR ours rand mask 3 w temp cond_stage2]<2023-11-11_22-15-31>')
analyze_name_list.append('[CAD GR ours rand mask 4 w temp cond_stage2]<2023-11-11_22-15-47>')
analyze_name_list.append('[CAD GR ours rand mask 5 w temp cond_stage2]<2023-11-11_22-16-06>')
analyze_name_list.append('[CAD GR ours rand mask 6 w temp cond_stage2]<2023-11-12_09-32-36>')
analyze_name_list.append('[CAD GR ours rand mask 7 w temp cond_stage2]<2023-11-12_09-32-48>')
analyze_name_list.append('[CAD GR ours rand mask 8 w temp cond_stage2]<2023-11-12_09-32-58>')
analyze_name_list.append('[CAD GR ours rand mask 9 w temp cond_stage2]<2023-11-12_09-33-18>')
analyze_name_list.append('[CAD GR ours rand mask 10 w temp cond_stage2]<2023-11-12_09-33-31>')
analyze_name_list.append('[CAD GR ours rand mask 11 w temp cond_stage2]<2023-11-12_09-33-44>')
analyze_name_list.append('[CAD GR ours rand mask 12 w temp cond_stage2]<2023-11-12_09-33-59>')

# define model names
model_name_list = []
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
model_name_list.append(r'Ours w/ MPM ($det=12$)')

# write excel
save_excel_file_path = os.path.join(saved_result_analysis_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}.xlsx')
write_excel(analyze_name_list, model_name_list, saved_result_dir, save_excel_file_path, is_paper=True)

# refine excel
refine_excel(save_excel_file_path, is_paper=True)
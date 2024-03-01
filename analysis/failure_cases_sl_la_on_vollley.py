import json
import os
import shutil
import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(".")



ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
              'l_set', 'l-spike', 'l-pass', 'l_winpoint']

ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
           'moving', 'setting', 'spiking', 'standing',
           'waiting']

saved_result_dir = os.path.join('result')
saved_result_analysis_dir = os.path.join('result_analysis')
analyze_name = '[GR ours rand mask 5_stage2]<2023-10-16_22-26-54>'

# make a directory to save analized results
save_dir = os.path.join(saved_result_analysis_dir, 'failure_cases_sl_la_on_vollley')
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)

# read excel
read_excel_file_path = os.path.join(saved_result_dir, analyze_name, f'nn_video_scene.xlsx')
df = pd.read_excel(read_excel_file_path, index_col=0)
df_failures = df[df['fl_ga'] == False]
df_failures = df_failures.sample(frac=1)

# save failure cases
for failure_idx in range(5):
    df_failures_sampled = df_failures.iloc[failure_idx]
    gt_ga = int(df_failures_sampled['gt_ga'])
    nn_ga = int(df_failures_sampled['nn_ga'])

    save_dir_failure = os.path.join(save_dir, df_failures_sampled.name)
    if not os.path.exists(save_dir_failure):
        os.makedirs(save_dir_failure)

    vid_id, seq_id, img_id = df_failures_sampled.name.split('_')
    img_path = os.path.join('data', 'volleyball', 'videos', f'{vid_id}', f'{seq_id}', f'{seq_id}.jpg')
    shutil.copy(img_path, os.path.join(save_dir_failure, f'{df_failures_sampled.name}_{ACTIVITIES[gt_ga]}.jpg'))

    nn_id = df_failures_sampled['nn_id']
    vid_id_nn, seq_id_nn, img_id_nn = nn_id.split('_')
    img_path_nn = os.path.join('data', 'volleyball', 'videos', f'{vid_id_nn}', f'{seq_id_nn}', f'{seq_id_nn}.jpg')
    shutil.copy(img_path_nn, os.path.join(save_dir_failure, f'{nn_id}_{ACTIVITIES[nn_ga]}.jpg'))
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

saved_result_dir = os.path.join('result')
saved_result_analysis_dir = os.path.join('result_analysis')
analyze_name = '[GR ours rand mask 5_stage2]<2023-10-16_22-26-54>'

# read excel
read_excel_file_path = os.path.join(saved_result_dir, analyze_name, f'test_kmeans_scene.xlsx')
df = pd.read_excel(read_excel_file_path, index_col=0)

# make a directory to save analized results
save_dir = os.path.join(saved_result_analysis_dir, 'kmeans_sl_la_on_volley')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_dir_cluster = os.path.join(save_dir, 'cluster')
if not os.path.exists(save_dir_cluster):
    os.makedirs(save_dir_cluster)
save_dir_ga = os.path.join(save_dir, 'ga')
if not os.path.exists(save_dir_ga):
    os.makedirs(save_dir_ga)

# visualize group activity classes
for ga_idx, ga_class in enumerate(ACTIVITIES):
    print(f'ga_class: {ga_class}')
    plt.figure()
    plt.hist(df[df['GA'] == ga_idx]['cluster_id'], align='mid')
    plt.savefig(os.path.join(save_dir_ga, f'ga_id_{ga_class}.png'))

# visualize based on clustering results
sampling_num = 5
cluster_id_max = df['cluster_id'].max()
for cluster_id in range(cluster_id_max+1):
    print(f'cluster_id: {cluster_id}')
    plt.figure()
    plt.hist(df[df['cluster_id'] == cluster_id]['GA'], align='mid')
    plt.savefig(os.path.join(save_dir_cluster, f'cluster_id_{cluster_id}.png'))

    save_dir_cluster_child = os.path.join(save_dir_cluster, f'{cluster_id}')
    if os.path.exists(save_dir_cluster_child):
        shutil.rmtree(save_dir_cluster_child)
    os.makedirs(save_dir_cluster_child)

    data_id_list = df[df['cluster_id'] == cluster_id].index.tolist()
    data_id_list_sampled = random.sample(data_id_list, sampling_num)
    for data_id_sampled in data_id_list_sampled:
        vid_id, seq_id, img_id = data_id_sampled.split('_')
        img_path = os.path.join('data', 'volleyball', 'videos', f'{vid_id}', f'{seq_id}', f'{seq_id}.jpg')
        shutil.copy(img_path, os.path.join(save_dir_cluster_child, f'{data_id_sampled}.jpg'))
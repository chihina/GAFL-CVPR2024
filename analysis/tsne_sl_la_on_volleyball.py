import json
import os
import shutil
import sys
import pandas as pd
import numpy as np
sys.path.append(".")


ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
              'l_set', 'l-spike', 'l-pass', 'l_winpoint']

saved_result_dir = os.path.join('result')
saved_result_analysis_dir = os.path.join('result_analysis')
analyze_name = '[GR ours rand mask 5_stage2]<2023-10-16_22-26-54>'

# read excel
read_excel_file_path = os.path.join(saved_result_dir, analyze_name, f'test_tsne_scene.xlsx')
df = pd.read_excel(read_excel_file_path, index_col=0)

# make a directory to save analized results
save_dir = os.path.join(saved_result_analysis_dir, 'tsne_sl_la_on_volley')

# set range directories
range_dic_all = {}

# l-spike
ga_class = 'l-spike'
range_dic = {}
range_dic[f'{ga_class}_grp1'] = {'range':[25, 30, 5, 10]}
range_dic[f'{ga_class}_grp2'] = {'range':[40, 50, 15, 20]}
range_dic_all[ga_class] = range_dic

# l-pass
ga_class = 'l-pass'
range_dic = {}
range_dic[f'{ga_class}_grp1'] = {'range':[-30, -20, 15, 20]}
range_dic[f'{ga_class}_grp2'] = {'range':[5, 10, 10, 15]}
range_dic_all[ga_class] = range_dic

# r-pass
ga_class = 'r-pass'
range_dic = {}
range_dic[f'{ga_class}_grp1'] = {'range':[-40, -35, -15, -10]}
range_dic[f'{ga_class}_grp2'] = {'range':[-5, 5, -15, -10]}
range_dic_all[ga_class] = range_dic

# l-winpoint
ga_class = 'l_winpoint'
range_dic = {}
range_dic[f'{ga_class}_grp1'] = {'range':[-40, -35, -5, 5]}
range_dic[f'{ga_class}_grp2'] = {'range':[-15, -5, 5, 10]}
range_dic_all[ga_class] = range_dic

# search result files
for ga_class, range_dic in range_dic_all.items():
    for grp_name, grp_dic in range_dic.items():
        x_min, x_max, y_min, y_max = map(int, grp_dic['range'])
        df_x_indice = (df['dim1'] > x_min) & (df['dim1'] < x_max)
        df_y_indice = (df['dim2'] > y_min) & (df['dim2'] < y_max)
        df_select = df[df_x_indice & df_y_indice]
        print('Before GA filtering:', len(df_select))
        df_select = df_select[df_select['GA'] == ACTIVITIES.index(ga_class)]
        range_dic[grp_name]['ids'] = df_select.index
        print(grp_name, len(df_select.index))

for ga_class, range_dic in range_dic_all.items():
    for grp_name, grp_dic in range_dic.items():
        data_ids = grp_dic['ids']
        for data_id in data_ids:
            vid_num, seq_num, img_num = data_id.split('_')
            save_dir_child = os.path.join(saved_result_analysis_dir, 'tsne_sl_la_on_volley', grp_name, f'{vid_num}_{seq_num}')
            if os.path.exists(save_dir_child):
                shutil.rmtree(save_dir_child)
            os.makedirs(save_dir_child)

            vid_id, seq_id, img_id = data_id.split('_')
            for pad_ran in range(-20, 20, 5):
                img_id_pad = int(seq_id) + pad_ran
                img_path = os.path.join('data', 'volleyball', 'videos', vid_id, seq_id, f'{img_id_pad}.jpg')
                img_path_save = os.path.join(save_dir_child, f'{img_id_pad}.jpg')
                shutil.copy(img_path, img_path_save)
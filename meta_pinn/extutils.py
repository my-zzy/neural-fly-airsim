
import os, re
from typing import List, Dict
from ast import literal_eval
from collections import namedtuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

folder = './data/experiment'
filename_fields = ['vehicle', 'trajectory', 'method', 'condition']

def save_data(Data : List[dict], folder : str, fields=['t', 'p', 'p_d', 'v', 'v_d', 'q', 'R', 'w', 'T_sp', 'q_sp', 'hover_throttle', 'fa', 'pwm']):
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print('Created data folder ' + folder)
    for data in Data:
        if 'fa' in fields and 'fa' not in data:
            data['fa'] = data['fa_num_Tsp']

        import pandas as pd
        df = pd.DataFrame()
        missing_fields = []
        for field in fields:
            try:
                df[field] = data[field].tolist()
            except KeyError as err:
                missing_fields.append(field)
        if len(missing_fields) > 0:
            print('missing fields ', ', '.join(missing_fields))

        filename = '_'.join(data[field] for field in filename_fields)
        df.to_csv(f"{folder}/{filename}.csv")

def load_data(folder : str, expnames = None) -> List[dict]:
    Data = []
    if expnames is None:
        filenames = sorted(os.listdir(folder))
    elif isinstance(expnames, str):
        filenames = []
        for filename in sorted(os.listdir(folder)):
            if re.search(expnames, filename) is not None:
                filenames.append(filename)
    elif isinstance(expnames, list):
        filenames = [expname + '.csv' for expname in expnames]
    else:
        raise NotImplementedError()

    task_idx = 0
    for filename in filenames:
        if not filename.endswith('.csv'):
            continue
        import pandas as pd
        df = pd.read_csv(os.path.join(folder, filename))

        for field in df.columns[1:]:
            if isinstance(df[field][0], str):
                df[field] = df[field].apply(literal_eval)

        Data.append({})
        for field in df.columns[1:]:
            Data[-1][field] = np.array(df[field].tolist(), dtype=float)

        namesplit = filename.split('.')[0].split('_')
        for i, field in enumerate(filename_fields):
            if i < len(namesplit):
                Data[-1][field] = namesplit[i]

        Data[-1]['filename'] = filename
        Data[-1]['filepath'] = os.path.join(folder, filename)

        print(f"[Task {task_idx}] ← {filename}")
        task_idx += 1
    return Data

SubDataset = namedtuple('SubDataset', 'X Y C meta')
feature_len: Dict[str, int] = {}

def format_data(RawData: List[Dict['str', np.ndarray]], features: 'list[str]' = ['v', 'q', 'pwm'], output: str = 'fa', hover_pwm_ratio = 1.):
    Data = []
    for i, data in enumerate(RawData):
        X = []
        for feature in features:
            if feature == 'pwm':
                X.append(data[feature] / 1000 * hover_pwm_ratio)
            else:
                X.append(data[feature])
            feature_len[feature] = len(data[feature][0])
        X = np.hstack(X)
        Y = data[output]
        C = i
        Data.append(SubDataset(
            X, Y, C,
            {
                'method': data.get('method', ''),
                'condition': data.get('condition', ''),
                't': data['t'],
                'filename': data.get('filename', ''),
                'filepath': data.get('filepath', '')
            }
        ))
    return Data

def plot_subdataset(data, features, title_prefix=''):
    fig, axs = plt.subplots(1, len(features)+1, figsize=(10,4))
    idx = 0
    for feature, ax in zip(features, axs):
        for j in range(feature_len[feature]):
            ax.plot(data.meta['t'], data.X[:, idx], label = f"{feature}_{j}")
            idx += 1
        ax.legend(); ax.set_xlabel('time [s]')
    ax = axs[-1]
    ax.plot(data.meta['t'], data.Y); ax.legend(('fa_x', 'fa_y', 'fa_z'))
    ax.set_xlabel('time [s]')
    title = f"{title_prefix} {data.meta.get('condition','')}: c={data.C}"
    if data.meta.get('filename'): title += f"\n{data.meta['filename']}"
    fig.suptitle(title); fig.tight_layout()

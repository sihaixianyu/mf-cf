import os
import pickle

import numpy as np
import pandas as pd


def load_data(data_dir: str) -> (np.ndarray, np.ndarray, dict):
    pos_train_path = os.path.join(data_dir, 'pos_train.csv')
    pos_test_path = os.path.join(data_dir, 'pos_test.csv')
    neg_path = os.path.join(data_dir, 'neg.dict')
    info_path = os.path.join(data_dir, 'info.dict')

    check_file(pos_train_path, pos_test_path, neg_path)

    pos_train_arr = pd.read_csv(pos_train_path).to_numpy()
    pos_test_arr = pd.read_csv(pos_test_path).to_numpy()

    with open(neg_path, 'rb') as f:
        neg_dict = pickle.load(f)

    with open(info_path, 'rb') as f:
        info_dict = pickle.load(f)

    return pos_train_arr, pos_test_arr, neg_dict, info_dict


def check_file(*files: str):
    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError('{} not exist, please run process script!'.format(file))


def print_res(content: str, split='-', num=75):
    print(split * num)
    print(content)
    print(split * num)


if __name__ == '__main__':
    load_data('data/ml-1m/')

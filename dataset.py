import numpy as np
import scipy.sparse as sp
import torch.utils.data as data


class TrainDataset(data.Dataset):
    def __init__(self, pos_arr: np.ndarray, info_dict: dict, neg_num=4):
        self.user_num = info_dict['user_num']
        self.item_num = info_dict['item_num']

        self.train_mat = self.build_train_mat(pos_arr)
        self.train_arr = self.neg_sample(pos_arr, neg_num)

    def __len__(self):
        return len(self.train_arr)

    def __getitem__(self, idx):
        uid = self.train_arr[idx][0]
        pos_iid = self.train_arr[idx][1]
        neg_iid = self.train_arr[idx][2]

        return uid, pos_iid, neg_iid

    def neg_sample(self, pos_arr, neg_num):
        assert neg_num > 0, 'neg_num must be larger than 0'

        train_arr = []
        for arr in pos_arr:
            uid, pos_iid = arr[0], arr[1]
            for _ in range(neg_num):
                neg_iid = np.random.randint(self.item_num)
                while (uid, neg_iid) in self.train_mat:
                    neg_iid = np.random.randint(self.item_num)
                train_arr.append([uid, pos_iid, neg_iid])

        return train_arr

    def build_train_mat(self, pos_arr):
        train_mat = sp.dok_matrix((self.user_num, self.item_num), dtype=np.float32)
        for arr in pos_arr:
            train_mat[arr[0], arr[1]] = 1.0

        return train_mat


class TestDataset(data.Dataset):
    def __init__(self, pos_arr: np.ndarray, info_dict: dict, neg_dict: dict):
        self.user_num = info_dict['user_num']
        self.item_num = info_dict['item_num']

        self.test_arr = self.build_test_arr(pos_arr, neg_dict)

    def __len__(self):
        return len(self.test_arr)

    def __getitem__(self, idx):
        uid = self.test_arr[idx][0]
        iid = self.test_arr[idx][1]

        return uid, iid

    @staticmethod
    def build_test_arr(pos_arr, neg_dict):
        test_arr = []
        for arr in pos_arr:
            uid = arr[0]
            pos_iid = arr[1]
            test_arr.append([uid, pos_iid])
            for neg_iid in neg_dict[uid]:
                test_arr.append([uid, neg_iid])

        return test_arr

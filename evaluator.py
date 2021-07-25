import numpy as np
import torch
from torch.utils.data import DataLoader

from model import MF


class Evaluator:
    def __init__(self, loader: DataLoader, model: MF, topk=10):
        self.loader = loader
        self.model = model
        self.topk = topk

    def evaluate(self):
        self.model.eval()

        hit_list, ndcg_list = [], []
        for uids, iids in self.loader:
            uids = uids.to(self.model.device)
            iids = iids.to(self.model.device)

            preds = self.model.predict(uids, iids)
            _, idxs = torch.topk(preds, self.topk)
            rec_list = torch.take(iids, idxs).cpu().numpy().tolist()

            pos_item = iids[0].item()
            hit_list.append(self.hit(pos_item, rec_list))
            ndcg_list.append(self.ndcg(pos_item, rec_list))

        return np.mean(hit_list), np.mean(ndcg_list)

    @staticmethod
    def hit(iid, rec_list):
        if iid in rec_list:
            return 1
        return 0

    @staticmethod
    def ndcg(iid, rec_list):
        if iid in rec_list:
            idx = rec_list.index(iid)
            return np.reciprocal(np.log2(idx + 2))
        return 0

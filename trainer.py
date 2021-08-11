import numpy as np
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from model import BrpMF


class Trainer:
    def __init__(self, loader: DataLoader, model: BrpMF, optimizer: Optimizer):
        self.loader = loader
        self.model = model
        self.optimizer = optimizer

    def train(self) -> float:
        self.model.train()

        loss_list = []
        for uids, pos_iids, neg_iids in self.loader:
            self.optimizer.zero_grad()
            uids = uids.to(self.model.device)
            pos_iids = pos_iids.to(self.model.device)
            neg_iids = neg_iids.to(self.model.device)

            pos_preds, neg_preds = self.model.forward(uids, pos_iids, neg_iids)
            brp_loss = - (pos_preds - neg_preds).sigmoid().log().sum()
            brp_loss.backward()
            self.optimizer.step()

            loss_list.append(brp_loss.detach().cpu().numpy())

        return np.mean(loss_list).item()

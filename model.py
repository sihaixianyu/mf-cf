import torch
import torch.nn as nn

from torch import LongTensor


class BrpMF(nn.Module):
    def __init__(self, user_num: int, item_num: int, latent_dim: int, device='cpu'):
        super(BrpMF, self).__init__()
        self.embed_user = nn.Embedding(user_num, latent_dim)
        self.embed_item = nn.Embedding(item_num, latent_dim)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        self.device = device
        self.to(device)

    def forward(self, uids: LongTensor, pos_iids: LongTensor, neg_iids: LongTensor):
        user_vecs = self.embed_user(uids)
        pos_item_vecs = self.embed_item(pos_iids)
        neg_item_vecs = self.embed_item(neg_iids)

        pos_preds = (user_vecs * pos_item_vecs).sum(dim=1)
        neg_preds = (user_vecs * neg_item_vecs).sum(dim=1)

        return pos_preds, neg_preds

    def predict(self, uids: LongTensor, iids: LongTensor):
        with torch.no_grad():
            user_vecs = self.embed_user(uids)
            item_vecs = self.embed_item(iids)
            preds = (user_vecs * item_vecs).sum(dim=1)

        return preds

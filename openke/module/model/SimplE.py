import torch
import torch.nn as nn
from .Model import Model
import numpy as np

class SimplE(Model):

    def __init__(self, add_mol, entity_description_path, ent_tot, rel_tot, dim = 100):
        super(SimplE, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.rel_inv_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.add_mol = add_mol

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_inv_embeddings.weight.data)

        """ 把词向量矩阵作为参数复制过去 """
        self.ent_description = nn.Embedding(self.ent_tot, self.dim)
        entity_description = np.load(entity_description_path)
        self.ent_description.weight.data.copy_(torch.from_numpy(entity_description))

    def _calc_avg(self, h, t, r, r_inv):
        return (torch.sum(h * r * t, -1) + torch.sum(h * r_inv * t, -1))/2

    def _calc_ingr(self, h, r, t):
        return torch.sum(h * r * t, -1)

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        r_inv = self.rel_inv_embeddings(batch_r)
        score = self._calc_avg(h, t, r, r_inv)

        if self.add_mol:
            # 添加描述
            h_d = self.ent_description(batch_h)
            t_d = self.ent_description(batch_t)
            score1 = self._calc_avg(h_d, t_d, r, r_inv)
            score = score + score1

        return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        r_inv = self.rel_inv_embeddings(batch_r)
        regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2) + torch.mean(r_inv ** 2)) / 4
        """加的"""
        if self.add_mol:
            # 添加描述
            h_d = self.ent_description(batch_h)
            t_d = self.ent_description(batch_t)
            regul = (regul + (torch.mean(h_d ** 2) + torch.mean(t_d ** 2) + torch.mean(r ** 2) + torch.mean(r_inv ** 2)) / 4)/2

        return regul

    def predict(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        score = -self._calc_ingr(h, r, t)

        """        """
        if self.add_mol:
            # 修改为新的
            h_d = self.ent_description(batch_h)
            t_d = self.ent_description(batch_t)
            score = score - self._calc_ingr(h_d, r, t_d)

        return score.cpu().data.numpy()
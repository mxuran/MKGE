# !/usr/bin/env Python3
# -*- coding: utf-8 -*-
# @Author   : Tensory
# @email    : 1581554849@qq.com
# @FILE     : PairRE.py
# @Time     : 2021/12/13 7:44
import torch
import torch.autograd as autograd
import torch.nn as nn
from .Model import Model
import numpy as np


class PairRE(Model):
    def __init__(self, add_mol, entity_description_path, ent_tot, rel_tot, dim=100, margin = 6.0, epsilon = 2.0):
        super(PairRE, self).__init__(ent_tot, rel_tot)

        self.margin = margin
        self.epsilon = epsilon

        self.dim_e = dim
        self.dim_r = dim * 2
        self.add_mol = add_mol
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)

        """ 把词向量矩阵作为参数复制过去 """
        self.ent_description = nn.Embedding(self.ent_tot, self.dim_e)
        entity_description = np.load(entity_description_path)
        self.ent_description.weight.data.copy_(torch.from_numpy(entity_description))

        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_e]),
            requires_grad=False
        )

        nn.init.uniform_(
            tensor=self.ent_embeddings.weight.data,
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item()
        )

        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_r]),
            requires_grad=False
        )

        nn.init.uniform_(
            tensor=self.rel_embeddings.weight.data,
            a=-self.rel_embedding_range.item(),
            b=self.rel_embedding_range.item()
        )

        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False


    def _calc(self, h, t, r, mode):

        re_head, re_tail = torch.chunk(r, 2, dim=-1)

        h = nn.functional.normalize(h, 2, -1)
        t = nn.functional.normalize(t, 2, -1)
        # print(h.shape, t.shape, re_head.shape, re_tail.shape)
        h = h.view(-1, r.shape[0], h.shape[-1]).permute(1, 0, 2)
        t = t.view(-1, r.shape[0], t.shape[-1]).permute(1, 0, 2)
        re_head = re_head.view(-1, r.shape[0], re_head.shape[-1]).permute(1, 0, 2)
        re_tail = re_tail.view(-1, r.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
        # print(h.shape, t.shape, re_head.shape, re_tail.shape)       # torch.Size([2000, 1, 300]) torch.Size([2000, 26, 300]) torch.Size([2000, 1, 300]) torch.Size([2000, 1, 300])
        score = h * re_head - t * re_tail
        score = torch.norm(score, p=1, dim=2) # 计算对应实体和关系向量的L1范数 torch.Size([2000, 26])

        # print((score.permute(1, 0).flatten()).shape) # torch.Size([52000])
        return score.permute(1, 0).flatten()

        return score


    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        score = self._calc(h, t, r, mode)

        if self.add_mol:
            h_d_re = self.ent_description(batch_h)
            t_d_re = self.ent_description(batch_t)
            score2 = self._calc(h_d_re, t_d_re, r, mode)
            score += score2
        score = self.margin - score
        return score

    # 正则化要改
    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        print("开始正则化_______________________________________________")
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3
        return regul

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()

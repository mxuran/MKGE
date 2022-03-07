import torch
import torch.nn as nn
from .Model import Model
import numpy as np

class ComplEx(Model):
    def __init__(self, add_mol, entity_description_path, ent_tot, rel_tot, dim = 100):
        super(ComplEx, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.ent_re_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.ent_im_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_re_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.rel_im_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.add_mol = add_mol

        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)

        # my
        entity_description = np.load(entity_description_path)
        self.ent_description_re_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.ent_description_re_embeddings.weight.data.copy_(torch.from_numpy(entity_description))

        self.ent_description_im_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.ent_description_im_embeddings.weight.data.copy_(torch.from_numpy(entity_description))

    def _calc(self, h_re, h_im, t_re, t_im, r_re, r_im):
        return torch.sum(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            -1
        )

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)

        score = self._calc(h_re, h_im, t_re, t_im, r_re, r_im)

        if self.add_mol:
            h_d_re = self.ent_description_re_embeddings(batch_h)
            h_d_im = self.ent_description_im_embeddings(batch_h)

            t_d_re = self.ent_description_re_embeddings(batch_t)
            t_d_im = self.ent_description_im_embeddings(batch_t)

            score2 = self._calc(h_d_re, h_d_im, t_d_re, t_d_im, r_re, r_im)
            score +=  score2

        return score

    def regularization(self, data):

        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)

        regul = (torch.mean(h_re ** 2) +
                 torch.mean(h_im ** 2) +
                 torch.mean(t_re ** 2) +
                 torch.mean(t_im ** 2) +
                 torch.mean(r_re ** 2) +
                 torch.mean(r_im ** 2)) / 6

        if self.add_mol:
            # 添加描述
            h_d_re = self.ent_description_re_embeddings(batch_h)
            h_d_im = self.ent_description_im_embeddings(batch_h)

            t_d_re = self.ent_description_re_embeddings(batch_t)
            t_d_im = self.ent_description_im_embeddings(batch_t)
            regul2 = (torch.mean(h_d_re ** 2) +
                     torch.mean(h_d_im ** 2) +
                     torch.mean(t_d_re ** 2) +
                     torch.mean(t_d_im ** 2) +
                     torch.mean(r_re ** 2) +
                     torch.mean(r_im ** 2)) / 6
            regul = (regul + regul2)/2
        return regul

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()
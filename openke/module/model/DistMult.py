import torch
import torch.nn as nn
from .Model import Model
import numpy as np

class DistMult(Model):

	def __init__(self, add_mol, entity_description_path, ent_tot, rel_tot, dim = 100, margin = None, epsilon = None):
		super(DistMult, self).__init__(ent_tot, rel_tot)

		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
		self.add_mol = add_mol

		""""		"""
		# 把词向量矩阵作为参数复制过去
		self.ent_description = nn.Embedding(self.ent_tot, self.dim)
		entity_description = np.load(entity_description_path)
		self.ent_description.weight.data.copy_(torch.from_numpy(entity_description))



		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)

	def _calc(self, h, t, r, mode):
		if mode == 'head_batch':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
			score = h * (r * t)
		elif mode=='tail_batch':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
			score = (h * r) * t
		elif mode=='rel_batch':
			h = h.view(-1, h.shape[0], h.shape[-1])
			t = t.view(-1, t.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
			score = (h * r) * t
		else:
			score = (h * r) * t

		score = torch.sum(score, -1).flatten()
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
			# 增加基于描述的的向量表示
			h_d = self.ent_description(batch_h)
			t_d = self.ent_description(batch_t)
			score2 = self._calc(h_d, t_d, r, mode)
			score = score + score2
		return score

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3

		if self.add_mol:
			# 添加描述
			h_d = self.ent_description(batch_h)
			t_d = self.ent_description(batch_t)
			regul = (regul + (torch.mean(h_d ** 2) + torch.mean(t_d ** 2) + torch.mean(r ** 2)) / 3) / 2

		return regul

	def l3_regularization(self):
		return (self.ent_embeddings.weight.norm(p = 3)**3 + self.rel_embeddings.weight.norm(p = 3)**3)

	def predict(self, data):
		score = -self.forward(data)
		return score.cpu().data.numpy()

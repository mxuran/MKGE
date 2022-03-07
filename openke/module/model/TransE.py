import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
import numpy as np


class TransE(Model):

	def __init__(self,	add_mol, entity_description_path, ent_tot, rel_tot, dim = 100,
				 p_norm = 1, norm_flag = True, margin = None, epsilon = None):
		super(TransE, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm

		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

		self.add_mol = add_mol

		if self.add_mol:
			""" 把词向量矩阵作为参数复制过去 """
			self.ent_description = nn.Embedding(self.ent_tot, self.dim)
			entity_description = np.load(entity_description_path)
			self.ent_description.weight.data.copy_(torch.from_numpy(entity_description))
			# 是否需要冻结训练好的分子结构信息？


		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

		""" 原论文中的，先注释掉
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
		"""

		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False


	def _calc(self, h, t, r, mode):
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)

		# 链接预测的时候
		if mode == 'head_batch' or mode == 'tail_batch':
			# h.shape: torch.Size([1710, 300])	t.shape: torch.Size([1, 300])	r.shape: torch.Size([1, 300])
			h = h.view(-1, r.shape[0], h.shape[-1])  # [1710, 1, 300]
			t = t.view(-1, r.shape[0], t.shape[-1])  # [1, 1, 300]
			r = r.view(-1, r.shape[0], r.shape[-1])  # [1, 1, 300]

		# 关系预测的时候
		if mode == 'rel_batch':
			#  h.shape: torch.Size([1, 300]) t.shape: torch.Size([1, 300]) r.shape: torch.Size([86, 300])
			h = h.view(-1, h.shape[0], h.shape[-1])  # [1, 1, 300]
			t = t.view(-1, h.shape[0], t.shape[-1])  # [1, 1, 300]
			r = r.view(-1, r.shape[0], r.shape[-1])  # [1, 86, 300]				score = # [1, 86, 300]

		""" their results are same. It is written for understanding. 
		When you set sampling_mode=‘head_batch’,you will find that
		the sizes of h, r and t are different under 'head_batch' and 'tail_batch'，such as
		head_batch, h=torch.Size([26, 2721, 200]), r=torch.Size([1, 2721, 200]), t=torch.Size([1, 2721, 200])
		tail_batch, h=torch.Size([1, 2721, 200]), r=torch.Size([1, 2721, 200]), t=torch.Size([26, 2721, 200])
		"""
		if mode == 'head_batch':
			score = h + (r - t)		# [1710, 1, 300]
		else:
			score = (h + r) - t
		score = torch.norm(score, self.p_norm, -1).flatten()
		return score

	def forward(self, data):
		# 'batch_h': array([59,  434,  554, ..., 1698, 1376,  712]),
		# 'batch_t': array([1185, 1097,  834, ...,  401, 1265,  969]),
		# 'batch_r': array([32, 48, 74, ..., 48, 15, 69]),
		# 'batch_y': array([ 1.,  1.,  1., ..., -1., -1., -1.], dtype=float32), 'mode': 'normal'}

		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		score = self._calc(h, t, r, mode)

		if self.add_mol:
			h_d = self.ent_description(batch_h)
			t_d = self.ent_description(batch_t)
			# 两部分相加，MKGE中
			# score = score + self._calc(h_d, t_d, r, mode)

			# DKRL的思想
			score = score + self._calc(h_d, t_d, r, mode) + self._calc(h_d, t, r, mode) + self._calc(h, t_d, r, mode)

		if self.margin_flag:
			return self.margin - score
		else:
			return score

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)

		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2)) / 3

		# 新增
		if self.add_mol:
			h_d = self.ent_description(batch_h)
			t_d = self.ent_description(batch_t)
			regul = regul + (torch.mean(h_d ** 2) +
			 torch.mean(t_d ** 2) +
			 torch.mean(r ** 2)) / 3
		print('正则化')
		return regul

	def predict(self, data):
		score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()
import torch
import torch.autograd as autograd
import torch.nn as nn
from .Model import Model
import numpy as np
class RotatE(Model):

	def __init__(self, add_mol, entity_description_path, ent_tot, rel_tot, dim = 100, margin = 6.0, epsilon = 2.0):
		super(RotatE, self).__init__(ent_tot, rel_tot)

		self.margin = margin
		self.epsilon = epsilon

		self.dim_e = dim * 2
		self.dim_r = dim
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
			tensor = self.ent_embeddings.weight.data, 
			a=-self.ent_embedding_range.item(), 
			b=self.ent_embedding_range.item()
		)

		self.rel_embedding_range = nn.Parameter(
			torch.Tensor([(self.margin + self.epsilon) / self.dim_r]), 
			requires_grad=False
		)

		nn.init.uniform_(
			tensor = self.rel_embeddings.weight.data, 
			a=-self.rel_embedding_range.item(), 
			b=self.rel_embedding_range.item()
		)

		self.margin = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False

	def _calc(self, h, t, r, mode):
		pi = self.pi_const

		re_head, im_head = torch.chunk(h, 2, dim=-1)
		re_tail, im_tail = torch.chunk(t, 2, dim=-1)



		phase_relation = r / (self.rel_embedding_range.item() / pi)

		re_relation = torch.cos(phase_relation)
		im_relation = torch.sin(phase_relation)

		"""原始
		re_head = re_head.view(-1, re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
		re_tail = re_tail.view(-1, re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
		im_head = im_head.view(-1, re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
		im_tail = im_tail.view(-1, re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
		im_relation = im_relation.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
		re_relation = re_relation.view(-1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)
		"""

		if mode == "head_batch":
			# print(re_head.shape, im_head.shape, re_tail.shape, im_tail.shape)

			# 添加
			re_head = re_head.view(-1, re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
			re_tail = re_tail.view(-1, re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
			im_head = im_head.view(-1, re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
			im_tail = im_tail.view(-1, re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
			im_relation = im_relation.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
			re_relation = re_relation.view(-1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)
			# print(re_relation.shape[0])
			# print(re_head.shape, im_head.shape, re_tail.shape, im_tail.shape, im_relation.shape,re_relation.shape)
			# breakpoint()

			re_score = re_relation * re_tail + im_relation * im_tail
			im_score = re_relation * im_tail - im_relation * re_tail
			re_score = re_score - re_head
			im_score = im_score - im_head

		elif mode == "rel_batch":

			# 添加
			re_head = re_head.view(-1, re_head.shape[0], re_head.shape[-1]).permute(1, 0, 2)
			re_tail = re_tail.view(-1, re_tail.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
			im_head = im_head.view(-1, im_head.shape[0], im_head.shape[-1]).permute(1, 0, 2)
			im_tail = im_tail.view(-1, im_tail.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
			im_relation = im_relation.view(-1, re_relation.shape[0], re_relation.shape[-1])
			re_relation = re_relation.view(-1, re_relation.shape[0], re_relation.shape[-1])

			re_score = re_head * re_tail + im_head * im_tail
			im_score = re_head * im_tail - im_head * re_tail
			re_score = re_score - re_relation
			im_score = im_score - im_relation

		else:
			# 添加
			re_head = re_head.view(-1, re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
			re_tail = re_tail.view(-1, re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
			im_head = im_head.view(-1, re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
			im_tail = im_tail.view(-1, re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
			im_relation = im_relation.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
			re_relation = re_relation.view(-1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

			re_score = re_head * re_relation - im_head * im_relation
			im_score = re_head * im_relation + im_head * re_relation
			re_score = re_score - re_tail
			im_score = im_score - im_tail

		score = torch.stack([re_score, im_score], dim = 0)	 	# torch.Size([2, 2000, 26, 150])
		score = score.norm(dim = 0).sum(dim = -1)		# torch.Size([2000, 26])
		# print((score.permute(1, 0).flatten()).shape)	# torch.Size([52000])
		return score.permute(1, 0).flatten()

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		score1 = self.margin - self._calc(h, t, r, mode)

		if self.add_mol:
			# 加入描述
			h_d = self.ent_description(batch_h)
			t_d = self.ent_description(batch_t)
			# MKGE的
			# score2 = self.margin - self._calc(h_d, t_d, r, mode)
			# score = score1 + score2

			# MKRL的
			score2 = self.margin - self._calc(h_d, t_d, r, mode)
			score3 = self.margin - self._calc(h_d, t, r, mode)
			score4 = self.margin - self._calc(h, t_d, r, mode)
			score = score1 + score2 + score3 + score4

		else:
			score = score1

		return score

	def predict(self, data):
		score = -self.forward(data)
		return score.cpu().data.numpy()

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
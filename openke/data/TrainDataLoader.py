# coding:utf-8
import os
import ctypes
import numpy as np

class TrainDataSampler(object):

	def __init__(self, nbatches, datasampler):
		self.nbatches = nbatches
		self.datasampler = datasampler
		self.batch = 0

	def __iter__(self):
		return self

	def __next__(self):
		self.batch += 1 
		if self.batch > self.nbatches:
			raise StopIteration()
		return self.datasampler()

	def __len__(self):
		return self.nbatches

class TrainDataLoader(object):

	def __init__(self, 
		in_path = "./",		# 数据所在的根目录
		tri_file = None,				# 训练集
		ent_file = None,				# 实体集
		rel_file = None,				# 关系集
		batch_size = None,				# 批次大小
		nbatches = None,				# 批次数
		threads = 8,					# 线程数量
		sampling_mode = "normal",		# 采样方法
		bern_flag = False,
		filter_flag = True,
		neg_ent = 1,
		neg_rel = 0):
		
		base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
		self.lib = ctypes.cdll.LoadLibrary(base_file)
		"""argtypes"""
		self.lib.sampling.argtypes = [		# C与Python数据类型的转换，回调
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_int64,
			ctypes.c_int64,
			ctypes.c_int64,
			ctypes.c_int64,
			ctypes.c_int64,
			ctypes.c_int64,
			ctypes.c_int64
		]
		self.in_path = in_path			# 路径
		self.tri_file = tri_file
		self.ent_file = ent_file
		self.rel_file = rel_file
		if in_path != None:				# 将训练集、实体集、关系集的路径分别存放在对应的属性中
			self.tri_file = in_path + "train2id.txt"
			self.ent_file = in_path + "entity2id.txt"
			self.rel_file = in_path + "relation2id.txt"
		"""set essential parameters"""
		self.work_threads = threads
		self.nbatches = nbatches
		self.batch_size = batch_size
		self.bern = bern_flag
		self.filter = filter_flag
		self.negative_ent = neg_ent		# 负例实体
		self.negative_rel = neg_rel		# 负例关系
		self.sampling_mode = sampling_mode
		self.cross_sampling_flag = 0
		self.read()

	# 读训练数据
	def read(self):
		if self.in_path != None:
			self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
		else:
			self.lib.setTrainPath(ctypes.create_string_buffer(self.tri_file.encode(), len(self.tri_file) * 2))
			self.lib.setEntPath(ctypes.create_string_buffer(self.ent_file.encode(), len(self.ent_file) * 2))
			self.lib.setRelPath(ctypes.create_string_buffer(self.rel_file.encode(), len(self.rel_file) * 2))
		
		self.lib.setBern(self.bern)
		self.lib.setWorkThreads(self.work_threads)		# 设置工作线程
		self.lib.randReset()			# 重置所有线程的随机种子
		self.lib.importTrainFiles()		# 读取训练集
		self.relTotal = self.lib.getRelationTotal()		# 获取关系总数
		self.entTotal = self.lib.getEntityTotal()		# 获取实体总数
		self.tripleTotal = self.lib.getTrainTotal()		# 获取训练三元组总数

		if self.batch_size == None:
			self.batch_size = self.tripleTotal // self.nbatches		# 根据样本总数与batches的大小，计算batch_size的大小
		if self.nbatches == None:
			self.nbatches = self.tripleTotal // self.batch_size		# 根据样本总数与batch_size的大小，计算batches的大小
		self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)

		'''
		np.zeros返回来一个给定形状和类型的用0填充的数组；
		zeros(shape, dtype=float, order=‘C’)
			shape:形状
			dtype:数据类型，可选参数，默认numpy.float64
			order:可选参数，c代表与c语言类似，行优先；F代表列优先
		'''
		# 定义batch数据，包含头实体、尾实体、关系、标签，以及他们对应的数组首地址，其中标签batch_y，1表示原始三元组，-1表示替换后的三元组
		self.batch_h = np.zeros(self.batch_seq_size, dtype=np.int64)
		self.batch_t = np.zeros(self.batch_seq_size, dtype=np.int64)
		self.batch_r = np.zeros(self.batch_seq_size, dtype=np.int64)
		self.batch_y = np.zeros(self.batch_seq_size, dtype=np.float32)
		self.batch_h_addr = self.batch_h.__array_interface__["data"][0]
		self.batch_t_addr = self.batch_t.__array_interface__["data"][0]
		self.batch_r_addr = self.batch_r.__array_interface__["data"][0]
		self.batch_y_addr = self.batch_y.__array_interface__["data"][0]

	def sampling(self):
		self.lib.sampling(
			self.batch_h_addr,
			self.batch_t_addr,
			self.batch_r_addr,
			self.batch_y_addr,
			self.batch_size,
			self.negative_ent,
			self.negative_rel,
			0,
			self.filter,
			0,
			0
		)
		return {
			"batch_h": self.batch_h, 
			"batch_t": self.batch_t, 
			"batch_r": self.batch_r, 
			"batch_y": self.batch_y,
			"mode": "normal"
		}

	def sampling_head(self):
		self.lib.sampling(
			self.batch_h_addr,
			self.batch_t_addr,
			self.batch_r_addr,
			self.batch_y_addr,
			self.batch_size,
			self.negative_ent,
			self.negative_rel,
			-1,
			self.filter,
			0,
			0
		)
		return {
			"batch_h": self.batch_h,
			"batch_t": self.batch_t[:self.batch_size],
			"batch_r": self.batch_r[:self.batch_size],
			"batch_y": self.batch_y,
			"mode": "head_batch"
		}

	def sampling_tail(self):
		self.lib.sampling(
			self.batch_h_addr,
			self.batch_t_addr,
			self.batch_r_addr,
			self.batch_y_addr,
			self.batch_size,
			self.negative_ent,
			self.negative_rel,
			1,
			self.filter,
			0,
			0
		)
		return {
			"batch_h": self.batch_h[:self.batch_size],
			"batch_t": self.batch_t,
			"batch_r": self.batch_r[:self.batch_size],
			"batch_y": self.batch_y,
			"mode": "tail_batch"
		}

	def cross_sampling(self):
		self.cross_sampling_flag = 1 - self.cross_sampling_flag 
		if self.cross_sampling_flag == 0:
			return self.sampling_head()
		else:
			return self.sampling_tail()

	"""interfaces to set essential parameters"""

	def set_work_threads(self, work_threads):
		self.work_threads = work_threads

	def set_in_path(self, in_path):
		self.in_path = in_path

	def set_nbatches(self, nbatches):
		self.nbatches = nbatches

	def set_batch_size(self, batch_size):
		self.batch_size = batch_size
		self.nbatches = self.tripleTotal // self.batch_size

	def set_ent_neg_rate(self, rate):
		self.negative_ent = rate

	def set_rel_neg_rate(self, rate):
		self.negative_rel = rate

	def set_bern_flag(self, bern):
		self.bern = bern

	def set_filter_flag(self, filter):
		self.filter = filter

	"""interfaces to get essential parameters"""

	def get_batch_size(self):
		return self.batch_size

	def get_ent_tot(self):
		return self.entTotal

	def get_rel_tot(self):
		return self.relTotal

	def get_triple_tot(self):
		return self.tripleTotal

	def __iter__(self):
		if self.sampling_mode == "normal":
			return TrainDataSampler(self.nbatches, self.sampling)
		else:
			return TrainDataSampler(self.nbatches, self.cross_sampling)

	def __len__(self):
		return self.nbatches
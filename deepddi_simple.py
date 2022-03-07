import openke
from openke.config import Trainer, Tester
from openke.module.model import SimplE
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# True表示加入分子结构信息，False表示不加入
add_mol = True 	# False or True

# 数据集名字
dataset = "DeepDDI"
# 药物表示的路径
# entity_description_path = './gin_infomax_'+dataset+'.npy'

# MGbert 256
entity_description_path = './mgbert_'+dataset+'.npy'

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/%s/"%dataset,
	nbatches = 10,		# 30占3.3g gpu
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 0,
	neg_rel = 25
)


# define the model
simple = SimplE(
	add_mol = add_mol,
	entity_description_path = entity_description_path,
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 256
)

# define the loss function
model = NegativeSampling(
	model = simple, 
	loss = SoftplusLoss(),
	batch_size = train_dataloader.get_batch_size(),
	regul_rate = 1.0
)


# ----------------------------------------------训练----------------------------------------
"""
# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 2000, alpha = 0.5, use_gpu = True, opt_method = "adagrad", checkpoint_dir = "./checkpoint_DeepDDI", save_steps = 1000)
trainer.run()
if add_mol:
	simple.save_checkpoint('./checkpoint_'+dataset+'/simple_my.ckpt')
else:
	simple.save_checkpoint('./checkpoint_'+dataset+'/simple.ckpt')
"""

# ----------------------------------------------测试----------------------------------------
# 读取模型
if add_mol:
	simple.load_checkpoint('./checkpoint_'+dataset+'/simple_my.ckpt')
else:
	simple.load_checkpoint('./checkpoint_'+dataset+'/simple.ckpt')

"""
# 1、链接预测
test_dataloader = TestDataLoader("./benchmarks/%s/"%dataset, "test2id.txt", "link")
tester = Tester(model = simple, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
"""
"""
# 2、三元组分类
# 2.1先用验证集找到最好的threshold
print("---------------------------------------三元组分类验证--------------------=--------------")
valid_dataloader = TestDataLoader("./benchmarks/DeepDDI/", "valid2id.txt", "classification")
valider = Tester(model = complEx, data_loader = valid_dataloader, use_gpu = True)
acc, precision, recall, F1, roc_auc, pr_auc, threshlod = valider.run_triple_classification()
# 2.2拿得到的threshold去测试
print("---------------------------------------三元组分类测试--------------------=--------------")
test_dataloader = TestDataLoader("./benchmarks/DeepDDI/", "test2id.txt", "classification")
tester = Tester(model = complEx, data_loader = test_dataloader, use_gpu = True)
acc, precision, recall, F1, roc_auc, pr_auc, threshlod = tester.run_triple_classification(threshlod = threshlod)
print("准确率为：%.4f;精确率为：%.4f;召回率为：%.4f;F1值为：%.4f;roc_auc值为：%.4f;pr_auc为：%.4f" % (acc, precision, recall, F1, roc_auc, pr_auc))
"""

""""""
# 3、关系预测
test_dataloader = TestDataLoader("./benchmarks/%s/"%dataset, "test2id.txt", "relation")
tester = Tester(model = simple, data_loader = test_dataloader, use_gpu = True)
tester.run_relation_prediction()

"""4、DDI分类 """
tester = Tester(model = simple, data_loader = None, use_gpu = True)
tester.run_ddi_classification("./benchmarks/DeepDDI/", "test2id.txt")
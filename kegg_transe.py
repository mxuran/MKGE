import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# True表示加入分子结构信息，False表示不加入
add_mol = True 	# False or True

# 数据集名字
dataset = "KEGG"

# 药物表示的路径

# entity_description_path = "./benchmarks/"+dataset+"/cnn_"+dataset+"_300.npy"

# cnn+gin拼A
# entity_description_path = "./benchmarks/"+dataset+"/gin_infomax_cnn_"+dataset+"_600.npy"

# cnn+gin相加取平均
# entity_description_path = "./benchmarks/"+dataset+"/gin_infomax_cnn_"+dataset+"_300.npy"

# cnn+0拼接
entity_description_path = "./benchmarks/"+dataset+"/gin_infomax_0_"+dataset+"_600.npy"
# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/%s/"%dataset,
	nbatches = 10,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1,
	filter_flag = 1,
	neg_ent = 25,
	neg_rel = 0)



# define the model
transe = TransE(
	add_mol=add_mol,
	entity_description_path = entity_description_path,
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 600,
	p_norm = 1, 
	norm_flag = True)

# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

""""""
# ----------------------------------------------训练----------------------------------------
""""""
# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 200, alpha = 0.5, use_gpu = True)
trainer.run()
if add_mol:
	transe.save_checkpoint('./checkpoint_'+dataset+'/transe_my.ckpt')
else:
	transe.save_checkpoint('./checkpoint_'+dataset+'/transe.ckpt')


# ----------------------------------------------测试----------------------------------------
# 读取模型
if add_mol:
	transe.load_checkpoint('./checkpoint_'+dataset+'/transe_my.ckpt')
else:
	transe.load_checkpoint('./checkpoint_'+dataset+'/transe.ckpt')
""""""
# 1、链接预测
test_dataloader = TestDataLoader("./benchmarks/%s/"%dataset, "test2id.txt", "link")
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)


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


# 3、关系预测
test_dataloader = TestDataLoader("./benchmarks/%s/"%dataset, "test2id.txt", "relation")
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_relation_prediction()

"""4、DDI分类 """
print("---------------------------------------DDI分类验证--------------------=--------------")
valider = Tester(model = transe, data_loader = None, use_gpu = True)
threshlod = valider.run_ddi_classification("./benchmarks/%s/"%dataset, "valid2id.txt")
print("---------------------------------------DDI分类测试--------------------=--------------")
tester = Tester(model = transe, data_loader = None, use_gpu = True)
x = tester.run_ddi_classification("./benchmarks/%s/"%dataset, "test2id.txt", threshlod=threshlod)
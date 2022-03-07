# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
from sklearn import metrics
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support,accuracy_score
from sklearn.preprocessing import minmax_scale

class Tester(object):

    def __init__(self, model = None, data_loader = None, use_gpu = True):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testRel.argtypes = [ctypes.c_void_p, ctypes.c_int64]   #加的
        self.lib.test_link_prediction.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkMR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit10.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit3.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit1.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.restype = ctypes.c_float
        self.lib.getTestLinkMR.restype = ctypes.c_float
        self.lib.getTestLinkHit10.restype = ctypes.c_float
        self.lib.getTestLinkHit3.restype = ctypes.c_float
        self.lib.getTestLinkHit1.restype = ctypes.c_float

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.model.cuda()

    def set_model(self, model):
        self.model = model

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def test_one_step(self, data):        
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })

    def run_link_prediction(self, type_constrain = False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = tqdm(self.data_loader)
        for index, [data_head, data_tail] in enumerate(training_range):
            score = self.test_one_step(data_head)
            # print("预测头时候的分数",score,len(score))
            self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
            score = self.test_one_step(data_tail)
            self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
        self.lib.test_link_prediction(type_constrain)

        mrr = self.lib.getTestLinkMRR(type_constrain)
        mr = self.lib.getTestLinkMR(type_constrain)
        hit10 = self.lib.getTestLinkHit10(type_constrain)
        hit3 = self.lib.getTestLinkHit3(type_constrain)
        hit1 = self.lib.getTestLinkHit1(type_constrain)
        print(hit10)
        return mrr, mr, hit10, hit3, hit1

    def get_best_threshlod(self, score, ans):
        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        res_mx = 0.0
        threshlod = None
        for index, [ans, score] in enumerate(res):
            if ans == 1:
                total_current += 1.0
            res_current = (2 * total_current + total_false - index - 1) / total_all
            if res_current > res_mx:
                res_mx = res_current
                threshlod = score
        return threshlod, res_mx

    def run_triple_classification(self, threshlod = None):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('classification')
        score = []
        ans = []
        training_range = tqdm(self.data_loader)

        for index, [pos_ins, neg_ins] in enumerate(training_range):
            """ pos_ins
                {
                'batch_h': array([   7,   12,   12, ..., 1265, 1265,  846]),
                'batch_t': array([  1,  13,  17, ..., 706, 733, 650]),
                'batch_r': array([ 0,  1,  1, ..., 84, 84, 85]), 'mode': 'normal'
                }
            """
            res_pos = self.test_one_step(pos_ins)
            # ans此处为[1, 1, 1, 1, 1, 1, 1...1, 1, 1, 1, 1, 1],表示为正样本的分数
            ans = ans + [1 for i in range(len(res_pos))]
            score.append(res_pos)
            res_neg = self.test_one_step(neg_ins)
            ans = ans + [0 for i in range(len(res_pos))]
            score.append(res_neg)

        score = np.concatenate(score, axis = -1)

        ans = np.array(ans)

        # threshlod越小表示越真。
        if threshlod == None:
            threshlod, _ = self.get_best_threshlod(score, ans)

        res = np.concatenate([ans.reshape(-1, 1), score.reshape(-1,1)], axis = -1)
        """res
        [[ 1.         44.91710663]
         [ 1.         41.00870514]
         [ 1.         39.7370224 ]
         ...
         [ 0.         46.83553314]
        """

        # 对分数进行排序
        order = np.argsort(score) # 返回为从小到大的索引
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)    # 19228

        total_false = total_all - total_true    # 19228

        """
        # 原始的只有acc
        for index, [ans, score] in enumerate(res):
            # [ans, score] = [0.0, 22.982868194580078]
            # threshlod = 40.60258865356445

            if score > threshlod:
                acc = (2 * total_current + total_false - index) / total_all
                break
            elif ans == 1:
                total_current += 1.0

        print("原代码的精确率为：%.4f" % acc)
        """

        tp = 0.0    # 真正例，预测为真，实际为真
        fn = 0.0    # 假负例，预测为假，实际为真
        fp = 0.0    # 假正例，预测为真，实际为假
        tn = 0.0    # 真负例，预测为假，实际为假

        # 小于threshold为预测真，ans=1表示本来为真
        for index, [ans, score] in enumerate(res):
            if score < threshlod and ans == 1:
                tp += 1.0
            elif score < threshlod and ans == 0:
                fp += 1.0
            elif score >= threshlod and ans == 1:
                fn += 1.0
            elif score >= threshlod and ans == 0:
                tn += 1.0

        acc = (tp + tn) / (tp + tn + fn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fpr = fp / (fp + tn)    # FPR表示，所有真实为假的，被预测成真的比例。称为伪阳性率
        tpr = tp / (tp + fn)    # TPR表示，所有真实为真的，被预测成真的比例。称为真阳性率

        F1 = (2 * precision * recall)/(precision+recall)

        # print(fpr, tpr, precision, recall)
        # print("修改后的准确率为：%.4f;精确率为：%.4f;召回率为：%.4f;F1值为：%.4f" % (acc, precision, recall, F1))


        # 计算roc_auc和pr_auc值
        """        """
        labels = res[:, 0]
        scores = -res[:, 1]     # roc_curve中大于阈值是真的,我们的原始分数越小表示为真，因此
        # print("分数为", scores.shape, scores)
        
        # 归一化
        # scores = (scores - scores.min()) / (scores.max() - scores.min())
        # print("分数为", scores.shape, scores)


        fpr1, tpr1, thresholds = metrics.roc_curve(labels, scores)  # 假正率, 真正率 (recall = tpr)
        # print("阈值为", thresholds.shape,thresholds)
        roc_auc = metrics.auc(fpr1, tpr1)


        precision1, recall1, thresholds = metrics.precision_recall_curve(labels, scores)
        pr_auc = metrics.auc(recall1, precision1)

        return acc, precision, recall, F1, roc_auc, pr_auc, threshlod

    # 自己写一下关系预测
    def run_relation_prediction(self, type_constrain = False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('relation')

        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0

        training_range = tqdm(self.data_loader)

        for index, [relation_batch] in enumerate(training_range):
            """ relation_batch["batch_r"]
            [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
             24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
             48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
             72 73 74 75 76 77 78 79 80 81 82 83 84 85]
            """
            score = self.test_one_step(relation_batch)
            # print("分数的地址为：", score.__array_interface__["data"][0])
            self.lib.testRel(score.__array_interface__["data"][0], index)
        self.lib.test_relation_prediction()

    # DDI 分类
    def run_ddi_classification(self, in_path="", test_file="", threshlod = None):
        self.lib.initTest()
        self.lib.setInPath(ctypes.create_string_buffer(in_path.encode(), len(in_path) * 2))
        self.lib.randReset()
        self.lib.importTestFiles(ctypes.create_string_buffer(test_file.encode(), len(test_file) * 2))

        self.testTotal = self.lib.getTestTotal()
        self.relationTotal = self.lib.getRelationTotal()

        self.test_h = np.zeros(self.testTotal, dtype=np.int64)
        self.test_t = np.zeros(self.testTotal, dtype=np.int64)
        self.test_r = np.zeros(self.testTotal, dtype=np.int64)

        self.test_h_addr = self.test_h.__array_interface__["data"][0]
        self.test_t_addr = self.test_t.__array_interface__["data"][0]
        self.test_r_addr = self.test_r.__array_interface__["data"][0]

        # 1、读取所有的测试集
        self.lib.getTestList.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ]
        self.lib.getTestList(self.test_h_addr, self.test_t_addr, self.test_r_addr,)

        h = np.array(self.test_h)
        t = np.array(self.test_t)
        r = np.array(self.test_r)
        # print(h, t, r, type(h))

        # 2、对得到的测试集进行多标签标记，每个实体对对应的关系视为标签（可能为多个）
        # 2.1、遍历训练集，找到药物对，创建关系
        drug_pairs = {}

        for i in range(self.testTotal):     # 初始化drug_pairs
            drug_pair = str(h[i])+'_'+str(t[i])
            drug_pairs[drug_pair] = [0] * self.relationTotal

        for i in range(self.testTotal):     # 对drug_pairs赋值
            drug_pair = str(h[i]) + '_' + str(t[i])
            drug_pairs[drug_pair][r[i]] = 1



        pr = []
        roc = []
        p1 = []
        p3 = []
        p5 = []

        labels_f1 = []
        scores_f1 = []
        ans = []

        """"""
        # {'846_650': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,...], ...}
        for key in drug_pairs.keys():
            labels = drug_pairs[key]
            ans = ans + [i for i in labels]
            y_ture = torch.unsqueeze(torch.tensor(labels), 0).long().cpu().numpy()
            # print(y_ture)

            # 返回的为距离,本来是越小越好
            h_temp = np.array([int(key.split('_')[0])])
            t_temp = np.array([int(key.split('_')[1])])
            r_temp =  np.array([i for i in range(self.relationTotal)])


            scores = self.model.predict({
                'batch_h': self.to_var(h_temp, self.use_gpu),
                'batch_t': self.to_var(t_temp, self.use_gpu),
                'batch_r': self.to_var(r_temp, self.use_gpu),
                'mode': "rel_batch"
            })

            labels_f1.append(labels)
            scores_f1.append(scores.tolist())

            """  """
            # 在计算AUC、PR时，加上负号后则越大表示为真概率越大，即值越大越好
            scores = np.expand_dims(-scores, axis=0)

            metric = self.metric_report(y_ture, scores)

            pr.append(metric['pr'])
            roc.append(metric['roc'])
            p1.append(metric['p@1'])
            p3.append(metric['p@3'])
            p5.append(metric['p@5'])



        print('pr', np.mean(pr))
        print('roc', np.mean(roc))
        print('p@1', np.mean(p1))
        print('p@3', np.mean(p3))
        print('p@5', np.mean(p5))

        # # 计算F1值，所以得到threshlod，进行比较后，判断预测标签是否为真
        tp = 0.0  # 真正例，预测为真，实际为真
        fn = 0.0  # 假负例，预测为假，实际为真
        fp = 0.0  # 假正例，预测为真，实际为假
        tn = 0.0  # 真负例，预测为假，实际为假


        """自己写的计算F1的, micro-F1?
        scores_f1 = np.concatenate(scores_f1, axis=-1)
        ans = np.array(ans)
        if threshlod == None:
            threshlod, _ = self.get_best_threshlod(scores_f1, ans)
        res = np.concatenate([ans.reshape(-1, 1), scores_f1.reshape(-1, 1)], axis=-1)
        for index, [ans, score] in enumerate(res):
            if score < threshlod and ans == 1:
                tp += 1.0
            elif score < threshlod and ans == 0:
                fp += 1.0
            elif score >= threshlod and ans == 1:
                fn += 1.0
            elif score >= threshlod and ans == 0:
                tn += 1.0

        print(tp, fn, fp, tn)
        acc = (tp + tn) / (tp + tn + fn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fpr = fp / (fp + tn)  # FPR表示，所有真实为假的，被预测成真的比例。称为伪阳性率
        tpr = tp / (tp + fn)  # TPR表示，所有真实为真的，被预测成真的比例。称为真阳性率
        F1 = (2 * precision * recall) / (precision + recall)
        print("准确率为：%.4f;精确率为：%.4f;召回率为：%.4f;F1值为：%.4f" % (acc, precision, recall, F1))
        """
        lab_tru =[]
        for line in labels_f1:
            temp = line.index(max(line))
            lab_tru.append(temp)
        # print(len(lab_tru), lab_tru)

        lab_pre = []
        for line in scores_f1:
            temp = line.index(min(line))
            lab_pre.append(temp)
        # print(len(lab_pre), lab_pre)

        # average=None,取出每一类的P,R,F1值
        p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_true=lab_tru, y_pred=lab_pre,
                                                                                   labels= [i for i in range(self.relationTotal)], average=None)
        # print('各类单独F1:', f_class)
        print('各类F1取平均：', f_class.mean())
        print('F1-macro', f1_score(lab_tru, lab_pre, labels=[i for i in range(self.relationTotal)], average='macro'))
        print('F1-micro', f1_score(lab_tru, lab_pre, labels=[i for i in range(self.relationTotal)], average='micro'))
        print('accuracy', accuracy_score(lab_tru, lab_pre))
        return threshlod

    def metric_report(self, y, y_prob):
        rocs = []
        prs = []
        ks = [1, 3, 5]
        pr_score_at_ks = []

        for k in ks:
            pr_at_k = []
            for i in range(y_prob.shape[0]):
                y_prob_index_topk = np.argsort(y_prob[i])[::-1][:k]
                inter = set(y_prob_index_topk) & set(y[i].nonzero()[0])
                pr_ith = len(inter) / k
                pr_at_k.append(pr_ith)
            pr_score_at_k = np.mean(pr_at_k)
            pr_score_at_ks.append(pr_score_at_k)

        # for i in range(y.shape[1]):
        #     if (sum(y[:, i]) < 1):
        #         continue
        roc = roc_auc_score(y[0], y_prob[0])
        rocs.append(roc)

        prauc = average_precision_score(y[0], y_prob[0])
        prs.append(prauc)


        precision, recall, thresholds = metrics.precision_recall_curve(y[0], y_prob[0])
        roc_auc = sum(rocs) / len(rocs)
        pr_auc = sum(prs) / len(prs)

        return {'pr': pr_auc,
                'roc': roc_auc,
                'p@1': pr_score_at_ks[0],
                'p@3': pr_score_at_ks[1],
                'p@5': pr_score_at_ks[2]}


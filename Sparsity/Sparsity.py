# !/usr/bin/env Python3
# -*- coding: utf-8 -*-
# @Author   : Tensory
# @email    : 1581554849@qq.com
# @FILE     : Sparsity.py
# @Time     : 2020/11/21 14:37

dataset = "DeepDDI"


# 稀疏性：
Triple = 153828
Relation = 86
Entity = 1710
"""
Triple = 7047
Relation = 2928
Entity = 6893
"""
data_path = "../benchmarks/%s/train2id.txt"%dataset

#######################################################################################################################
# 采用了实体数量与事实数量、关系数量与事实数量的比值来从实体-关系两个角度来衡量数据整体的稀疏程度
# Sparsity and Noise: Where Knowledge Graph Embeddings Fall Short. EMNLP. 2017

Entity_Density = 2 * Triple / Entity
Relation_Density = Triple / Relation
print(Entity_Density, Relation_Density)

#######################################################################################################################
# 使用单一实体正则化后的频率来进行细致描述，因为并非所有实体都是稀疏的
# Iteratively Learning Embeddings and Rules for Knowledge Graph Reasoning. WWW. 2019
# 统计每个实体的稀疏值，然后画图
#freq_e 是实体e作为主语或宾语实体参与三元组的频率

freq_e_all = {}   # {'C11414': 2, 'C11417': 2, 'C18022': 3, 'C21192': 2, ...}

with open(data_path, 'r') as f:
    next(f)
    for line in f.readlines():
        h = line.strip().split('\t')[0]
        t = line.strip().split('\t')[2]

        if h not in freq_e_all.keys():
            freq_e_all[h] = 1
        else:
            freq_e_all[h] += 1

        if t not in freq_e_all.keys():
            freq_e_all[t] = 1
        else:
            freq_e_all[t] += 1
print(len(freq_e_all))

freq_min = min(freq_e_all.values())
freq_max = max(freq_e_all.values())

print(freq_min, freq_max)

for e in freq_e_all.keys():
    freq_e_all[e] = 1 - (freq_e_all[e]-freq_min) / (freq_max-freq_min)

sparse_num = 0
for e in freq_e_all.keys():
    if freq_e_all[e] > 0.995:
        sparse_num += 1
print(sparse_num/Entity)

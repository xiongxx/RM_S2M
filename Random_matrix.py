# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 22:52:43 2017

@author: Asual
"""

import numpy as np
from scipy import stats
from sklearn import random_projection
import copy

N = 10000; K=100
beta = 0.1
epsilon = 0.5
d = 30  #

c = (np.exp(epsilon)+1) / (np.exp(epsilon)+1)
prob = (np.exp(epsilon)) / (np.exp(epsilon)+1)

delta_power2 = np.log(2*K/beta) / (N*epsilon*epsilon)
m = int(np.floor(np.log(K+1)*np.log(2.0/beta) / delta_power2))

def gen_random_matrix():
    print("生成维度从", K, "降至", K, "的矩阵: ")
    X = np.random.rand(N,K)                     # 原始数据 （n*k）
    transformer = random_projection.SparseRandomProjection(n_components=K, density=1, eps=np.sqrt(delta_power2),random_state=100)
    transformer.fit_transform(X)                # 拟合变化后的数据 (n*m)
    random_matrix = transformer.components_         # 随机映射的矩阵 (m*k)
    print(random_matrix)
    return random_matrix

def gen_random_matrix_Phi():
    random_matrix = np.ones([m,K])*(1./np.sqrt(m))
    for j in range(K):
        for i in range(m):
            random_matrix[i][j] = np.random.choice([-1.,1.]) * random_matrix[i][j]
    return random_matrix

'''Algorithm 5生成一个正交矩阵(Collection.....文章的附录中的算法5)
@para: d表示长度，d是k的2次幂
'''
def gen_random_ortho_matrix(k):
    d = np.power(2,k)
    S = [[1,-1],[1,1]]    
    while(len(S)<d):
        Z = []        
        for i in S:            
            temp = copy.deepcopy(i)
            temp.extend(i)
            # Z[S.index(i)] = copy.deepcopy(temp)
            Z.append(temp)            
            temp = copy.deepcopy(i)
            temp.extend((np.array(temp)*-1).tolist())            
            Z.append(temp)
        S = copy.deepcopy(Z)        
    return np.array(S)
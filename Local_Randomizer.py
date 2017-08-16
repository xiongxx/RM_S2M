# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 21:10:52 2017

@author: Asual
@
"""
import numpy as np
import copy
import pandas as pd

from scipy import stats
from sklearn import random_projection
from random import shuffle
from scipy.stats import entropy
from numpy.linalg import norm
# from copy import deepcopy

import matplotlib.pyplot as plt

"""
@function: Randomized Projection Method
"""
def local_randomizer_location(x_vec, location_i, epsilon_i):    
    # 生成基向量
    basic_vec_i = Matrix_e[TAU.index(location_i)]    
    # 利用基向量与传入的d-bit(k，随机矩阵的一随机行）点乘，得到随机矩阵对应位置的值
    x_loc_i = np.dot(x_vec, basic_vec_i)
    # 参数c和p
    c = (np.exp(epsilon_i)+1) / (np.exp(epsilon_i)-1)
    prob = (np.exp(epsilon_i)) / (np.exp(epsilon_i)+1)    
    # 随机化 x_i    
    if(stats.bernoulli.rvs(prob) == 1):
        z_loc_i = (c * m * x_loc_i)
    else:
        z_loc_i = -(c * m * x_loc_i)
    # 返回随机结果
    return z_loc_i

''' implement randomized matrix by J_L lemna  '''
def gen_random_matrix():
    print("生成维度从", K, "降至", m, "的矩阵: ")
    X = np.random.rand(N,K)                     # 原始数据 （n*k）
    # transformer = random_projection.SparseRandomProjection(n_components=m, eps=np.sqrt(delta_power2), density=1,random_state=100)
    transformer = random_projection.SparseRandomProjection(n_components=m,eps=0.1,density=1,random_state=100)
    transformer.fit_transform(X)                # 拟合变化后的数据 (n*m)
    random_matrix = transformer.components_         # 随机映射的矩阵 (m*k)    
    return random_matrix

'''implement randomized matrix by independently fliping coin '''
def gen_random_matrix_Phi():
    random_matrix = np.ones([m,K])*(1./np.sqrt(m))
    for j in range(K):
        for i in range(m):
            random_matrix[i][j] = np.random.choice([-1.,1.]) * random_matrix[i][j]
    return random_matrix

'''Algorithm 5生成一个正交矩阵(Collection.....文章的附录中的算法5)
@para: d表示长度(2的n次幂)
'''
def gen_random_ortho_matrix(k):
    d = np.power(2,k)
    a = np.random.choice([-1,1]) # 生成随机正交矩阵
    S = [[a,-a],[a,a]]    
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
    return np.array(S)*1/(np.sqrt(d))          
                
"""Algorithm 1: Personalized count estimation protocol PCEP"""
def PCEP(Locations,epsilon):
    Matrix_Phi = gen_random_matrix_Phi()   # (m,tau) 
    # Matrix_Phi = gen_random_ortho_matrix(np.log2(K))    
    z_vec = np.zeros(m)                 # m-维的向量
    f_vec = np.zeros(len(TAU))          # |TAU|-维的向量 
    
    for i in range(N): 
        location_i = Locations[i]           # 为达到实验效果，每个用户随机选择选择一个位置        
        j = np.random.choice(range(m))      # 服务器随机选择矩阵一行
        phi_j = Matrix_Phi[j]               # 服务器发送矩阵的一随机行(len=d(or k))给 user_i
            
        z_i = local_randomizer_location(phi_j, location_i, epsilon)
        z_vec[j] += z_i        
        
    for k in range(len(TAU)):                     
        f_vec[k] = np.dot(np.dot(Matrix_Phi, Matrix_e[k]), z_vec)        
    return f_vec

"""
@function: Metric function
"""
def JS_divergence(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

if __name__ == '__main__':    
    
    N = 1600; K=8
    beta = 0.1
    m = K

    TAU = ["L"+str(i) for i in range(K)]
    Matrix_e = np.eye(len(TAU))
    
    '''生成模拟数据集'''    
    Locations = [val for val in TAU for i in range(int(N/K))]   #数据均匀分布
    
#    Locations = [val for val in TAU for i in range(5)]          # Peak分布
#    element = np.random.choice(TAU)
#    Locations.extend(element for i in range(N-len(Locations)))
    
    shuffle(Locations)                                          # 将模拟数据乱序处理
                          
    real_x = pd.Series(Locations).value_counts().sort_index()   # 每个位置真实的用户分布，且进行排序处理

    epsilons = np.linspace(0.1,1,10)
    p_MSE = []    
    p_KL = []
    p_JS = []
    noise_x = pd.Series([0], index=TAU)
    
    times = 10
    for epsilon in epsilons:
        for time in range(times):
            noise_x += pd.Series(PCEP(Locations,epsilon), index=TAU)
        noise_x = abs(round(noise_x / times))
        
        p_MSE.append((((real_x - noise_x)/N)**2).mean(axis=None))
        
        p_KL.append(entropy(real_x, noise_x)) # 输入：变量分布和概率分布
        p_JS.append(JS_divergence(real_x, noise_x))
    
    print(noise_x.sum())
    
    plt.figure(1)
    plt.subplot(211)
    plt.xlabel('Epsilon')
    plt.ylabel('MSE')
    plt.plot(epsilons, p_MSE, 'r.-',epsilons, p_KL, 'b.-',epsilons, p_JS, 'g.-')
    plt.show        
        
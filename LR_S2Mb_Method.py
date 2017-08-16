# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 21:10:52 2017

@author: Asual
@
"""
import numpy as np
from scipy import stats
from sklearn import random_projection
import copy
from random import shuffle
import pandas as pd

from scipy.stats import entropy
from numpy.linalg import norm

import matplotlib.pyplot as plt

from copy import deepcopy
import math

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


"""
@function: S2Mb Method
"""
# 获取一个size=s的集合，且该集合不包含元素t
def getRandSet(arr, t, s):
    Rand_set = list()
    #deleted_arr = np.delete(arr, np.argwhere(arr == t)) # 抽取元素的index，然后删除该元素
    arr.remove(t)
    arr = np.random.choice(arr, size = s, replace=False )  # choice without replacement
    #arr = np.random.choice(arr, size = s)  # choice with replacement    
    for i in arr:
        Rand_set.append(i)    
    return Rand_set

# Algorithm 1
def single_protocol(t, TAU, s, prob):         
    arr_K = deepcopy(TAU)   #复制一个副本，进行删除处理，不然的话会删除原始数据（赋值和浅赋值都是通过指针的方式）           
    #if(np.random.rand(1) > p):
    if(np.random.choice([0,1],p=[1-prob,prob])==1):               
        R_u = getRandSet(arr_K, t, s-1)
        R_u.append(t) 
    else:
        R_u = getRandSet(arr_K, t, s)    
    return R_u

# 
"""Algorithm 2
@paragram 
    @reported set: Rep_set:
    @category size: F
    @t is included in R_u with probability: p
    @R_u set size: s

"""""

def aggregate_protocol_S2M(R_u, F, s, p):    
    w = {x:y for x, y in zip(TAU, [0]*len(TAU))}     # 定义长度为F的dict
    noise_x = {x:y for x, y in zip(TAU, [0]*len(TAU))}
    q = (s-p)/(F-1)
    for j in range(N):    
        for i in R_u[j]:                                
            w[i] += 1                       
    for i in TAU:
        noise_x[i] = round((-q*N + w[i])/(p - q))
    return noise_x


        
def aggregate_protocol_S2Mb(R_u, F, s, p):    
    w = {x:y for x, y in zip(TAU, [0]*len(TAU))}     # 定义长度为F的dict
    noise_x = {x:y for x, y in zip(TAU, [0]*len(TAU))}
    new_x = {x:y for x, y in zip(TAU, [0]*len(TAU))} 
    # L = {x:y for x, y in zip(TAU, [0]*len(TAU))}              
    
    q = (s-p)/(F-1)    
    
    for j in range(N): 
        for i in R_u[j]:
            w[i] += 1
    for i in TAU:
        noise_x[i] = w[i]   
                
    Times =0  # times
    Threshould = 1
    while(True):
        Z = 0
        Times += 1
        #print(times,'####')        # 为迭代的次数
               
        for i in TAU:
            new_x[i] = w[i] / ((p-q)*noise_x[i] + q*s*N)                                     
        for i in new_x.values():
            Z += i        
        
        for i in TAU:                        
            new_x[i] = noise_x[i] * ((p-q)*new_x[i] + q*Z)
            
        totalCurErr = 0.0        
        for i in TAU:
            totalCurErr  += abs(new_x[i]-noise_x[i])            
        #print('totalCurErr = ',totalCurErr) 
        
        for i in TAU:
            noise_x[i] = new_x[i]            
            
        # if((totalCurErr < Threshould )):     # 为提前终止迭代的条件（自己预设的）
        if((totalCurErr < Threshould ) or (Times >= 1000)):          
            break
            
    for i in TAU:
        noise_x[i] = round(noise_x[i]/s)
        #print(noise_x[i])                
    return noise_x

def f_gen_mse_jsd(Locations,epsilons):
    mse_RP = []
    mse_S2Mb = []
    jsd_RP = []
    jsd_S2Mb = []
    real_x = pd.Series(Locations).value_counts().sort_index()   # 每个位置真实的用户分布，且进行排序处理
    for epsilon in epsilons:
        noise_x_RP = pd.Series([0], index=TAU)     # 添加个绝对值
        
        s = max(math.floor(F/(1+ np.exp(epsilon))), 1)   # get a list which size is 3
        p = (np.exp(epsilon)*s)/(F-s+np.exp(epsilon)*s)
        R_u = np.array(np.zeros((N,s)), dtype=str)
        noise_x_S2Mb = pd.Series([0], index=TAU)
        
        for time in range(times):
            noise_x_RP += pd.Series(PCEP(Locations,epsilon), index=TAU)   # RP噪声计数
            
            for i in range(N):                                            # S2Mb噪声计数  
                R_u[i] = single_protocol(Locations[i], TAU, s, p)
            noise_x_S2Mb += pd.Series(aggregate_protocol_S2Mb(R_u, F, s, p))
            
        noise_x_RP = abs(round(noise_x_RP / times))
        noise_x_S2Mb = round(noise_x_S2Mb / times)
        
        print('####')
        
        mse_RP.append((((real_x - noise_x_RP)/N)**2).mean(axis=None))
        jsd_RP.append(JS_divergence(real_x/N,noise_x_RP/N))
        
        mse_S2Mb.append((((real_x - noise_x_S2Mb)/N)**2).mean(axis=None))
        jsd_S2Mb.append(JS_divergence(real_x/N,noise_x_S2Mb/N))
        
    return mse_RP, mse_S2Mb, jsd_RP, jsd_S2Mb
        #p_KL.append(entropy(real_x, noise_x)) # 输入：变量分布和概率分布
        #p_JS.append(JS_divergence(real_x, noise_x))

if __name__ == '__main__':    
    
    np.random.seed(0)
    N = 1000; K=10
    #beta = 0.1
    m = K
    F = K

    TAU = ["L"+str(i) for i in range(K)]
    Matrix_e = np.eye(len(TAU))
    # Matrix_Phi = gen_random_matrix_Phi()        # 只生成1个随机矩阵
    
    '''生成模拟数据集'''    
    Locations_U = [val for val in TAU for i in range(int(N/K))]   #均匀分布
    shuffle(Locations_U)
    
    Locations_N = []    
    norm_v = np.random.normal(np.ceil(K/2), 2, N )              #正太分布
    for i in norm_v:
        temp = int(np.ceil(i))
        if((temp >= 10) or (temp < 0)):
            temp = 0
        Locations_N.append("L"+str(temp))
    #shuffle(Locations_N)
    
    Locations_P = [val for val in TAU for i in range(5)]          # Peak分布
    element = np.random.choice(TAU)
    Locations_P.extend(element for i in range(N-len(Locations_P)))
    shuffle(Locations_P)
    
    Locations_R = [np.random.choice(TAU) for i in range(N)]            # 随机分布    
    shuffle(Locations_R)                                          # 将模拟数据乱序处理                          
    

    epsilons = np.linspace(0.1,1,10)

    times = 10
    
    # plt.figure(figsize=(10,2))
    plt.figure(figsize=(12,4))
    
    MSE_U = f_gen_mse_jsd(Locations_U,epsilons)    
    plt.subplot(241)
    plt.xlabel('Epsilon')
    plt.ylabel('MSE')
    plt.xticks(np.linspace(0,1,6,endpoint=True))  # 设置x轴刻度
    #plt.yticks(np.linspace(0,0.5,10,endpoint=True))  # 设置y轴刻度
    plt.plot(epsilons, MSE_U[0],'-k.',label='RM')
    plt.plot(epsilons, MSE_U[1], '-k^',label='PLAS')
    plt.legend()
    
    plt.subplot(245)
    plt.xlabel('Epsilon')
    plt.ylabel('JS-divergence')
    plt.xticks(np.linspace(0,1,6,endpoint=True))  # 设置x轴刻度
    #plt.yticks(np.linspace(0,0.5,10,endpoint=True))  # 设置y轴刻度
    plt.plot(epsilons, MSE_U[2],'-k.',label='RM')
    plt.plot(epsilons, MSE_U[3], '-k^',label='PLAS')
    plt.legend()
    
    
    MSE_N = f_gen_mse_jsd(Locations_N,epsilons)
    plt.subplot(242)
    plt.title('')
    plt.xlabel('Epsilon')
    plt.ylabel('MSE')
    plt.xticks(np.linspace(0,1,6,endpoint=True))  # 设置x轴刻度
    #plt.yticks(np.linspace(0,0.5,2,endpoint=True))  # 设置y轴刻度
    plt.plot(epsilons, MSE_N[0], '-k.',label='RM')
    plt.plot(epsilons,MSE_N[1], '-k^',label='PLAS')
    plt.legend()
    
    plt.subplot(246)
    plt.xlabel('Epsilon')
    plt.ylabel('JS-divergence')
    plt.xticks(np.linspace(0,1,6,endpoint=True))  # 设置x轴刻度
    #plt.yticks(np.linspace(0,0.5,10,endpoint=True))  # 设置y轴刻度
    plt.plot(epsilons, MSE_N[2],'-k.',label='RM')
    plt.plot(epsilons, MSE_N[3], '-k^',label='PLAS')
    plt.legend()
    
    
    MSE_P = f_gen_mse_jsd(Locations_P,epsilons)
    plt.subplot(243)
    plt.xlabel('Epsilon')
    plt.ylabel('MSE')
    plt.xticks(np.linspace(0,1,6,endpoint=True))  # 设置x轴刻度
    #plt.yticks(np.linspace(0,0.5,2,endpoint=True))  # 设置y轴刻度    
    plt.plot(epsilons, MSE_P[0], '-k.',label='RM')
    plt.plot(epsilons,MSE_P[1], '-k^',label='PLAS')
    plt.legend()
    
    plt.subplot(247)
    plt.xlabel('Epsilon')
    plt.ylabel('JS-divergence')
    plt.xticks(np.linspace(0,1,6,endpoint=True))  # 设置x轴刻度
    #plt.yticks(np.linspace(0,0.5,10,endpoint=True))  # 设置y轴刻度
    plt.plot(epsilons, MSE_P[2],'-k.',label='RM')
    plt.plot(epsilons, MSE_P[3], '-k^',label='PLAS')
    plt.legend()
    

    MSE_R = f_gen_mse_jsd(Locations_R,epsilons)
    plt.subplot(244)
    plt.xlabel('Epsilon')
    plt.ylabel('MSE')
    plt.xticks(np.linspace(0,1,6,endpoint=True))  # 设置x轴刻度
    #plt.yticks(np.linspace(0,0.5,2,endpoint=True))  # 设置y轴刻度
    plt.plot(epsilons, MSE_R[0], '-k.',label='RM')
    plt.plot(epsilons,MSE_R[1], '-k^',label='PLAS')
    plt.legend()
    
    plt.subplot(248)
    plt.xlabel('Epsilon')
    plt.ylabel('JS-divergence')
    plt.xticks(np.linspace(0,1,6,endpoint=True))  # 设置x轴刻度
    #plt.yticks(np.linspace(0,0.5,10,endpoint=True))  # 设置y轴刻度
    plt.plot(epsilons, MSE_R[2],'-k.',label='RM')
    plt.plot(epsilons, MSE_R[3], '-k^',label='PLAS')
    plt.legend()
    
    plt.subplots_adjust(hspace=0.4,wspace=0.4)    
    plt.show()
    # plt.savefig('mse_RPSB.tiff')

          
        
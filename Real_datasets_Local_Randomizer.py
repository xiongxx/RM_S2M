# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 21:10:52 2017

@author: Asual
"""
import numpy as np
from scipy import stats
from sklearn import random_projection
import copy
import pandas as pd

from scipy.stats import entropy
from numpy.linalg import norm

import matplotlib.pyplot as plt

#N = 1000; K=10
#beta = 0.1
#epsilon = 0.5

# delta_power2 = np.log(2*K/beta) / N
#delta_power2 = np.log(2*K/beta) / (N*np.power(epsilon,2))
#m = int(np.floor(np.log(K+1)*np.log(2.0/beta) / delta_power2))
#m = K
#
#TAU = ["L"+str(i) for i in range(K)]
#Matrix_e = np.eye(len(TAU))

""" user 
@name: Local randomizer_loc
@param: 
    1. d-bit string --- x(向量是d*1的)
    2. user u_i's location --- i (全部可能位置中的第i个位置)
    ## 3. user's privacy parameter ----epsilon_i
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


""" Server """
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
def PCEP(Locations):
    Matrix_Phi = gen_random_matrix_Phi()   # (m,tau) 
    # Matrix_Phi = gen_random_ortho_matrix(np.log2(K))    
    z_vec = np.zeros(m)                 # m-维的向量
    f_vec = np.zeros(len(TAU))          # |TAU|-维的向量 
    
    for i in range(len(Locations)): 
        location_i = Locations[i]           # 为达到实验效果，每个用户随机选择选择一个位置        
        j = np.random.choice(range(m))      # 服务器随机选择矩阵一行
        phi_j = Matrix_Phi[j]               # 服务器发送矩阵的一随机行(len=d(or k))给 user_i
            
        z_i = local_randomizer_location(phi_j, location_i, epsilon)
        z_vec[j] += z_i        
        
    for k in range(len(TAU)):                     
        f_vec[k] = np.dot(np.dot(Matrix_Phi, Matrix_e[k]), z_vec)        
    return f_vec

def JS_divergence(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

if __name__ == '__main__':
    
    # Matrix_Phi = gen_random_matrix()              # There is a Randomized matrix in runing times
    # Matrix_Phi = gen_random_matrix_Phi()
    # Matrix_Phi = gen_random_ortho_matrix(np.log2(K)) 
    
    '''真实数据集'''
    Path = 'F:\实验数据\Gowalla 签到数据\loc-gowalla_totalCheckins\LV.xlsx'
    column_names = ['user','time','latitude','longitude','location']
    data_type = {'user':np.int,'time':str,'latitude':np.float32,'longitude':np.float32,'location':str}  
      
    
    xlsx = pd.ExcelFile(Path)
    df = pd.read_excel(xlsx, names=column_names)

    # Locations =  
    step = 0.01
    epsilon = 0.8
    
    location_randomized_counts = pd.Series()
    location_real_counts = pd.Series()    
    
    for lat in np.arange(-115.35,-115.00+step,step):
        for long in np.arange(35.55,36.35+step,step):
            small_df = df[(df.longitude>=lat) & (df.longitude<lat+step) & (df.latitude>=long) & (df.latitude<long+step)]
            if(small_df.empty):
                continue
            location_counts = small_df.location.value_counts()  #Series()            
            TAU = list(location_counts.index)       # 位置域（set）
            if(len(TAU) <= 2):
                continue
            
            print('Small Location domain: ' ,len(TAU))
            location_real_counts = location_real_counts.append(location_counts)
            
            # print(small_df.location.values)
            Locations = small_df.location.values
            K = len(TAU)
            m = K
            Matrix_e = np.eye(K)
            
            times = 10    # 可以去除计数值为负值
            sum_vec = np.zeros(K)
            for i in range(times):
                f_vec = PCEP(Locations)
                sum_vec += f_vec
            f_vec = sum_vec / times
            
            location_randomized_counts = location_randomized_counts.append(pd.Series(f_vec,index=TAU))
    jd_d = JS_divergence(location_randomized_counts,location_real_counts)        
        
        
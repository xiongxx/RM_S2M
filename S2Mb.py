# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 21:57:53 2017

@author: Asual
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:23:37 2017
@from the paper: Differential Private Data Collection and Analysis Based on Randomized Multiple Dummies for Untrusted Mobile Crowdsensing

@author: Asual
"""

import numpy as np
import pandas as pd
import math

from scipy import stats
from numpy.linalg import norm
from random import shuffle
from copy import deepcopy

import matplotlib.pyplot as plt

""" Analysis of Randomized Response """""
## generate a matrix which size is (11*11), and of which diagonal elements is o.5 and other is 0.05, 
#a = np.zeros((11,11)) + 0.05
#b = np.eye(11) * 0.45
#mMat = np.mat(a + b)        # array = matrix
#mMat_inv = mMat.I
#
#
## input matrix x=[]
#xVec = np.mat([100,0,0,0,0,0,0,0,0,0,0])
#yVec = xVec * mMat                      # Generate the unbiased estimation
#print(yVec)
#
#yVec = np.mat([40,6,6,6,6,6,6,6,6,6,6]) # Genertate the estimated vector yVec 
#
#xVec = mMat_inv * yVec.transpose()
#
#for i in np.array(xVec):
#    print(i)

"""Algorithm 1 Node Protocol for a Participant u"""""



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
def single_protocol(t, ARR_K, s, prob):         
    arr_K = deepcopy(ARR_K)   #复制一个副本，进行删除处理，不然的话会删除原始数据（赋值和浅赋值都是通过指针的方式）           
    #if(np.random.rand(1) > p):
    if(stats.bernoulli.rvs(prob) == 1):               
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
    w = {x:y for x, y in zip(ARR_K, [0]*len(ARR_K))}     # 定义长度为F的dict
    noise_x = {x:y for x, y in zip(ARR_K, [0]*len(ARR_K))}
    q = (s-p)/(F-1)
    for j in range(N):    
        for i in R_u[j]:                                
            w[i] += 1                       
    for i in ARR_K:
        noise_x[i] = round((-q*N + w[i])/(p - q))
    return noise_x


        
def aggregate_protocol_S2Mb(R_u, F, s, p):    
    w = {x:y for x, y in zip(ARR_K, [0]*len(ARR_K))}     # 定义长度为F的dict
    noise_x = {x:y for x, y in zip(ARR_K, [0]*len(ARR_K))}
    new_x = {x:y for x, y in zip(ARR_K, [0]*len(ARR_K))} 
    # L = {x:y for x, y in zip(ARR_K, [0]*len(ARR_K))}              
    
    q = (s-p)/(F-1)    
    
    for j in range(N): 
        for i in R_u[j]:
            w[i] += 1
    for i in ARR_K:
        noise_x[i] = w[i]   
                
    times =0  # times
    
    while(True):
        Z = 0
        times += 1
        # print(times,'####')        # 为迭代的次数
               
        for i in ARR_K:
            new_x[i] = w[i] / ((p-q)*noise_x[i] + q*s*N)                                     
        for i in new_x.values():
            Z += i        
        
        for i in ARR_K:                        
            new_x[i] = noise_x[i] * ((p-q)*new_x[i] + q*Z)
            
        totalCurErr = 0.0        
        for i in ARR_K:
            totalCurErr  += abs(new_x[i]-noise_x[i])            
        #print('totalCurErr = ',totalCurErr) 
        
        for i in ARR_K:
            noise_x[i] = new_x[i]            
            
        if((totalCurErr < Threshould )):     # 为提前终止迭代的条件（自己预设的）           
            break
            
    for i in ARR_K:
        noise_x[i] = round(noise_x[i]/s)
        # print(noise_x[i])                
    return noise_x

def JS_divergence(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (stats.entropy(_P, _M) + stats.entropy(_Q, _M))

if __name__ == '__main__': 
    
    
    F = 10
    N = 10
    # Threshould = 0.1*s
    
    ARR_K = ["L"+str(i) for i in range(F)] 

    '''模拟均匀分布的S2Mb的实验'''
    what_you_want = [val for val in ARR_K for i in range(int(N/F))]    # 均匀分布
    
    # what_you_want = [np.random.choice(ARR_K) for i in range(N)]       # 随机分布   
    
    # what_you_want = [val for val in ARR_K for i in range(20)]   # Peak分布
    # element = np.random.choice(ARR_K)
    # what_you_want.extend(element for i in range(N-len(what_you_want)))
    
    shuffle(what_you_want)              # 洗牌，打乱顺序
    
    real_x = pd.Series(what_you_want).value_counts().sort_index()       
    # 实时
    epsilons = np.linspace(0.1,1,10)
    p_MSE = []
    p_KL = []
    p_JS = []
    for epsilon in epsilons:
        s = max(math.floor(F/(1+ np.exp(epsilon))), 1)   # get a list which size is 3        
        p = (np.exp(epsilon)*s)/(F-s+np.exp(epsilon)*s)
        Threshould = 0.01*s
        print('epsilon=',epsilon, ' s=',s,'  p=',p)        

        # sum = pd.Series([0],index=real_x.index)
        R_u = np.array(np.zeros((N,s)), dtype=str)
        for i in range(N):
            R_u[i] = single_protocol(what_you_want[i], ARR_K, s, p)
        noise_x = pd.Series(aggregate_protocol_S2Mb(R_u, F, s, p))        
    
        p_MSE.append((((real_x - noise_x)/N)**2).mean(axis=None))
        p_KL.append(stats.entropy(real_x, noise_x)) # 输入：变量分布和概率分布
        p_JS.append(JS_divergence(real_x, noise_x))
    
    plt.figure(1)
    plt.subplot(211)
    plt.xlabel('Epsilon')
    plt.ylabel('MSE')
    plt.plot(epsilons, p_MSE, 'r.-')
    plt.show
    plt.savefig('tmp.tiff')
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 17:25:46 2017

@author: Asual
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 21:10:52 2017

@author: Asual
@
"""
import numpy as np
from scipy import stats
from sklearn import random_projection
import pandas as pd

from scipy.stats import entropy
from numpy.linalg import norm

import matplotlib.pyplot as plt
import copy
from copy import deepcopy
import math

import matplotlib as mpl
mpl.use('Agg')

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
            
        if((totalCurErr < Threshould ) or (Times > 1000)):     # 为提前终止迭代的条件（自己预设的）           
            break
            
    for i in TAU:
        noise_x[i] = round(noise_x[i]/s)
        #print(noise_x[i])                
    return noise_x

def f_noise_counts(Locations,epsilon): 
    N = len(Locations)    
        
    s = max(math.floor(F/(1+ np.exp(epsilon))), 1)   # get a list which size is 3
    p = (np.exp(epsilon)*s)/(F-s+np.exp(epsilon)*s)
    R_u = np.array(np.zeros((N,s)))
    noise_count_S2Mb = pd.Series([0], index=TAU)    # noise_count_S2Mb初始值
        
    for time in range(times):       
        for i in range(N):                                            # S2Mb噪声计数  
            R_u[i] = single_protocol(Locations[i], TAU, s, p)
        noise_count_S2Mb += pd.Series(aggregate_protocol_S2Mb(R_u, F, s, p))
        
    
    noise_count_S2Mb = round(noise_count_S2Mb / times)
        
    return noise_count_S2Mb

def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')
        # 柱形图边缘用白色填充，纯粹为了美观
        rect.set_edgecolor('white')

if __name__ == '__main__':    
    
    np.random.seed(0)

    # Matrix_Phi = gen_random_matrix_Phi()        # 只生成1个随机矩阵
    
    '''真实数据集'''
    Path = 'F:\实验数据\Gowalla 签到数据\loc-gowalla_totalCheckins\LV.xlsx'
    column_names = ['user','time','latitude','longitude','location']
    data_type = {'user':np.int,'time':str,'latitude':np.float32,'longitude':np.float32,'location':str}  
      
    
    xlsx = pd.ExcelFile(Path)
    df = pd.read_excel(xlsx, names=column_names)
    
    '''剔除计数大于4的位置'''
    location_counts = df.location.value_counts()
    li = list() 
    for index, value in location_counts.iteritems():
        if(value > 4):
            li.append(index)
        else:
            break        
    df_redu = pd.concat([df.query('location == @i') for i in li])
    df_redu = df_redu.sort_index() 
    df_redu = df_redu.reset_index(drop=True)
    df = df_redu    

    # Locations =  
    step = 0.01   
    times = 10
    epsilons_low = np.linspace(0.25,0.75,3)
    epsilons_mid = np.linspace(0.5, 1., 3)
    epsilons_high = np.linspace(0.75,1.25,3)
    
    mse_RP = []
    jsd_RP = []
    mse_S2Mb = []
    jsd_S2Mb = []    
    
    location_real_counts = pd.Series()
    location_randomized_counts_low = pd.Series()
    location_randomized_counts_mid = pd.Series()
    location_randomized_counts_high = pd.Series()
    
    for lat in np.arange(-115.35,-115.00+step,step):
        for long in np.arange(35.55,36.35+step,step):
            small_df = df[(df.longitude>=lat) & (df.longitude<lat+step) & (df.latitude>=long) & (df.latitude<long+step)]
            if(small_df.empty):             # 跳过空域
                continue
            location_counts = small_df.location.value_counts()  #Series()            
            TAU = list(location_counts.index)       # 位置域（set）
            if(len(TAU) <= 2):              # 跳过阈值小于等于2的域
                continue
            
            print('Small Location domain: ' ,len(TAU))                
            location_real_counts = location_real_counts.append(location_counts)           
            Locations = small_df.location.values           
            
            N = len(Locations)
            K = len(TAU)
            m = K
            F = K
            Matrix_e = np.eye(K)
                
            times = 1
            
            epsilon = np.random.choice(epsilons_low)            
            noise_counts = f_noise_counts(Locations,epsilon)
            location_randomized_counts_low = location_randomized_counts_low.append(noise_counts)

            epsilon = np.random.choice(epsilons_mid)
            noise_counts = f_noise_counts(Locations,epsilon)
            location_randomized_counts_mid = location_randomized_counts_mid.append(noise_counts)
            
            epsilon = np.random.choice(epsilons_high)
            noise_counts = f_noise_counts(Locations,epsilon)
            location_randomized_counts_high = location_randomized_counts_high.append(noise_counts)
    
    jsd_Real_low = JS_divergence(location_real_counts,location_randomized_counts_low)
    jsd_Real_mid = JS_divergence(location_real_counts, location_randomized_counts_mid)
    jsd_Real_high = JS_divergence(location_real_counts,location_randomized_counts_high)
    
    custom_font = mpl.font_manager.FontProperties(fname='D:\workspace\Spyder\LDP/Fonts/华文细黑.ttf')
    fig_size = (10, 6) # 图表大小    
    
    names = ('Real', 'Low Eps', 'Mid Eps', 'High Eps')
    epsilons = location_real_counts.sort_values(ascending=0)[0:10].index           
    
    # 更新图表大小
    mpl.rcParams['figure.figsize'] = fig_size
    # 设置柱形图宽度
    bar_width = 0.2

    index = np.arange(10)

    rects1 = plt.bar(index, location_real_counts.sort_values(ascending=0)[0:10].values, bar_width, edgecolor='k', label=names[0])

    rects2 = plt.bar(index + bar_width, location_randomized_counts_low.sort_values(ascending=0)[0:10].values, bar_width, color='r', edgecolor='k', hatch='//', label=names[1])
    
    rects3 = plt.bar(index + bar_width*2, location_randomized_counts_mid.sort_values(ascending=0)[0:10].values, bar_width, color='g', edgecolor='k', hatch='o', label=names[2])
    
    rects4 = plt.bar(index + bar_width*3, location_randomized_counts_high.sort_values(ascending=0)[0:10].values, bar_width, color='b', edgecolor='k', hatch='.', label=names[3])
    
    # rects4 = plt.bar(index + bar_width*3, location_randomized_counts_high.sort_values(ascending=0)[0:10].values, bar_width, color='k', label=names[3])
    
    plt.xticks(index + bar_width, epsilons, fontproperties=custom_font)
    plt.ylim(ymin=0)
    
    plt.title('汇聚结果对比', fontproperties=custom_font)
    plt.xlabel('location ID')
    plt.ylabel('User Counting')
    
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=5, prop=custom_font)
    plt.legend(loc='upper right', fancybox=True, ncol=5, prop=custom_font)

#    add_labels(rects1)
#    add_labels(rects2)
#    add_labels(rects3)
#    add_labels(rects4)

    plt.tight_layout()     
    plt.show()     
    # plt.savefig('Count_RPSB.tiff')

          
        
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 17:03:12 2022

@author: Ian
"""



import sys
import pandas as pd
from sklearn import metrics
from keras.utils import plot_model
import numpy as np 
import os
from keras.models import Model, load_model
import matplotlib.pyplot as plt
import argparse
import matplotlib
matplotlib.use('Agg')


parser = argparse.ArgumentParser(description='Mynet') 
parser.add_argument('--m6a_seq', type=str, default='m6a_seq_test.npy', help='file path of m6a_seq')
parser.add_argument('--m6a_str', type=str, default='m6a_str_test.npy', help='file path of m6a_str')
parser.add_argument('--label', type=str, default='m6a_label_test.npy', help='file path of label')

parser.add_argument('--data', type=str, default='data_101', help='data source')
parser.add_argument('--parameter', type=str, default='motif', help='Optimizing parameters ')
args = parser.parse_args()  # ArgumentParser 通过 parse_args() 方法解析参数

# 创建存放结果的文件夹
file_path='./result/'+ args.data +'/'+ args.parameter +'/'
if not os.path.exists(file_path):                   # 如果不存在则创建一个新的文件夹来存储文件
 	os.makedirs(file_path)


## load data: sequence
m6a_seq_data = np.load('./data/data_101/datanpy/' + args.m6a_seq)
m6a_str_data = np.load('./data/data_101/datanpy/' + args.m6a_str)
label =  np.load('./data/data_101/datanpy/' + args.label)

# 载入模型
model=load_model('./result/data_101/cnn_128_64/model_2.h5') # 载入每个数据包的模型
#print(model.layers)#输出所有层

# 将想要的层作为一个模型
#layer_model = Model(inputs=model.input, outputs=model.layers[2].output)
layer_model = Model(inputs=model.input, outputs=model.get_layer('m6a_seq_con_1st').output) #按名称
# 输出当前模型的结果
m6a_seq_output = layer_model.predict([m6a_seq_data,m6a_str_data]) # score为预测概率

print(m6a_seq_output.shape)



## 
# 存放最终的结果(128*11*4)
print("find motif")

sub_seq_sum_all=[]
sub_seq_sum_fre_all=[]

for i in range(128):
    
    cnn_kernel=m6a_seq_output[:,:,i] #一个卷积核的所有输入响应（2044,111），i一共有128个
    row_max_index=np.argmax(cnn_kernel, axis=1) # 求索引，每个样本一个索引
    row_max=np.max(cnn_kernel,axis=1) #每个样本的最大值
    
    #创建一个全为0.25和全为0的数组（11*4）
    seq_N = np.ones((11,4)) * 0.25
    sub_seq_sum = np.zeros((11,4))
    
    #循环样本
    count=0 #最大值为0的数
    for j in range(len(row_max_index)): # 
        if(row_max[j]==0):
            count+=1
            # print("最大值为0")
            continue
            
        if(row_max_index[j]<5):
            sub_seq=np.vstack((seq_N[:5-row_max_index[j],:],m6a_seq_data[j,0:row_max_index[j]+6,:]))
            sub_seq_sum+=sub_seq
            
        elif(row_max_index[j]>95):
            sub_seq=np.vstack((m6a_seq_data[j,row_max_index[j]-5:,:],seq_N[:row_max_index[j]-95,:]))
            sub_seq_sum+=sub_seq        
            
        else:
            sub_seq=m6a_seq_data[j,row_max_index[j]-5:row_max_index[j]+6,:]
            sub_seq_sum+=sub_seq
        
    # 对 sub_seq_sum 求概率
    if(count!=0):print("第",i,"个卷积核子序列数，有： ",len(label)-count)
    sub_seq_sum_fre=sub_seq_sum/(len(label)-count)
    
    if(len(sub_seq_sum_all)==0):
        sub_seq_sum_all=np.expand_dims(sub_seq_sum,axis=0)
        sub_seq_sum_fre_all=np.expand_dims(sub_seq_sum_fre,axis=0)
    else:
        sub_seq_sum_all=np.vstack((sub_seq_sum_all,np.expand_dims(sub_seq_sum,axis=0)))
        sub_seq_sum_fre_all=np.vstack((sub_seq_sum_fre_all,np.expand_dims(sub_seq_sum_fre,axis=0)))
        
# np.save("sub_seq_sum_all.npy",sub_seq_sum_all)
# np.save("sub_seq_sum_fre_all.npy",sub_seq_sum_fre_all)

print("end")        



## 创建motif_comp.txt文档
print("创建motif_comp.txt文档")  
f=file_path+'/motif_comp_RNA.txt'

count=0
while(os.path.exists(f)):                   # 如果文件已存在就重新创建一个（防止追加错误）
    count+=1
    f="./motif_comp_"+str(count)+".txt"
    
if(count!=0):
    print("当前文档已存在，重新改名为: motif_comp_"+str(count))

with open(f,"a+") as file:
    # 文件头
    file.write("MEME version 4\n")
    file.write("\n")
    file.write("ALPHABET= ACGU\n")   #  ****************区别RNA：ACGU 与DNA：ACGT***************  
    file.write("\n")
    file.write("strands: + -\n")
    file.write("\n")
    file.write("Background letter frequencies\n")
    file.write("A 0.25000 C 0.25000 G 0.25000 U 0.25000 \n")  #  ****************替换 T 与 U ***************  
    file.write("\n")
    
    for i in range(len(sub_seq_sum_fre_all)):
    #for i in range(2):    
        file.write("MOTIF kernel_"+str(i+1) +"\n")
        file.write("letter-probability matrix: alength= 4 w= 9 nsites= 20 E= 0\n")
        for j in range(len(sub_seq_sum_fre_all[0])):
            file.write(str(round(sub_seq_sum_fre_all[i][j][0],6))+"\t")
            file.write(str(round(sub_seq_sum_fre_all[i][j][2],6))+"\t")
            file.write(str(round(sub_seq_sum_fre_all[i][j][1],6))+"\t")
            file.write(str(round(sub_seq_sum_fre_all[i][j][3],6))+"\n")
        file.write("\n")
file.close()         
print("创建完毕") 
##

'''
# 返回a中每行最大值
np.max(a,axis=1)

# 返回a中每行最大值的索引
c=np.argmax(a, axis=1)

#创建一个全为0.25的数组（9*4）
seq_N = np.ones((9,4)) * 0.25

#数组垂直拼接
np.vstack((seq_N[:4,:],a))

# 增加维度
b=np.expand_dims(seq_N,axis=0)

# 三维拼接

##

# 保存前6位小数
a=0.1234567
round(a,6)

'''










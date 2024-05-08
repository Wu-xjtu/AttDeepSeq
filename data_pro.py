# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 16:01:31 2021

@author: Ian

提取以DRACH为中心的正负样本
其中 D = A , G , U   R = A , G , H = A, C, U)

"""

import argparse
import os
import pandas as pd
import numpy as np
from pyfasta import Fasta
from random import shuffle
from random import sample
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='parameters') 
parser.add_argument('--minidata', type=str, default='False', help='是否为小样本数据')
parser.add_argument('--length', type=int, default=151, help='ipsize')
parser.add_argument('--path_to_save', type=str, default='data_151', help='data source')
args = parser.parse_args()  # ArgumentParser 通过 parse_args() 方法解析参数
length=args.length



# 定义DRACH序列,判断是否符合,符合返回True
def is_DRACH(seq):
    if seq[0]!='A' and seq[0]!='a' and seq[0]!='G' and seq[0]!='g' and seq[0]!='T' and seq[0]!='t':
        return False
    if seq[1]!='A' and seq[1]!='a' and seq[1]!='G' and seq[1]!='g':
        return False
    if seq[2]!='A' and seq[2]!='a':
        return False
    if seq[3]!='C' and seq[3]!='c':
        return False     
    if seq[4]!='A' and seq[4]!='a' and seq[4]!='C' and seq[4]!='c' and seq[4]!='T' and seq[4]!='t':
        return False    
    return True

# 检查序列是否以DRACH序列为中心
def is_DRACH_seq(seq):
    mid=int((length-1)/2)
    DRACH=seq[mid-2:mid+3]
    if is_DRACH(DRACH):
        return True
    else:
        print('当前序列中间不符合DRACH基序',DRACH)
        return False
        

data_file='./data/'
if args.minidata=='True':
    data_file='./data_0/'    
print('数据存储路径:',data_file)

    
file_paths=data_file + args.path_to_save +'/datanpy/'  ## 创建./data/data_101/datanpy
if not os.path.exists(file_paths):                   
 	os.makedirs(file_paths)
     
file_path=data_file + args.path_to_save +'/'  #data1负样本来源路径(不舍阈值时的补集)




# step 1 裁剪正负样本  
"""
## hg19参考基因组
#f = Fasta('E:/m6a/reference/hg19.fa')
f = Fasta('/data1/yanhao/DeepTACT/hg19.fa')                     # 参考基因组hg19


## 正样本裁剪
## 提取长度为length的正样本写入data.fa
count1=0 
count0=0
data1_path='./raw_data/hs_data/data_all_sorted_1w.bed'
filename='data.fa'           #***************** 文件名称*********************
f1 = open(file_path+filename,'w')        
data0=[]

input = open(data1_path)
for line in input:
    row = line.split()
    chrom=row[0]
    start=int(row[1])
    end=int(row[2])
    strand=row[5]
    # 
    left=end-int((length-1)/2)
    right=end+int((length-1)/2)

    source=chrom+':'+str(start)+'-'+str(end)
    
    # 判断中间位置是否符合DRACH基序
    seq=f.sequence({'chr': chrom, 'start': left, 'stop': right, 'strand': strand}) #截取时包含start和stop位点
    
    # 正样本
    substr=seq[int((length-1)/2-2):int((length-1)/2+3)]
    if is_DRACH(substr)and not('N' in seq or 'n' in seq )  : # 查看seq中是否有N或者n,或不符合基序,如果有则输出，没有则写入
        check=is_DRACH_seq(seq)
        region=chrom+': ' + str(left)+'-'+str(right)+' from '+ source
        f1.write('>label 1 ' +region +'\n')  # 
        f1.write(seq + '\n')            
        count1+=1
    #else:
    #    print('当前位点不符合DRACH: ',source+' '+substr)
        
    # 负样本
    seq_0=f.sequence({'chr': chrom, 'start': end+length, 'stop': end+length+200, 'strand': strand})
    for i in range(2,len(seq_0)-2):
        if (seq_0[i]=='A' or seq_0[i]=='a') and (seq_0[i+1]=='C' or seq_0[i+1]=='c'):
            motif=seq_0[i-2:i+3]
            if is_DRACH(motif):
                # print(motif)
                #重新裁剪
                l=end+length+i-int((length-1)/2)
                r=end+length+i+int((length-1)/2)
                if strand=='-':  ## 新加
                    l=end+length+200-i-int((length-1)/2)
                    r=end+length+200-i+int((length-1)/2)
                seq_0_DRACH=f.sequence({'chr': chrom, 'start': l, 'stop': r, 'strand': strand})
                
                check=is_DRACH_seq(seq_0_DRACH)  ## 核对
                # print(seq_0_DRACH)
                
                region_0= chrom+': ' + str(l)+'-'+str(r)+' from '+ source
                data0.append('>label 0 ' +region_0)
                data0.append(seq_0_DRACH)
                #f0.write('>label 0 ' +region_0 +'\n')   
                #f0.write(seq_0_DRACH + '\n')
                count0+=1
                break
input.close()
f1.close()
#f0.close()
print('data_1的序列数量：',count1) #  
print('data_0的序列数量：',count0) #  

f1 = open(file_path+filename,'a') # 将负样本追加
for i in range(len(data0)):
        f1.write(data0[i] + '\n')
f1.close()
"""

"""
# step 2 去冗余及获取二级结构
# 利用 CD-HIT-EST 去除冗余
# data.fa --> 1637546831.fas.1
## 获取二级结构(RNAfold < 1648035043.fas.1 > data.txt)
os.system('RNAfold < '+ file_path+'1637546831.fas.1 > '+ file_path+'data.txt')
"""


# step 3 one_hot编码
## 提取序列、二级结构、以及标签
"""
"""

str_txt= open(file_path+'data.txt','r')   #  *****************注意修改 *****************
str_ = str_txt.readlines()

## 将1中 的数据按顺序打包， 然后选取和负样本一样的数据，最后再合并在一起
m6a_seq_0=[]
m6a_str_0=[]
label_0=[]

m6a_seq_1=[]
m6a_str_1=[]
label_1=[]

i=0
while(i<len(str_)/3):
    
    label=str_[3*i].split(' ')[1]
    if label=='1':
        m6a_seq_1.append(str_[3*i+1].split('\n')[0])
        m6a_str_1.append(str_[3*i+2].split(' ')[0])
        label_1.append(1)
    else:
        m6a_seq_0.append(str_[3*i+1].split('\n')[0])
        m6a_str_0.append(str_[3*i+2].split(' ')[0])
        label_0.append(0)
    i+=1

print("正样本数量：",len(label_1))
print("负样本数量：",len(label_0))  
    
## 平衡训练集和测试集（正负样本对其）
min_num = len(m6a_seq_0) if len(m6a_seq_0)<len(m6a_seq_1) else len(m6a_seq_1)
print('向下对齐值：',min_num)

m6a_seq_0=m6a_seq_0[:min_num]
m6a_str_0=m6a_str_0[:min_num]
label_0=label_0[:min_num]
m6a_seq_1=m6a_seq_1[:min_num]
m6a_str_1=m6a_str_1[:min_num]
label_1=label_1[:min_num]


m6a_seq=m6a_seq_1+m6a_seq_0
m6a_str=m6a_str_1+m6a_str_0
m6a_label=label_1+label_0

m6a_label=np.asarray(m6a_label).astype('int32') 



## 编码所需函数

def base_to_onehot(base):
    if base == 'A' or base=='a':
        return np.array([1, 0, 0, 0])
    elif base == 'G' or base=='g':
        return np.array([0, 1, 0, 0])
    elif base == 'C' or base=='c':
        return np.array([0, 0, 1, 0])
    # elif base == 'T' or base=='t' or base=='U' or base=='u' : # 
    elif base=='U' or base=='u' :
        return np.array([0, 0, 0, 1])
    else:
        print('error') # 如果出错则输出
        
def str_to_onehot(base):
    if base == '.':
        return np.array([1, 0, 0])
    elif base == '(':
        return np.array([0, 1, 0])
    elif base == ')':
        return np.array([0, 0, 1])
    else:
        print('出现新的编码：',base)        

def onehot_seq(data):
    all_samples = np.zeros((len(data), length, 4))
    for i in range(len(data)):
        current_sample = np.zeros((length, 4))
        if i % 10000 == 0:                    # 服务器
            print('processing\t: ' + str(i))
        for j in range(len(data[i])):
            current_sample[j] = base_to_onehot(data[i][j])
        all_samples[i] = current_sample
    return all_samples

def onehot_str(data):
    all_samples = np.zeros((len(data), length, 3))
    for i in range(len(data)):
        current_sample = np.zeros((length, 3))
        if i % 10000 == 0:                    # 服务器
            print('processing\t: ' + str(i))
        for j in range(len(data[i])):
            current_sample[j] = str_to_onehot(data[i][j])
        all_samples[i] = current_sample
    return all_samples


## seq & str编码
print('编码m6a_seq')
m6a_seq_onehot=onehot_seq(m6a_seq)
print('编码m6a_str')
m6a_str_onehot=onehot_str(m6a_str)


## 划分训练集和测试集

print('打包中...')
mix = list(zip(m6a_seq_onehot, m6a_str_onehot, m6a_label))

print('打乱中...')
shuffle(mix)

print('拆包中...')
m6a_seq_onehot, m6a_str_onehot, m6a_label = zip(*mix)
print('拆包完成。')

if args.minidata==True:
    num_data=20000
else:
    num_data = len(m6a_label) 
    
number=int(num_data*0.9)

# 训练集
m6a_seq_train=m6a_seq_onehot[:number]
m6a_str_train=m6a_str_onehot[:number]
m6a_label_train=m6a_label[:number]

# 测试集
m6a_seq_test=m6a_seq_onehot[number:num_data]
m6a_str_test=m6a_str_onehot[number:num_data]
m6a_label_test=m6a_label[number:num_data]

## 保存
np.save(file_paths+'m6a_seq_train.npy',m6a_seq_train)
np.save(file_paths+'m6a_str_train.npy',m6a_str_train)
np.save(file_paths+'/m6a_label_train.npy',m6a_label_train)

np.save(file_paths+'m6a_seq_test.npy',m6a_seq_test)
np.save(file_paths+'m6a_str_test.npy',m6a_str_test)
np.save(file_paths+'/m6a_label_test.npy',m6a_label_test)

print('The number of train :',len(m6a_label_train))
print('The number of test :',len(m6a_label_test))






##  选取部分数据进行length优化
## 提取部分数据

"""
data1_path='./raw_data/hs_data/data_all_sorted.bed'     # 总数：59234
f_out = open('./raw_data/hs_data/data_all_sorted_1w.bed','w')  

count=0
input = open(data1_path)
for line in input:
    count+=1
    if count==1 or count%5==0:
        f_out.write(line)

input.close()
f_out.close()
print("处理结束")
"""













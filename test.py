# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:10:54 2021

@author: Ian
"""

import numpy as np 
import pandas as pd
from sklearn import metrics
from keras.utils import plot_model
from random import shuffle
import os
from keras.models import Model, load_model
import argparse
from pyfasta import Fasta

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


## step1 截取序列

parser = argparse.ArgumentParser(description='parameters') 
parser.add_argument('--minidata', type=str, default='False', help='是否为小样本数据')
parser.add_argument('--length', type=int, default=101, help='ipsize')

parser.add_argument('--path_to_save', type=str, default='RMBase_motif_score_349', help='data source/文件夹名称')
parser.add_argument('--file_name', type=str, default='RMBase_motif_score_349_k.txt', help='原始文件名称') 
parser.add_argument('--fa_name', type=str, default='data.fa', help='根据原始文件提取.fa文件') 
# 去冗余
parser.add_argument('--red_fa_name', type=str, default='1641818919.fas.1', help='（未）去冗余.fa文件') 
parser.add_argument('--balance_fa_name', type=str, default='test_motif_score_349.fa', help='调整比例后的.fa文件') 
parser.add_argument('--label', type=str, default='1', help='调整比例时隔行取正/负样本')
# 二级结构编码
parser.add_argument('--str_out', type=str, default='test_motif_score_349.txt', help='二级结构编码的输出名称')

args = parser.parse_args()  # ArgumentParser 通过 parse_args() 方法解析参数
length=args.length


'''
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

## hg19参考基因组
#f = Fasta('E:/m6a/reference/hg19.fa')
f = Fasta('/data1/yanhao/DeepTACT/hg19.fa')                     # 参考基因组hg19


## 正样本裁剪
## 提取长度为length的正样本写入data.fa
count1=0 
count0=0
data1_path='./raw_data/hs_data/'+args.file_name
filename= args.fa_name         #***************** 文件名称*********************
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
        f1.write(seq.upper() + '\n')            
        count1+=1
    else:
        print('当前位点不符合DRACH: ',source+' '+substr)
        
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
                
                if is_DRACH_seq(seq_0_DRACH) and not('N' in seq_0_DRACH or 'n' in seq_0_DRACH ):## 核对
                    region_0= chrom+': ' + str(l)+'-'+str(r)+' from '+ source
                    data0.append('>label 0 ' +region_0)
                    data0.append(seq_0_DRACH.upper())
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
'''


## step2 去冗余
"""
手动去冗余   
input: args.fa_name 
output:args.(un)red_fa_name
"""

'''
## step 3 正负样本比例设为1:1 , 同时将seq_name替换为更短的
head_inf=[]
seq=[]

data1_path='./data/'+ args.path_to_save +'/' + args.red_fa_name     # *******************（未）去冗余的.fa文件************************
f_out = open('./data/'+ args.path_to_save +'/'+ args.balance_fa_name,'w') # ***********调整比例后的.fa文件**********************

input = open(data1_path)
for line in input:
    if line[0]=='>':
        head_inf.append(line)
    else:
        seq.append(line)


count1=0
count_pos=0
count_neg=0
for i in range(len(seq)):
    label=head_inf[i].split()[1]
    if label==args.label:
        count1+=1
        if count1 % 1==0:
            count_pos+=1
            f_out.write(head_inf[i][:9]+'seq_'+str(count_pos)+'\n')
            f_out.write(seq[i].upper())
    else:
        count_neg+=1
        f_out.write(head_inf[i][:9]+'seq_'+str(count_neg)+'\n')
        f_out.write(seq[i].upper()) 
        
if args.label=='1':
    print('最终正样本：',count_pos)
    print('最终负样本：',count_neg)
else:
    print('最终负样本：',count_pos)
    print('最终正样本：',count_neg)       

f_out.close()
'''


## 利用RNAfold获得二级结构
"""
RNAfold < test_motif_score_349.fa > test_motif_score_349.txt
RNAfold <args.balance_fa_name > args.str_out    #
"""


## 将测试数据进行编码(用与自身模型)

data_file='./data/'
if args.minidata=='True':
    data_file='./data_0/'    
print('数据存储路径:',data_file)

    
file_paths=data_file + args.path_to_save +'/datanpy/'  ## 创建./data/data_101/datanpy
if not os.path.exists(file_paths):                   
 	os.makedirs(file_paths)
     
file_path=data_file + args.path_to_save +'/'  #data1负样本来源路径(不舍阈值时的补集)


## 提取序列、二级结构、以及标签

str_txt= open(file_path + args.str_out,'r')  # **************调整比例后获取二级结构的.txt文件**********************
str_ = str_txt.readlines()

    
m6a_seq=[]
m6a_str=[]
label=[]

i=0
while(i<len(str_)/3):
    
    label.append(str_[3*i].split(' ')[1])
    m6a_seq.append(str_[3*i+1].split('\n')[0])
    m6a_str.append(str_[3*i+2].split(' ')[0])
    i+=1
    
m6a_label=np.asarray(label).astype('int32') 



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
        print('error:', base) # 如果出错则输出
        
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



# 测试集
m6a_seq_test=m6a_seq_onehot
m6a_str_test=m6a_str_onehot
m6a_label_test=m6a_label
## 保存

np.save(file_paths+'m6a_seq_test.npy',m6a_seq_test)
np.save(file_paths+'m6a_str_test.npy',m6a_str_test)
np.save(file_paths+'/m6a_label_test.npy',m6a_label_test)


print('The number of test :',len(m6a_label_test))
















'''
## 筛选出一定量的数据
input_file='./raw_data/hs_data/RMBase_motif_score_349.txt'     # *******************（未）去冗余的.fa文件************************
output_file = open('./raw_data/hs_data/RMBase_motif_score_349_k.txt','w') # ***********调整比例后的.fa文件**********************

input = open(input_file)
count=0
for line in input:
    if count% 10 ==0:
        output_file.write(line)
    count+=1
    

input.close()
output_file.close()
print("筛选结束")
'''
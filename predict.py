# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:51:57 2021

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

parser.add_argument('--minidata', type=str, default='False', help='是否为小样本数据')
parser.add_argument('--data', type=str, default='data_101', help='data source')
parser.add_argument('--parameter', type=str, default='cnn_128_64_01', help='记录改变的参数，用以存储结果')
args = parser.parse_args()  # ArgumentParser 通过 parse_args() 方法解析参数


data_path='./data/'
result_path='./result/'

if args.minidata=='True':
    result_path='./result_0/'
    data_path='./data_0/'


# 创建存放结果的文件夹
file_path=result_path + args.data +'/'+ args.parameter +'/'
if not os.path.exists(file_path):                   # 如果不存在则创建一个新的文件夹来存储文件
 	os.makedirs(file_path)


## load data: sequence
m6a_seq_data = np.load(data_path + args.data +'/datanpy/'   + args.m6a_seq)
m6a_str_data = np.load(data_path + args.data +'/datanpy/'  + args.m6a_str)
label =  np.load(data_path + args.data +'/datanpy/'  + args.label)


model=load_model(file_path +'model_2.h5') # 载入每个数据包的模型
# model.summary()
score = model.predict([m6a_seq_data,m6a_str_data]) # score为预测概率
# score = model.predict(m6a_seq_data)
pred = (score > 0.5).astype(int) # 以0.5为阈值划分score作为预测值，数据格式转化为int,最后转化为一行
acc=metrics.accuracy_score(label, pred)
print('acc:', acc)

fpr, tpr, threshold = metrics.roc_curve(label, score)
roc_auc = metrics.auc(fpr, tpr)
print('roc_auc:',roc_auc)

auprc = metrics.average_precision_score(label, score)
print('auprc:',auprc)


confusion = metrics.confusion_matrix(label,pred)
f1_score = metrics.f1_score(label,pred)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
# print('Accuracy:',(TP+TN)/float(TP+TN+FP+FN))
Sensitivity=TP / float(TP+FN)
Specificity=TN / float(TN+FP)
print('Sensitivity:',Sensitivity)
print('Specificity:',Specificity) 
print('f1_score:',f1_score) 


# plt.title('Validation ROC')s
# plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.savefig(file_path +'Test_Roc.png')
# # plt.show()


# precision, recall, threshold = metrics.precision_recall_curve(label, score)
# roc_aupr = metrics.auc(recall, precision)

# plt.clf()
# plt.plot([0, 1], [0, 1], 'k--',label='ROC curve (area = %0.2f)' % roc_aupr)
# plt.plot(recall, precision)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('ROC_PR curve')
# plt.legend(loc='best')
# plt.savefig(file_path +'Test_aupr.png')
# plt.show()



## 结果存入csv文件

# filename = result_path+'struct.csv' # 查看是否存在储存整个结果的文件夹，如果没有则新建一个并储存当前结果

# if not os.path.exists(filename):
#     # os.makedirs('./genes_result/')
#     result_all =pd.DataFrame(columns=('parameter','acc','roc_auc','auprc','sen','spe','f1'))# 结果储存
#     result_all = result_all.append([{'parameter':args.parameter,'acc':acc,'roc_auc':roc_auc,
#                                       'auprc':auprc,'sen':Sensitivity,'spe':Specificity,'f1':f1_score}], ignore_index=True)
#     result_all.to_csv(filename,index=False)  #保存信息头

# # 如果存在文佳，打开进行存储
# else:
#     result_all=pd.read_csv(filename)
#     result_all = result_all.append([{'parameter':args.parameter,'acc':acc,'roc_auc':roc_auc,
#                                       'auprc':auprc,'sen':Sensitivity,'spe':Specificity,'f1':f1_score}], ignore_index=True)
    
#     result_all.to_csv(filename,index=False)  #合并数据存入CSV文件  









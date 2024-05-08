# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 15:28:53 2021

@author: Ian
"""
import numpy as np 
import pandas as pd
from sklearn import metrics
from keras.utils import plot_model
import os
from keras.models import Model, load_model
import argparse

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

'''
command line
python base_test.py --model_path=./result/data_101//
'''

## base 模型性能检测
parser = argparse.ArgumentParser(description='Mynet') 
parser.add_argument('--m6a_seq', type=str, default='m6a_seq_test.npy', help='file path of m6a_seq')
parser.add_argument('--m6a_str', type=str, default='m6a_str_test.npy', help='file path of m6a_str')
parser.add_argument('--label', type=str, default='m6a_label_test.npy', help='file path of label')

parser.add_argument('--model_path', type=str, default='./result/data_101/cnn_128_64/' , help='mode_path')
parser.add_argument('--data_path', type=str, default='./data/hek293_sysy_5508/datanpy/' , help='data_path')
parser.add_argument('--result_path', type=str, default='./result/hek293_sysy_5508/base/', help='result_path')
args = parser.parse_args()  # ArgumentParser 通过 parse_args() 方法解析参数

# 模型来源路径
file_path=args.model_path

if not os.path.exists(args.result_path):                   
 	os.makedirs(args.result_path)


## 数据来源
m6a_seq_data = np.load( args.data_path + args.m6a_seq)
m6a_str_data = np.load( args.data_path + args.m6a_str)
label =  np.load( args.data_path + args.label)


model=load_model(file_path +'model_2.h5') # 载入每个数据包的模型
score = model.predict([m6a_seq_data,m6a_str_data]) # score为预测概率
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









# plt.title('Validation ROC')
# plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.savefig(args.result_path +'test_Roc.png')
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
# plt.savefig(args.result_path +'test_PR.png')
# plt.show()



## 结果存入csv文件
'''
filename = './result/parameter.csv' # 查看是否存在储存整个结果的文件夹，如果没有则新建一个并储存当前结果
if not os.path.exists(filename):
    # os.makedirs('./genes_result/')
    result_all =pd.DataFrame(columns=('parameter','acc','roc_auc','auprc'))# 结果储存
    result_all = result_all.append([{'parameter':'test','acc':acc,'roc_auc':roc_auc,
                                     'auprc':auprc}], ignore_index=True)
    result_all.to_csv(filename,index=False)  #保存信息头

# 如果存在文佳，打开进行存储
else:
    result_all=pd.read_csv(filename)
    result_all = result_all.append([{'parameter':'test','acc':acc,'roc_auc':roc_auc,
                                     'auprc':auprc}], ignore_index=True)
    
    result_all.to_csv(filename,index=False)  #合并数据存入CSV文件   
'''
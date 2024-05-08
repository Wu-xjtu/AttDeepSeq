# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 21:14:30 2022

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


parser = argparse.ArgumentParser(description='Mynet') 
parser.add_argument('--m6a_seq', type=str, default='m6a_seq_test.npy', help='file path of m6a_seq')
parser.add_argument('--m6a_str', type=str, default='m6a_str_test.npy', help='file path of m6a_str')
parser.add_argument('--label', type=str, default='m6a_label_test.npy', help='file path of label')

parser.add_argument('--model_path_1', type=str, default='./result/data_101/cnn_128_64/' , help='mode_path1')
parser.add_argument('--file_path1', type=str, default='test_balance_red_full.txt' , help='mode_path2')
parser.add_argument('--file_path2', type=str, default='test_balance_red_mature.txt' , help='mode_path3')
parser.add_argument('--file_path3', type=str, default='predout_test_balance_red_cnn.tsv' , help='mode_path4')
parser.add_argument('--file_path4', type=str, default='predout_test_balance_red_rnn.tsv' , help='mode_path4')

parser.add_argument('--data_path', type=str, default='./data/hek293_sysy_5508/datanpy/' , help='data_path')

args = parser.parse_args()  # ArgumentParser 通过 parse_args() 方法解析参数


## 数据来源
seq_data = np.load( args.data_path + args.m6a_seq)
str_data = np.load( args.data_path + args.m6a_str)
label =  np.load( args.data_path + args.label)



def multi_models_roc(names, sampling_methods, linestyles, colors, file_paths, X_test, str_data, y_test, save=True, dpin=100):
    """
    将多个机器模型的roc图输出到一张图上
    
    
    Args:
        names: list, 多个模型的名称
        sampling_methods: list, 多个模型的实例化对象
        save: 选择是否将结果保存（默认为png格式）
        
    Returns:
        返回图片对象plt
    """
    plt.figure(figsize=(10, 10), dpi=dpin)

    for (name, method, colorname, linestyle, file_path) in zip(names, sampling_methods, colors, linestyles, file_paths):
        
        fpr=[]
        tpr=[]
        
        if name=='Our method':
            y_test_predprob = method.predict([X_test,str_data])
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_predprob, pos_label=1)
        elif name=='SRAMP_full' or name=='SRAMP_mature':
            fpr, tpr=SRAMP(file_path,'ROC')
        else:
            fpr, tpr=DeepM6Aseq(file_path,'ROC')
        
        plt.plot(fpr, tpr, linestyle=linestyle, lw=3, label='{} (AUC={:.3f})'.format(name, metrics.auc(fpr, tpr)),color = colorname)
        plt.plot([0, 1], [0, 1], '--', lw=2, color = 'grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('False Positive Rate',fontsize=20)
        plt.ylabel('True Positive Rate',fontsize=20)
        plt.title('ROC Curve',fontsize=20)
        plt.legend(loc='lower right',fontsize=20)

    if save:
        plt.savefig('multi_models_roc.png')
        
    return plt


def multi_models_prc(names, sampling_methods, linestyles, colors, file_paths, X_test, str_data, y_test, save=True, dpin=100):
    """
    将多个机器模型的prc图输出到一张图上
    
    Args:
        names: list, 多个模型的名称
        sampling_methods: list, 多个模型的实例化对象
        save: 选择是否将结果保存（默认为png格式）
        
    Returns:
        返回图片对象plt
    """
    plt.figure(figsize=(10, 10), dpi=dpin)

    for (name, method, colorname,linestyle, file_path) in zip(names, sampling_methods, colors, linestyles, file_paths):
        

        if name=='Our method':
            y_test_predprob = method.predict([X_test,str_data])
            precision, recall, threshold = metrics.precision_recall_curve(y_test, y_test_predprob)
        elif name=='SRAMP_full' or name=='SRAMP_mature':
            precision, recall=SRAMP(file_path,'prc')
        else:
            precision, recall=DeepM6Aseq(file_path,'prc')        
        
        plt.plot(recall, precision, linestyle=linestyle,lw=3, label='{} (PRC={:.3f})'.format(name, metrics.auc(recall, precision)),color = colorname)
        # plt.plot([0, 1], [0, 1], '--', lw=5, color = 'grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Recall',fontsize=20)
        plt.ylabel('Precision',fontsize=20)
        plt.title('PR Curve',fontsize=20)
        plt.legend(loc='lower right',fontsize=20)

    if save:
        plt.savefig('models_prc.png')
        
    return plt


def DeepM6Aseq(input_file,para):
    
    result=pd.DataFrame(columns=('seq_num','score','label','pred'))
    test_output= open('../DeepM6ASeq/demo_test/'+input_file,'r')
    lines = test_output.readlines()
    
    for i in range(len(lines)):
        line=lines[i].split('\t')
        seq_num=line[0]
        label=line[0].split(' ')[1]
        score=float(line[1].split('\n')[0])
        pred= 0 if score < 0.5 else 1
        
        result=result.append([{'seq_num':seq_num,'score':score,'label':label ,'pred':pred}], ignore_index=True)  
                
    test_output.close()
    
    
    data_types_dict = {'label': int,'pred': int}   
    result= result.astype(data_types_dict)             
    # result.to_csv('m6a_out_full.csv',index=False)
        
    label=result['label'].values
    pred=result['pred'].values
    score=result['score'].values

    fpr, tpr, threshold = metrics.roc_curve(label, score)
    precision, recall, threshold = metrics.precision_recall_curve(label, score)
    if para=='prc':
        fpr=precision
        tpr=recall
    return fpr,tpr

def SRAMP(input_file,para):

    result=pd.DataFrame(columns=('seq_num','score','label','pred','calssification'))
    test_output= open('../sramp_simple/'+input_file,'r')
    lines = test_output.readlines()
    
    seq_num_last='label 1 seq_1'    # 为第一个序列号
    score_max=0.0
    pred_final=0
    class_final=' '
    
    for i in range(len(lines)):
        if (i>0):
            line=lines[i].split('\t')
            
            # 当前行序列号,分数，实际分类情况，预测值
            seq_num=line[0]
            score_now=float(line[4])
            class_now=line[5].split('\n')[0]
            pred_now=0
            if(class_now[:8]=='m6A site'):
                pred_now=1        
            
            if(seq_num==seq_num_last): #属于同一个序列则比较大小
                if(score_now > score_max):
                   score_max=score_now
                   pred_final=pred_now
                   class_final=class_now
                
            if seq_num!=seq_num_last : #遇到新的序列或者最后一行，写入序列号，分数，分类然后更新last
                result=result.append([{'seq_num':seq_num_last,'score':score_max,'label':seq_num_last.split(' ')[1] ,
                                       'pred':pred_final,'calssification':class_final}], ignore_index=True)
                seq_num_last=seq_num
                score_max=score_now
                pred_final=pred_now
                class_final=class_now 
      
    result=result.append([{'seq_num':seq_num_last,'score':score_max,'label':seq_num_last.split(' ')[1] ,
                                       'pred':pred_final,'calssification':class_final}], ignore_index=True)              
    test_output.close()
    
    
    data_types_dict = {'label': int,'pred': int}   
    result= result.astype(data_types_dict)             
    # result.to_csv('m6a_out_full.csv',index=False)
    

    label=result['label'].values
    pred=result['pred'].values
    score=result['score'].values
    # pred = (score > 0.5).astype(int)
    
    ## 判断指标
    fpr, tpr, threshold = metrics.roc_curve(label, score)
    precision, recall, threshold = metrics.precision_recall_curve(label, score)
    if para=='prc':
        fpr=precision
        tpr=recall
    return fpr,tpr



model1=load_model(args.model_path_1+'model_2.h5') # 载入每个数据包的模型



names = ['Our method',
         'SRAMP_full',
         'SRAMP_mature',
         'DeepM6Aseq_cnn',
         'DeepM6Aseq_rnn'
         ]

file_paths = ['',
           args.file_path1,
           args.file_path2,
           args.file_path3,
           args.file_path4,
           ]

sampling_methods = [model1,
                    model1,
                    model1,
                    model1,
                    model1
                   ]

colors = ['red',
          'orange',
          'blue',
          'mediumseagreen',
          # 'steelblue', 
          'mediumpurple'  
         ]

linestyles=['-','-',':','-.','-']

#ROC curves
train_roc_graph = multi_models_roc(names, sampling_methods, linestyles, colors, file_paths, seq_data, str_data,  label, save = True)
train_roc_graph.savefig('./result/ROC_compare_all.png')

train_prc_graph = multi_models_prc(names, sampling_methods, linestyles, colors, file_paths, seq_data, str_data, label, save = True)
train_prc_graph.savefig('./result/PRC_compare_all.png')

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:27:52 2021
修改：
conv1D:卷积核从1024改为128（两个）
全连接层（925）改为64
batch_size改为128

@author: Ian
"""
import sys
import datetime
from sklearn import metrics
from keras.utils import plot_model
import numpy as np 
import os
from keras.models import Model, load_model
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import Input
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional,BatchNormalization,Activation
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import adam
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 防止图片生成之后闪退

from keras.layers import Layer
from keras.layers import Lambda, dot, concatenate
import argparse

parser = argparse.ArgumentParser(description='Mynet') 
# file path
parser.add_argument('--m6a_seq', type=str, default='m6a_seq_train.npy', help='file path of m6a_seq')
parser.add_argument('--m6a_str', type=str, default='m6a_str_train.npy', help='file path of m6a_str')
parser.add_argument('--label', type=str, default='m6a_label_train.npy', help='file path of label')
# cnn
parser.add_argument('--con1st_filters', type=int, default=128, help='filters') #  128
parser.add_argument('--con1st_kernel_size', type=int, default=11, help='kernel_size') # 

parser.add_argument('--con2st_filters', type=int, default=64, help='filters') # 
parser.add_argument('--con2st_kernel_size', type=int, default=5, help='kernel_size') # 
# lstm
parser.add_argument('--ipsize', type=int, default=4, help='ipsize')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden_size') 
parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
# Dense
parser.add_argument('--opsize', type=int, default=128, help='opsize')
## parameter
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate') # 0.001
parser.add_argument('--epochs', type=int, default=30, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size') 

parser.add_argument('--minidata', type=str, default='False', help='是否为小样本数据')
parser.add_argument('--length', type=int, default=101, help='length')
parser.add_argument('--data', type=str, default='data_101', help='data source')
parser.add_argument('--parameter', type=str, default='cnn_128_64_01', help='Optimizing parameters ')
args = parser.parse_args()  # ArgumentParser 通过 parse_args() 方法解析参数


# 创建存放结果的文件夹，计时开始
starttime = datetime.datetime.now()
data_path='./data/'
result_path='./result/'

if args.minidata=='True':
    result_path='./result_0/'
    data_path='./data_0/'

    


file_path=result_path + args.data +'/'+ args.parameter +'/'
if not os.path.exists(file_path):                   # 如果不存在则创建一个新的文件夹来存储文件
 	os.makedirs(file_path)


# # 载入训练数据
# load data: sequence
m6a_seq_data = np.load(data_path + args.data +'/datanpy/' + args.m6a_seq)
m6a_str_data = np.load(data_path + args.data +'/datanpy/' + args.m6a_str)
label =  np.load(data_path + args.data +'/datanpy/'  + args.label)

print('文件来源路径：',data_path + args.data +'/datanpy/')
print('结果存储路径：',file_path)
  
# 定义注意力层
class Attention(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, hidden_states):
        """
        Many-to-one attention mechanism for Keras.
        @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
        @return: 2D tensor with shape (batch_size, 128)
        @author: felixhao28.
        """
        hidden_size = int(hidden_states.shape[2])
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = dot([score_first_part, h_t], [2, 1], name='attention_score')
        attention_weights = Activation('softmax', name='attention_weight')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
        pre_activation = concatenate([context_vector, h_t], name='attention_output')
        attention_vector = Dense(hidden_size, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        return attention_vector


## TACT模型 (在两卷积层上加上激活函数relu)

# m6a_seq
m6a_seq_input = Input(shape=(args.length, 4), dtype='float32', name='m6a_seq_input')

m6a_seq = layers.Conv1D(args.con1st_filters, args.con1st_kernel_size, padding='same',activation = 'relu',name='m6a_seq_con_1st')(m6a_seq_input)
m6a_seq=BatchNormalization()(m6a_seq)
m6a_seq=layers.Dropout(args.dropout)(m6a_seq)

m6a_seq = layers.Conv1D(args.con2st_filters, args.con2st_kernel_size, padding='same',activation = 'relu',name='m6a_seq_con_2st')(m6a_seq)
m6a_seq=BatchNormalization()(m6a_seq)
m6a_seq=layers.Dropout(args.dropout)(m6a_seq)


# m6a_str
m6a_str_input=Input(shape=(args.length , 3), dtype='float32', name='m6a_str_input')
m6a_str = layers.Conv1D(args.con1st_filters, args.con1st_kernel_size, padding='same',activation = 'relu',name='m6a_str_con_1st')(m6a_str_input)

m6a_str=BatchNormalization()(m6a_str)
m6a_str=layers.Dropout(args.dropout)(m6a_str)

m6a_str = layers.Conv1D(args.con2st_filters, args.con2st_kernel_size, padding='same',activation = 'relu',name='m6a_str_con_2st')(m6a_str)
m6a_str=BatchNormalization()(m6a_str)
m6a_str=layers.Dropout(args.dropout)(m6a_str)


#merge
concatenated = layers.concatenate([m6a_seq, m6a_str],axis=-1)

#concatenated=BatchNormalization()(concatenated)
#concatenated=layers.Dropout(args.dropout)(concatenated)
concatenated=Bidirectional(LSTM(args.hidden_size, return_sequences = True, dropout=0.2, recurrent_dropout=0.5), merge_mode = 'concat')(concatenated)  #注意修改连接的层******

concatenated=Attention()(concatenated)

# concatenated=BatchNormalization()(concatenated)
# concatenated=layers.Dropout(args.dropout)(concatenated)

model=Dense(args.opsize, activation = 'relu')(concatenated)   #原为925
# model=BatchNormalization()(model)
# model=layers.Dropout(args.dropout)(model)
# model=Activation('relu')(model)
answer=Dense(1, activation = 'sigmoid')(model)


# 实例化模型，两个输入，一个输出
model= Model([m6a_seq_input, m6a_str_input], answer)
#plot_model(model, to_file='model2.png', show_shapes='True')


model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.Adam(lr = args.lr),
              metrics = ['accuracy'])


filename = file_path +'model_2.h5' 

# modelcheckpoint 与 earlystopping
# callbacks_list=[
#     ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='auto'),
#     # EarlyStopping(monitor='val_loss',patience=20, mode='auto'),# 如果连续三个没有下降则停止训练
#     ModelCheckpoint(filename, monitor = 'val_accuracy', save_best_only = True, mode = 'max') # 保留在验证集上效果最好的模型
#     ]

# 保存val_accuracy值最大时的模型 
modelCheckpoint = ModelCheckpoint(filename, monitor = 'val_accuracy', save_best_only = True, mode = 'max')

history = model.fit([m6a_seq_data, m6a_str_data], label, epochs = args.epochs, batch_size = args.batch_size, validation_split = 0.2, callbacks = [modelCheckpoint])



#绘图
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(file_path +'T&V_loss.png')
# plt.show()

plt.clf()
acc=history_dict['accuracy']
val_acc=history_dict['val_accuracy']
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig(file_path +'T&V_acc.png') #
# plt.show()

endtime = datetime.datetime.now()
print ('总运行时间：',endtime - starttime)











## predict

## load data: sequence
# region1 = np.load(CELL+'/'+TYPE+'/bagData/'+filename1+'_Seq_18.npz')
# region2 = np.load(CELL+'/'+TYPE+'/bagData/'+filename2+'_Seq_18.npz')
# label = region1['label']

# region1_seq = region1['sequence']
# region1_seq = np.transpose(region1_seq,(0,3,2,1)) # 转换维度
# region1_seq = np.squeeze(region1_seq,axis = 3) # 去掉维度3
# region2_seq = region2['sequence']
# region2_seq = np.transpose(region2_seq,(0,3,2,1)) # 转换维度
# region2_seq = np.squeeze(region2_seq,axis = 3) # 去掉维度3 	

# ## load data: DNase
# region1 = np.load(CELL+'/'+TYPE+'/bagData/'+filename1+'_DNase_18.npz')
# region2 = np.load(CELL+'/'+TYPE+'/bagData/'+filename2+'_DNase_18.npz')
# region1_expr = region1['expr']
# region1_expr = np.transpose(region1_expr,(0,3,2,1)) # 转换维度
# region1_expr = np.squeeze(region1_expr,axis = 3) # 去掉维度3 
# region2_expr = region2['expr']
# region2_expr = np.transpose(region2_expr,(0,3,2,1)) # 转换维度
# region2_expr = np.squeeze(region2_expr,axis = 3) # 去掉维度3 



# model=load_model('./best_model_TACT.h5') # 载入每个数据包的模型（权重？）
# score = model.predict([region1_seq, region2_seq, region1_expr, region2_expr]) # score为预测概率
# pred = (score > 0.5).astype(int) # 以0.5为阈值划分score作为预测值，数据格式转化为int,最后转化为一行

# f1 = metrics.f1_score(label, pred)
# auprc = metrics.average_precision_score(label, score)

# print('f1:',f1,'\n','auprc:',auprc)






























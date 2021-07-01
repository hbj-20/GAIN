# -*- coding:utf-8 -*-
from __future__ import division
import numpy as np
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Input, Dropout, Add, Reshape,Concatenate, LSTM, Dense, Flatten, GRU,recurrent,Conv1D
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from GCN import GraphConvolution
from GAT import GraphAttention
import tensorflow as tf
import keras.backend as K
from keras.layers import Lambda
from keras.layers import Activation
from ilayer import iLayer
import keras
from read_data import lzq_load_data
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

len_test = 108*7
len_closeness = 6
len_period = 3
len_trend = 1
input_gcn_train, input_gcn_test, input_gat_train,input_gat_test, Y_train, Y_test,Y_tran_train, Y_tran_test = lzq_load_data(len_test,len_closeness,len_period,len_trend)

#构造节点信息 numpy.matrix, 列代表特征，行代表节点数
# Parameters
N = 36                            # Number of nodes in the graph
F = 36                             # Original feature dimension
n_classes = 36                    # Number of classes
F_ = 36                         # Output size of first GraphAttention layer
n_attn_heads = 3              # Number of attention heads in first GAT layer
dropout_rate = 0            # Dropout rate (between and inside GAT layers)
l2_reg = 5e-6               # Factor for l2 regularization
learning_rate = 1e-5          # Learning rate for Adam
epochs = 120            # Number of training epochs
es_patience = 20           # Patience fot early stopping
supports = 2   # k+1

#head = 1, epoch = 60, loss= 0.0018


def get_D(output):
    d = K.sum(output, axis=2)
    d = tf.matrix_diag(d)
    return d

def get_Laplacian(input):
    L = input[0]
    A = input[1]
    Df = K.pow(K.sum(A, axis=2), -1)
    Df = tf.matrix_diag(Df)
    Lm = K.batch_dot(Df, L)
    return Lm

def polynomial_0(A):
    d = K.sum(A, axis=2)
    d = tf.matrix_diag(d)
    Df = K.pow(K.sum(A, axis=2), -1)
    Df = tf.matrix_diag(Df)
    T_k = K.batch_dot(d,Df)
    return T_k

def Laplacian(A):
    #### get Laplacian
    D = Lambda(get_D)(A)
    A1 = Lambda(lambda x: x *(-1))(A)
    L_tilde = Add()([D, A1])
    L = Lambda(get_Laplacian)([L_tilde, A])
    P = Lambda(polynomial_0)(A)
    return L,P

def slice(x, index):
    return x[index]
def squeeze(x):
    return tf.squeeze(x,axis=2)

def GCN_block(X,P,L,filters):
    #H = Dropout(0.5)(X)
    H = GraphConvolution(filters[0], supports, activation='relu', kernel_regularizer=l2(5e-4))([X] + [P] + [L])
    #H = GraphConvolution(filters[1], supports, activation='relu', kernel_regularizer=l2(5e-4))([H] + [P] + [L])
    return H



outputs_flow = []
outputs_tran = []
inputs_gcn = []
inputs_gat = []

###closeness
def output_block(len_block):
    output_close = []
    output_trans_close = []
    for i in range(len_block):
        T_in = Input(shape=(N, F))
        inputs_gat.append(T_in)
        graph_attention_2 = GraphAttention(n_classes,
                                           attn_heads=n_attn_heads,
                                           attn_heads_reduction='average',
                                           dropout_rate=dropout_rate,
                                           activation='relu',
                                           kernel_regularizer=l2(l2_reg),
                                           attn_kernel_regularizer=l2(l2_reg))(T_in)
        trans_output = Lambda(slice, arguments={'index': n_attn_heads})(graph_attention_2)
        #trans_output = Reshape(target_shape=(N * N,1))(trans_output)
        #trans_output = Activation('relu')(trans_output)
        output_trans_close.append(trans_output)
        #####GCN
        flow_in_sequence = []
        for head in range(n_attn_heads):
            A = Lambda(slice, arguments={'index': head})(graph_attention_2)
            P,L = Laplacian(A)
            for i in range(2):
                X_in = Input(shape=(N, 1))
                output = GraphConvolution(2, supports, activation='tanh', kernel_regularizer=l2(5e-4))([X_in] + [P] + [L])
                #output = GraphConvolution(2, supports, activation='relu', kernel_regularizer=l2(5e-4))([output] + [P] + [L])
                inputs_gcn.append(X_in)
                flow_in_sequence.append(output)
        new_outputs = []
        for output in flow_in_sequence:
            new_outputs.append(iLayer()(output))
        main_output = keras.layers.add(new_outputs)
        main_output = Activation('relu')(main_output)
        main_output = Reshape(target_shape=(1, N * 2))(main_output)
        output_close.append(main_output)

    if len_block>1:
        output_close = Concatenate(axis=1)(output_close)
        output_close = recurrent.GRU(units=N * 2, input_shape=(len_block, N * 2), return_sequences=False,activation='relu', recurrent_activation='hard_sigmoid',
                                 kernel_initializer='glorot_uniform', recurrent_dropout=0.1)(output_close)
        #output_trans_close = Concatenate(axis=1)(output_trans_close)
        #output_trans_close = Conv1D(filters=1, kernel_size=3, padding='same')(output_trans_close)
        #output_trans_close = Lambda(squeeze)(output_trans_close)
        #output_trans_close = Activation('relu')(output_trans_close)
        #print(output_trans_close.shape)
        new_outputs = []
        for output in output_trans_close:
            new_outputs.append(iLayer()(output))
        output_trans_close = keras.layers.add(new_outputs)
        output_trans_close = Reshape(target_shape=(N * N,1))(output_trans_close)
        output_trans_close = Lambda(squeeze)(output_trans_close)
    else:
        output_close = Flatten()(output_close[0])
        output_trans_close = Flatten()(output_trans_close[0])
    output_trans_close = Activation('relu')(output_trans_close)
    #output_close = Dense(units = N*2)(output_close)
    #output_trans_close = Dense(units = N*N)(output_trans_close)
    #output_close = Dropout(dropout_rate)(output_close)
    #output_close = Activation("relu")(output_close)
    return output_close,output_trans_close

output_close,output_tran_close = output_block(len_closeness)
output_period,output_tran_period = output_block(len_period)
output_trend,output_tran_trend= output_block(len_trend)
outputsf = [output_close,output_period,output_trend]
outputst = [output_tran_close,output_tran_period,output_tran_trend]

final_outputsf = []
final_outputst = []
for output in outputsf:
    final_outputsf.append(iLayer()(output))
for output in outputst:
    final_outputst.append(iLayer()(output))

final_outputsf = keras.layers.add(final_outputsf)
final_outputsf= Activation('tanh',name = "final_outputf")(final_outputsf)
final_outputst = keras.layers.add(final_outputst)
final_outputst= Activation('tanh',name = "final_outputt")(final_outputst)

main_input = inputs_gcn + inputs_gat
# Build model
model = Model(inputs=main_input, outputs=[final_outputsf,final_outputst])
model.summary()


# Callbacks
es_callback = EarlyStopping(monitor='val_weighted_acc', patience=es_patience)
tb_callback = TensorBoard(batch_size=N)
mc_callback = ModelCheckpoint('logs/best_model.h5',
                              monitor='val_weighted_acc',
                              save_best_only=True,
                              save_weights_only=True)

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))
def root_mean_square_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

# Train model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto',factor=0.1,verbose = 2)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
model.compile(loss={'final_outputf': 'mean_squared_error',
                    'final_outputt': 'mean_squared_error'},
              loss_weights={'final_outputf': 1,
                            'final_outputt': 1}, optimizer='adam',metrics=['mae'])
model.fit(input_gcn_train+input_gat_train,
          [Y_train,Y_tran_train],
          epochs=epochs,
          batch_size=5,
          validation_split = 0.2,verbose=1,
          shuffle=False,callbacks=[reduce_lr,early])

pred = model.predict(input_gcn_test+input_gat_test)
print("done")
flow = np.array(pred[0])
tran = np.array(pred[1])
np.save("D:/hbj/shenzhen/gain/result_100/flow_100.npy",flow)
np.save("D:/hbj/shenzhen/gain/result_100/trans_100.npy",tran)
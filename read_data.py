# -*- coding:utf-8 -*-
from scipy.io import loadmat
import numpy as np
from sklearn import preprocessing

att_heads = 3


def lzq_load_data(len_test,len_close,len_period,len_trend,T_closeness=1,T_period=108,T_trend=108*7):
    map_count = loadmat("/Users/cathy/Desktop/icde/shenzhen_data/flow_ts_100.mat")
    map_a = map_count["in_ts"]  #原本的66列（香蜜湖站）没有进站只有出站
    map_b = map_count["out_ts"]
    len_total = 108 * 30
    map_a = np.array(map_a)[0:len_total,:]
    map_b = np.array(map_b)[0:len_total,:]

    # normalization
    robust_a = preprocessing.MinMaxScaler()
    map_a = robust_a.fit_transform(map_a)
    robust_b = preprocessing.MinMaxScaler()
    map_b = robust_b.fit_transform(map_b)

    #a = np.array([0]*len_total)
    #map_a = np.insert(map_a, 65, values=a, axis=1)
    node = map_a.shape[1]

    number_of_skip_hours = T_trend * len_trend
    all_data = np.concatenate((map_b, map_a), axis=1)
    Y = all_data[number_of_skip_hours:len_total]
    Y_train = Y[:-len_test]
    Y_test = Y[-len_test:]

    map_a = map_a.reshape((len_total, node,1))
    map_b = map_b.reshape((len_total, node,1))

    map_count = loadmat("/Users/cathy/Desktop/icde/shenzhen_data/get_transition_100.mat")
    map_tran = map_count["transition"]
    map_tran = np.array(map_tran)[0:len_total,:]
    # RobustScaler normalization
    robust_tran = preprocessing.MinMaxScaler()
    map_tran = robust_tran.fit_transform(map_tran)

    map_tran = map_tran.reshape((len_total,node,node))
    all_tran = map_tran[number_of_skip_hours:len_total]
    Y_tran_train = all_tran[:-len_test]
    Y_tran_test = all_tran[-len_test:]
    Y_tran_train = Y_tran_train.reshape((-1,36*36))
    Y_tran_test = Y_tran_test.reshape((-1, 36*36))


    input_gcn_train = []
    input_gcn_test = []
    input_gat_train = []
    input_gat_test = []

    for cl in range(T_closeness, len_close + T_closeness):
        all_b = map_b[number_of_skip_hours - cl:len_total - cl]
        all_a = map_a[number_of_skip_hours - cl:len_total - cl]
        for i in range(att_heads):
            input_gcn_train.append(all_b[:-len_test])
            input_gcn_train.append(all_a[:-len_test])
            input_gcn_test.append(all_b[-len_test:])
            input_gcn_test.append(all_a[-len_test:])
        all_tran = map_tran[number_of_skip_hours - cl:len_total - cl]
        input_gat_train.append(all_tran[:-len_test])
        input_gat_test.append(all_tran[-len_test:])

    for cl in range(T_period, len_period + T_period):
        all_b = map_b[number_of_skip_hours - cl:len_total - cl]
        all_a = map_a[number_of_skip_hours - cl:len_total - cl]
        for i in range(att_heads):
            input_gcn_train.append(all_b[:-len_test])
            input_gcn_train.append(all_a[:-len_test])
            input_gcn_test.append(all_b[-len_test:])
            input_gcn_test.append(all_a[-len_test:])
        all_tran = map_tran[number_of_skip_hours - cl:len_total - cl]
        input_gat_train.append(all_tran[:-len_test])
        input_gat_test.append(all_tran[-len_test:])

    for cl in range(T_trend, len_trend + T_trend):
        all_b = map_b[number_of_skip_hours - cl:len_total - cl]
        all_a = map_a[number_of_skip_hours - cl:len_total - cl]
        for i in range(att_heads):
            input_gcn_train.append(all_b[:-len_test])
            input_gcn_train.append(all_a[:-len_test])
            input_gcn_test.append(all_b[-len_test:])
            input_gcn_test.append(all_a[-len_test:])

        all_tran = map_tran[number_of_skip_hours - cl:len_total - cl]
        input_gat_train.append(all_tran[:-len_test])
        input_gat_test.append(all_tran[-len_test:])

    #print(input_gcn_train[0].shape,input_gcn_train[1].shape,input_gcn_train[2].shape)  #1815*94*!
    #print(input_gat_train[0].shape,input_gat_train[1].shape,input_gat_train[2].shape,)  #1815*94*94
    #print(Y_train.shape,Y_tran_train.shape)    #1815*188,  2592*94*94

    return input_gcn_train, input_gcn_test, input_gat_train,input_gat_test, Y_train, Y_test,Y_tran_train, Y_tran_test
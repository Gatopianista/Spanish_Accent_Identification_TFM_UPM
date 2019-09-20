"""
@author: andreafontalvo

Spanish accent identification
Master on Automation and Robotics
Final Project 
UPM - Universidad Politécnica de Madrid

This script stablishes the functions to declare the separation of data into test 
and train, from the datasets obtained previously. It also determines the 
arquitecture of the proposed model. 
These functions are later used on models_training.py.
"""  

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from datetime import datetime
import numpy as np
import time
import random

def tf_process(ar,es):

    random.shuffle(es)
    random.shuffle(ar)

    es_train = es[:80]
    ar_train = ar[:80]
    es_test = es[80:88]
    ar_test = ar[80:88]

    train = es_train+ar_train
    random.shuffle(train)
    test = es_test+ar_test
    random.shuffle(test)

    x_train = []
    y_train = []
    for data, target in train:
        x_train.append(data)
        y_train.append(target)

    x_test = []
    y_test = []

    for data, target in test:
        x_test.append(data)
        y_test.append(target)
        
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    print("done")

    return(x_train,y_train,x_test,y_test)


def tf_training(x_train,y_train,x_test,y_test,nf):

    dt = datetime.utcfromtimestamp(int(time.time())).strftime('%Y-%m-%d %H-%M-%S')
    name = f"model-{nf}f-{dt}"
    dirname = f"models/model-{nf}f-{dt}"
    os.mkdir(dirname)
    os.mkdir(dirname+"/checkpoints")

    model = Sequential()
    model.add(LSTM(128, input_shape=(x_train.shape[1:]),activation='relu',return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(128,input_shape=(x_train.shape[1:]),activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(2,activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(lr=1e-3,decay=1e-5)

    model.compile(loss='sparse_categorical_crossentropy' , optimizer=opt,metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir=f"logs\{name}")
    filepath = dirname+"/checkpoints/{epoch:02d}"  
    checkpoint = ModelCheckpoint("{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones          

    history = model.fit(x_train, y_train,epochs=5, validation_data=(x_test, y_test),callbacks=[tensorboard,checkpoint])
    return(history,name)


def tf_test(x_ptest,y_ptest):
    lmodel = load_model("models/model.model")
    y_pred = lmodel.predict(x)
    return y_pred


def separate_test(ft_ar,ft_es):
    lb_es = np.ones(90)
    lb_ar = np.zeros(90)

    es = []
    ar = []
    
    for i in range(90):
        es.append([ft_es[0],lb_es[0]])
        ar.append([ft_ar[0],lb_ar[0]])

    random.shuffle(es)
    random.shuffle(ar)

    es_posttest = es[88:]
    ar_posttest = ar[88:]

    es = es[:88]
    ar = ar[:88]

    ptest = es_posttest+ar_posttest
    random.shuffle(ptest)

    x_ptest = []
    y_ptest = []
    for data, target in ptest:
        x_ptest.append(data)
        y_ptest.append(target)

    x_ptest = np.array(x_ptest)
    y_ptest = np.array(y_ptest)

    print("done")    
    return x_ptest,y_ptest,ar,es
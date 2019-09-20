"""
@author: andreafontalvo

Spanish accent identification
Master on Automation and Robotics
Final Project 
UPM - Universidad Politécnica de Madrid

This scripts uses the functions declared on model_setup and it trains several models, 
which are saved if the specified criteria is met.
""" 

import os
import model_setup
import numpy as np
import json

minloss = 1000
trainnum = 0
numfeatures = 13
# numfeatures = 14
# numfeatures = 39
folder = f"{numfeatures}f-case"
os.mkdir(folder)
os.mkdir(f"{folder}/savemodels")
os.mkdir(f"{folder}/histories")

ft_ar = np.load(f"data/{numfeatures}f/features_ar.npy")
ft_es = np.load(f"data/{numfeatures}f/features_es.npy")

x_ptest,y_ptest,ar,es = model_setup.separate_test(ft_ar,ft_es)

np.save(f"{folder}/x_ptest",x_ptest)
np.save(f"{folder}/y_ptest",y_ptest)

for k in range(50):
        
    x_train,y_train,x_test,y_test= model_setup.tf_process(ar,es)

    hist,name = model_setup.tf_training(x_train,y_train,x_test,y_test,numfeatures)

    acc = hist.history["acc"]
    val_acc = hist.history["val_acc"]
    loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    val_loss = np.array(val_loss)

    flag = any(np.isnan(val_loss))

    if (flag == False):
        if(np.min(val_loss)<minloss):
                minloss = np.min(val_loss)
                trainnum = k
                print(trainnum)
                    
        print(trainnum)
        save = []
        try:
            for i in range(5):
                if (val_acc[i]>0.5):
                    if(val_loss[i]<0.7):
                        save.append(i+1) 
                        
            if (len(save)>0):
                os.mkdir(f"{folder}/savemodels/{name}/")
                for epoch in save:
                    os.rename(f"models/{name}/checkpoints/0{epoch}.model",f"{folder}/savemodels/{name}/0{epoch}.model")


            with open(f'{folder}/histories/{name}-history.json', 'w') as f:
                json.dump(hist.history, f)
        except:
            print("exception occurred")
        #os.remove("models/"+name+"/")

np.save(f"{folder}/numberk",k)
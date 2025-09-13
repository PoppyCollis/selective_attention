# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:24:09 2024

@author: pfkin
"""

import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import json

#where to save data
logdir="data/outputs/"

def slide(x,data,window_horiz):
    assert len(x)==len(data)
    
    ma=[]
    horiz=[]
    start_posn=0
    end_posn=0 # just to get started
    start_horiz=x[0]
    end_horiz=start_horiz+window_horiz

    # final_horiz=100
    # for i in range(start_horiz, final_horiz-window_horiz+1,1):
    while end_posn < len(x):
        try:
            end_posn=np.where(x>=end_horiz)[0][0]
        except: 
            end_posn=len(x)
        ma.append(np.mean(data[start_posn:end_posn]))
        horiz.append(start_horiz)
        start_horiz+=1
        end_horiz=start_horiz+window_horiz
        start_posn=np.where(x>=start_horiz)[0][0]
        
    return np.asarray(horiz)+window_horiz/2, np.asarray(ma)

def plot_model(x,y,config,model_confidence, true_confidence,model,subject,k,  T=None, savefile=True):
    # sort everything by x coord
    inds=np.argsort(x)
    x=x[inds]
    y=y[inds]
    model_confidence=model_confidence[inds]
    true_confidence=true_confidence[inds]
    
    x_axis=x
    
    #add sliding window
    window_horizon=20
    x_middled_true,slide_true_confidence=slide(x_axis,true_confidence, window_horizon)
    x_middled_model,slide_model_confidence=slide(x_axis,model_confidence, window_horizon)
    
    
    #plot
    # plt.figure()
    # plt.plot(x_middled_model,slide_model_confidence, "c-", label="model "+ str(model))
    # plt.plot(x_middled_true,slide_true_confidence, "k-", label="true data")
    # plt.title("Config "+str(config))
    # plt.ylim([1,4])
    # plt.legend()
    # plt.show()
    
    if savefile:
        try:
            filename="Subject_"+str(subject)+"_"+str(model)+"_"+"Config_"+str(config)+".json"
            metrics = {"config": config, "model": model, "ind":subject, "x_middled_model": x_middled_model.tolist(), "slide_model_confidence": slide_model_confidence.tolist(),"x_middled_true": x_middled_true.tolist(), "slide_true_confidence": slide_true_confidence.tolist(),"k": k.tolist(), "T": T}
            path = logdir+filename
            with open(path, "w") as file:
                json.dump(metrics, file)         
                #utils.save_json(metrics, logdir + "metrics.json")
                #data_to_save=np.stack((x_middled_model,slide_model_confidence,x_middled_true,slide_true_confidence))
                #np.save(logdir+filename,data_to_save)
        except:
            pass
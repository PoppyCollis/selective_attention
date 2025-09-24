# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:29:43 2024

@author: pfkin

Monte Carlo Simulations: The Monte Carlo simulation is implemented using NumPy's random number 
generation (np.random.randn), replicating the sensory noise and likelihood calculations.
Dirichlet Noise: The gamma.rvs function from scipy.stats is used to inject Dirichlet decision noise.
Negative Log-Likelihood Calculation: The negative log-likelihood (nllk) is computed by summing the 
log probabilities from the simulated data, similarly to the MATLAB code.
Data Structures: I assumed the input data (trl and stiPar) are provided as dictionaries. 
This should match your data structure, or it can be adjusted as necessary.
Config Indexing: In Python, indexing is zero-based, so I adjusted the indices accordingly 
during negative log-likelihood computation.
"""

import numpy as np
from scipy.stats import gamma
from plot_model_data import plot_model
import json

def nll_map_is(x, trl, stiPar, config_pool, subject, condIdx=None, savefile=True,return_nllk_only=True):
    """
    Max model with both inference noise and sensory noise
    This function uses Monte Carlo simulations to estimate response distribution
    and computes negative log-likelihood of the model.

    Args:
    - x: parameter vector [boundaries, sensory noise, inference noise, lapse rate]
    - trl: dictionary containing trial data
    - stiPar: dictionary containing stimulus parameters (ux, uy, sig_s, nr, ncat)
    - config_pool: list of configurations to consider
    - condIdx: indices for conditioning trials (default = all)

    Returns:
    - nllk: negative log-likelihood
    """
    if condIdx is None:
        condIdx = slice(None)  # Select all indices

    simntrl = 2000  # Sample size for Monte Carlo simulation per trial

    ux = stiPar['ux']
    uy = stiPar['uy']
    sig_s = stiPar['sig_s']
    nr = stiPar['nr']
    ncat = stiPar['ncat']

    # Filter trial data based on condIdx
    trl_estD = trl['estD'][condIdx]
    trl_estC = trl['estC'][condIdx]
    trl_config = trl['config'][condIdx]
    trl_target_x = trl['target_x'][condIdx]
    trl_target_y = trl['target_y'][condIdx]
    
    trl_estD=trl_estD.astype ('uint32')
    trl_estC=trl_estC.astype ('uint32')

    # Define parameter boundaries
    k = np.sort(np.concatenate(([-np.inf], x[:3], [np.inf])))  # Boundaries for 4-point confidence report
    sig_x =  10 ** x[4]  # Sensory noise
    alpha = 10 ** x[3]  # Dirichlet inference noise
    lapse = x[5]
    
    # # setup where we have no sensory noise
    # sig_x = 0  # Sensory noise
    # alpha = 10 ** x[3]  # Dirichlet inference noise
    # lapse = x[5]
    # #T=x[4] # as well as optimizing lapse, optimize temperature parameter.
    T=1
    
    #print("lapse", lapse, "temperature", T, "alpha:", alpha)
    
    pMat = [None] * (len(config_pool))
    pMat_dec = [None] * (len(config_pool))
    pMat_conf = [None] * (len(config_pool))

    for idxconfig,config in enumerate(config_pool):
        temp_s = trl_target_x[trl_config == config]
        ntrl = len(temp_s)
        ux_vec = ux[config-1, :]
        pMat[idxconfig] = np.empty((ncat, nr, ntrl))

        sim_xMat = np.tile(temp_s, (simntrl, 1)) + np.random.randn(simntrl, ntrl) * sig_x
        sim_xMat = np.repeat(sim_xMat[:, :, np.newaxis], ncat, axis=2)

        cMatx = np.tile(ux_vec, (simntrl * ntrl, 1))
        cMatx = np.reshape(cMatx, (simntrl, ntrl, ncat))

        # Run this part for 2D
        temp_sy = trl_target_y[trl_config == config]
        uy_vec = uy[config-1, :]
        sim_yMat = np.tile(temp_sy, (simntrl, 1)) + np.random.randn(simntrl, ntrl) * sig_x
        sim_yMat = np.repeat(sim_yMat[:, :, np.newaxis], ncat, axis=2)

        cMaty = np.tile(uy_vec, (simntrl * ntrl, 1))
        cMaty = np.reshape(cMaty, (simntrl, ntrl, ncat))

        sig_p = np.sqrt(sig_s ** 2 + sig_x ** 2)
        # pkadded in sqrt 22nov
        sim_lh = (1 / (2 * np.pi * sig_p ** 2)) * np.exp(
            -((sim_xMat - cMatx) ** 2 + (sim_yMat - cMaty) ** 2) / (2 * sig_p ** 2)
        )

        ############## expt to add T
        # take the likelihood and get log likelihood
        log_sim_lh= np.log(sim_lh)
        # weight LLK by temp parameter
        log_sim_lh *= T
        # convert back to likelihood
        sim_lh = np.exp(log_sim_lh)
        ######################################################

        # Inject Dirichlet decision noise
        sim_post = sim_lh / np.sum(sim_lh, axis=2, keepdims=True)
        sim_post = gamma.rvs(sim_post * alpha, scale=1)
        sim_post /= np.sum(sim_post, axis=2, keepdims=True)
        if np.isnan(sim_post).any():
            print("warning sim_post has nan")
            _=input()

        sim_c_map = np.max(sim_post, axis=2)
        sim_d = np.argmax(sim_post, axis=2) + 1  # Category decision

        # Transform internal confidence into 4-point confidence report
        sim_r = np.digitize(sim_c_map, k, right=True)
        sim_r = np.clip(sim_r, 1, 4)  # Avoid numerical issues

        for tr in range(ntrl):
            for d in range(ncat):
                for r in range(nr):
                    temp = np.sum((sim_d[:, tr] == d + 1) & (sim_r[:, tr] == r + 1)) / simntrl
                    pMat[idxconfig][d, r, tr] = (1 - lapse) * temp + lapse * (1 / (ncat * nr))

        pMat[idxconfig] = np.maximum(pMat[idxconfig], np.finfo(float).eps)
        # marginalise across pMat to get, for each sample, the prob of each decision and each confidence
        pMat_dec[idxconfig]= np.sum(pMat[idxconfig],axis=1)
        pMat_conf[idxconfig]= np.sum(pMat[idxconfig],axis=0)
        
        ###plot and save true and model confidences 
        model_confidence=np.mean(sim_r,axis=0)
        true_confidence=trl_estC[trl_config == config]
        #plot_model(temp_s,temp_sy,config,model_confidence, true_confidence,"Max",subject,k)
        if "T" in locals() and T!=1:
            plot_model(temp_s,temp_sy,config,model_confidence, true_confidence,"MaxwithTemp",subject,k, T=T,savefile=savefile)
        else: 
            plot_model(temp_s,temp_sy,config,model_confidence, true_confidence,"Max",subject,k,savefile=savefile)
     
    # Calculate negative log-likelihood
    nllk = 0
    nllk_dec=0
    nllk_conf=0
    joint_counts=0
    for idxconfig, config in enumerate(config_pool):
        temp_D = trl_estD[trl_config == config] - 1  # Convert to 0-based index
        temp_C = trl_estC[trl_config == config] - 1  # Convert to 0-based index
        ind = np.ravel_multi_index((temp_D, temp_C, np.arange(len(temp_D))), pMat[idxconfig].shape)
        llk = np.sum(np.log(pMat[idxconfig].flat[ind]))
        ind_dec=np.ravel_multi_index((temp_D, np.arange(len(temp_D))), pMat_dec[idxconfig].shape)
        llk_dec = np.sum(np.log(pMat_dec[idxconfig].flat[ind_dec]))
        ind_conf=np.ravel_multi_index((temp_C, np.arange(len(temp_C))), pMat_conf[idxconfig].shape)
        llk_conf = np.sum(np.log(pMat_conf[idxconfig].flat[ind_conf]))
        nllk -= llk
        nllk_dec -= llk_dec
        nllk_conf -= llk_conf

        # instead of returning nllk. we return number of joint distirbutions which are correct
        max_indices_flattened = np.argmax(pMat[idxconfig].reshape(3*4, ntrl), axis=0)
        joint_true=temp_D*4+temp_C
        correct_joint=max_indices_flattened==joint_true
        joint_counts+=np.sum(correct_joint)
        
    if savefile:
        try:
            #where to save data
            print("nllk ", np.round(nllk),"nllk_decision ", np.round(nllk_dec) ,"nllk_confidence ", np.round(nllk_conf) )
            #logdir="data/outputs/separate_pybads_new_sig_x=0_Ta/"
            logdir="data/outputs/"
            if "T" in locals() and T!=1:
                model="MaxwithTemp"
            else:
                model="Max"
            filename="NLLK_Subject_"+str(subject)+"_Model_"+str(model)+".json"
            metrics = {"Jointcount": joint_counts, "NLLK": nllk,"NLLK_dec": nllk_dec,"NLLK_conf": nllk_conf, "x": x.tolist(),"model": model, "ind":subject}
            path = logdir+filename
            with open(path, "w") as file:
                json.dump(metrics, file)         
                #utils.save_json(metrics, logdir + "metrics.json")
                #data_to_save=np.stack((x_middled_model,slide_model_confidence,x_middled_true,slide_true_confidence))
                #np.save(logdir+filename,data_to_save)
        except:
            pass

    if return_nllk_only:
        # return 4*84-joint_counts # this is the nuber of counts wrong, which we watn to minize
        return nllk
    else:
        return nllk, pMat, 4*ntrl-joint_counts 
        # return nllk, pMat

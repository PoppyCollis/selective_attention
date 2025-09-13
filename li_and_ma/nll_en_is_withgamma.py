# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:30:56 2024

@author: pfkin



DONT USE THIS MODEL FOR MULTIATTENTION.  THERE IS A SEPARATE FILE WITH THE CORRECT MODEL FOR THAT. 

"""

import numpy as np
from scipy.special import gammaln
from scipy.stats import gamma, chi2
from plot_model_data import plot_model
import json
import itertools

# helper functions
def normalise(v, axis=None):
    v = np.asarray(v, dtype=float)
    if axis is None:
        z = v.sum()
        return v / z if z > 0 else np.full_like(v, 1.0 / len(v))
    z = v.sum(axis=axis, keepdims=True)
    out = v / z
    out[np.isnan(out)] = 0.0
    return out

def gmm_responsibilities_with_attention(x, k, likelihoods, gamma=1., cardinality = 1):
    gamma=np.inf
    #gamma=10.
    
    p_x_given_s=likelihoods
 
    p_s=np.array([1/3,1/3,1/3])
    
    ### cardinality says how many different structure there can be
    ### but still need to hard code what these structures are : done here
    if cardinality == k:
        structures = np.eye(k,dtype=int) # likelihood P(s_i = 1 | phi) as indicator function for each latent
    # elif attended_states == 2:
    #     p_s_phi = np.eye(k) # likelihood P(s_i = 1 | phi) as 2-state indicator functions for each latent
    #     p_s_phi = np.fliplr(p_s_phi) 
    #     p_s_phi = np.where(p_s_phi == 1, 0, 1)
    elif cardinality >1 and cardinality<k:  
        ### bespoke list of structures of size = cardinality
        structures  = np.array([[1,1,0],[0,1,1]])
    elif cardinality == 1:
        structures = np.ones((1,k),dtype=int) # all three latents contribute
    
    else:
        raise ValueError("Not implemented for sets of Phi != 1, 2 or 3")
    
    p_s_given_phi=normalise(structures*p_s,axis=1)
    
    p_phi = np.full(cardinality, 1.0 / cardinality)  # p(φ) flat prior
    
    # evidence per structure
    #m_phi = np.einsum('ij,jkl->ikl',p_s_given_phi, p_x_given_s.T).T 
    p_x_given_phi =  p_x_given_s @ p_s_given_phi.T
    p_phi_given_x = normalise(p_phi * p_x_given_phi, axis=2) 
    
    # per-structure posteriors p(s|x,φ)
    p_s_given_x_phi_unnorm=p_s_given_phi[None,None,:,:] * p_x_given_s[:,:,None,:]
    p_s_given_x_phi = normalise(p_s_given_x_phi_unnorm, axis=-1)  # [P,S]
    # in general this should use p_x_given_s,phi but since phi is independent of x given s we can reduce it is p_x_given_s?
    
    # tempered structure weights
    if np.isinf(gamma):
        w = np.zeros_like(p_phi_given_x)
        idx = np.argmax(p_phi_given_x, axis=-1)
        w[np.arange(p_phi_given_x.shape[0])[:, None],  # 2000 x 1
          np.arange(p_phi_given_x.shape[1])[None, :],  # 1 x 84
          idx] = 1
        #w[np.argmax(p_phi_given_x,axis=-1)] = 1.0
    else:
        w = normalise(p_phi_given_x**gamma, axis=-1)             
    
    # now calcualte posterior from p(s|x) from p(phi|x) and p(s|x,phi))
    p_s_given_x_unnorm = (p_s_given_x_phi * w[...,:,None]).sum(axis=2)
    # normlaise just in case inputs arent proper distributons
    p_s_given_x=normalise(p_s_given_x_unnorm, axis=-1)
    
    #standard posterior (i.e. not via phi) is p(s|x) ∝ p(x|s) p(s)
    standard_posterior=normalise(p_x_given_s * p_s, axis=-1)
   
    
    # then need to use this posterior?
    return p_s_given_x, standard_posterior

def nll_en_is_withgamma(x, trl, stiPar, config_pool, subject, condIdx=None, attention =False,T_update=True, savefile=True,return_nllk_only=True,attentionTestNormalisation=False):
    """
    Entropy model with both inference noise and sensory noise using Monte-Carlo simulations 
    to estimate response distribution and compute the negative log-likelihood of the model.
    
    Args:
        x: Parameter vector.
        trl: Dictionary containing trial data and config labels. 
             Required keys: 'estD', 'estC', 'config', 'target_x', 'target_y'
        stiPar: Dictionary containing stimulus locations.
        config_pool: List of configuration indices.
        condIdx: Indices of conditions to use (optional).
    
    Returns:
        nllk: Negative log-likelihood of the model.
    """
    if condIdx is None:
        condIdx = slice(None)

    simntrl = 2000 # sample size for Monte-Carlo simulation per trial

    ux = stiPar['ux']
    uy = stiPar['uy']
    sig_s = stiPar['sig_s']
    nr = stiPar['nr']
    ncat = stiPar['ncat']

    # Apply condition indexing
    trl_estD = trl['estD'][condIdx]
    trl_estC = trl['estC'][condIdx]
    trl_config = trl['config'][condIdx]
    trl_target_x = trl['target_x'][condIdx]
    trl_target_y = trl['target_y'][condIdx]
    
    trl_estD=trl_estD.astype ('uint32')
    trl_estC=trl_estC.astype ('uint32')

    # Parameters
    k = np.sort(np.concatenate(([-np.inf], x[:3], [np.inf])))  # boundaries for 4-point confidence report
    sig_x = 10**x[4]  # sensory noise
    alpha = 10**x[3]  # Dirichlet inference noise
    lapse = x[5]
    
    # # setup where we have no sensory noise
    # sig_x =  0  # Sensory noise
    # alpha = 10 ** x[3]  # Dirichlet inference noise
    # lapse = x[5]
    
    # T= x[4] # as well as ptimizing lapse, optmiize temperature parameter. ################
    # if not T_update:
    #     T=1
    ##can't use the above if we are using x[4] for sig_x
    T=1
    
    #print("lapse", lapse, "temperature", T, "alpha", alpha)
    #print("lapse",lapse, "T", T)
    
    pMat = [None] * (len(config_pool))
    pMat_dec = [None] * (len(config_pool))
    pMat_conf = [None] * (len(config_pool))
    
    for idxconfig,config in enumerate(config_pool):
        temp_s = trl_target_x[trl_config == config]
        ntrl = len(temp_s)
        ux_vec = ux[config-1,:]
        pMat[idxconfig] = np.empty((ncat, nr, ntrl))

        # Simulate measurements
        sim_xMat = np.tile(temp_s, (simntrl, 1)) + np.random.randn(simntrl, ntrl) * sig_x
        sim_xMat = np.repeat(sim_xMat[:, :, np.newaxis], ncat, axis=2)

        cMatx = np.tile(ux_vec, (simntrl * ntrl, 1))
        cMatx = np.reshape(cMatx, (simntrl, ntrl, ncat))

        temp_sy = trl_target_y[trl_config == config]
        uy_vec = uy[config-1,:]
        sim_yMat = np.tile(temp_sy, (simntrl, 1)) + np.random.randn(simntrl, ntrl) * sig_x
        sim_yMat = np.repeat(sim_yMat[:, :, np.newaxis], ncat, axis=2)

        cMaty = np.tile(uy_vec, (simntrl * ntrl, 1))
        cMaty = np.reshape(cMaty, (simntrl, ntrl, ncat))

        sig_p = np.sqrt(sig_s**2 + sig_x**2)
        # pkadded in sqrt 22nov
        sim_lh = (1 / (2 * np.pi * sig_p ** 2)) * np.exp(
            -((sim_xMat - cMatx) ** 2 + (sim_yMat - cMaty) ** 2) / (2 * sig_p ** 2)
        )
        
        unnorm_sim_lh=sim_lh.copy() ####################### just calcualted for an expt re normalisation

        if attention:
            cardinality=2 # numbr of allowed structure
            p_s_given_x, standard_posterior = gmm_responsibilities_with_attention(x, 
                                                        k=sim_lh.shape[-1],                                                                                         
                                                        likelihoods=sim_lh,
                                                        cardinality=cardinality)
        
            sim_post=p_s_given_x
            # # now change likelihood into likelihood after attention cut down 
            # sim_lh=sim_lh * attended
            # sim_lh[sim_lh == 0] = np.finfo(float).eps  # prevent numerical issues 
            # ### WE COULD REMOVE THE UNATTENDED COLUMNS ENTIRELY, BUT DOESNT MAKE ANY DFFERENCE TO ENTROPY MEASURE, SO LEAVE IN

            
        # ############## expt to add T
        # # take the likelihood and get log likelihood
        # log_sim_lh= np.log(sim_lh)
        # # weight LLK by temp parameter
        # log_sim_lh *= T
        # # convert back to likelihood
        # sim_lh = np.exp(log_sim_lh)
        # ######################################################
        
        # if attentionTestNormalisation:
        #     ### alterntaive "Not normlised"
        #     sim_post = sim_lh / np.sum(unnorm_sim_lh, axis=2, keepdims=True)
        # else:
        #     # Dirichlet decision noise
        #     sim_post = sim_lh / np.sum(sim_lh, axis=2, keepdims=True) ###############
        
        
        #sim_post = np.random.gamma(sim_post * alpha, scale=1)
        sim_post[sim_post == 0] = np.finfo(float).eps  # prevent numerical issues
        sim_post = gamma.rvs(sim_post * alpha, scale=1)
        sim_post /= np.sum(sim_post, axis=2, keepdims=True)
        if np.isnan(sim_post).any():
            print("warning sim_post has nan")


        sim_d = np.argmax(sim_post, axis=2) + 1  # category decision
        sim_post[sim_post == 0] = np.finfo(float).eps  # prevent numerical issues
        
        sim_c_en = 1 - (-np.sum(sim_post * np.log2(sim_post), axis=2) / np.log2(ncat))  # ENTROPY confidence

        # Get probability map given a set of parameter values
        sim_r = np.digitize(sim_c_en, k)  # categorical confidence report based on k
        sim_r = np.clip(sim_r, 1, 4)  # prevent numerical issues


        for tr in range(ntrl):
            for d in range(ncat):
                for r in range(nr):
                    temp = np.sum((sim_d[:, tr] == d + 1) & (sim_r[:, tr] == r + 1)) / simntrl
                    pMat[idxconfig][d, r, tr] = (1 - lapse) * temp + lapse * (1 / (ncat * nr))

        pMat[idxconfig] = np.maximum(pMat[idxconfig], np.finfo(float).eps)
        # marginalise across pMat to get, for each sample, the prob of each decision and each confidence
        pMat_dec[idxconfig]= np.sum(pMat[idxconfig],axis=1)
        pMat_conf[idxconfig]= np.sum(pMat[idxconfig],axis=0)
        
        ###plot true and model confidences 
        model_confidence=np.mean(sim_r,axis=0)
        true_confidence=trl_estC[trl_config == config]
        if attention:
            if "T" in locals() and T!=1:
                ## note that in metrics we just save attend for the first sample (eg 0 of 10k).  this is becuase, with 0 sensory noise, sig_x, they are all the same
                plot_model(temp_s,temp_sy,config,model_confidence, true_confidence,"EntropywithSoftAttentionwithTemp",subject,k, T=T, attended=None,savefile=savefile)
            else: 
                plot_model(temp_s,temp_sy,config,model_confidence, true_confidence,"EntropywithSoftAttention",subject,k, attended=None,savefile=savefile)
        else:    
            if "T" in locals() and T!=1:
                plot_model(temp_s,temp_sy,config,model_confidence, true_confidence,"EntropywithTemp",subject,k, T=T,savefile=savefile)
            else: 
                plot_model(temp_s,temp_sy,config,model_confidence, true_confidence,"Entropy",subject,k,savefile=savefile)
            
    # Negative log-likelihood calculation
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
            if attention:
                if "T" in locals() and T!=1:
                    model="EntropywithSoftAttentionwithGammawithTemp"
                else:
                    model="EntropywithSoftAttentionwithGamma"
            
            else:
                if "T" in locals() and T!=1:
                    model="EntropywithTemp"
                else:
                    model="Entropy"
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

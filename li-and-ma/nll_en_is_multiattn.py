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

def gmm_responsibilities_with_attention(x, k, likelihoods, attended_states = 1):
    
    if attended_states == 1:
        p_s_phi = np.eye(k,dtype=int) # likelihood P(s_i = 1 | phi) as indicator function for each latent
    
    # elif attended_states == 2:
    #     p_s_phi = np.eye(k) # likelihood P(s_i = 1 | phi) as 2-state indicator functions for each latent
    #     p_s_phi = np.fliplr(p_s_phi) 
    #     p_s_phi = np.where(p_s_phi == 1, 0, 1)
    elif attended_states >1 and attended_states<k:
        combinations = list(itertools.combinations(range(k), attended_states))
        rows=[]
        for combo in combinations:
            row=np.zeros(k,dtype=int)
            row[list(combo)]=1
            rows.append(row)
        p_s_phi=np.array(rows)       
    
    elif attended_states == k:
        p_s_phi = np.ones((1,k),dtype=int) # all three latents contribute
    
    else:
        raise ValueError("Not implemented for sets of Phi != 1, 2 or 3")
        
    # p_x_s = np.zeros(k,)  # list of Gaussian likelihoods P(x| s_i=1, phi)
    
    # for k in range(k):
    #     p_x_s[k] = (multivariate_normal.pdf(x, mean=means[k], cov=covariances[k]))
    p_x_s = likelihoods
    
    # pick out individual likelihood for numerator using phi
    # only use likelihoods picked out with phi
    # this gives us posterior P(s_0| x)
    likelihood=np.dot(p_x_s, p_s_phi.T)
    
    ##EITHER
    ### this calcualtes the atttended states based on average across all the data points (and the samples)
    ## note posterior is the probability of each "pair" of 2 attended states
    # avg_likelihood=np.mean(likelihood, axis=(0,1))
    # posterior = avg_likelihood / np.sum(avg_likelihood) 
    # attended = p_s_phi[np.argmax(posterior)]
    
    ##OR
    ## this treats each of the data points differnetly (but each data point averaged across the samples)
    ## note posterior is the probability of each "pair" of 2 attended states
    posterior = likelihood / np.tile(np.sum(likelihood, axis=2)[:,:,np.newaxis],(1,1,k))
    attended = p_s_phi[np.argmax(posterior,axis=-1)]
    
    
    # then need to use this posterior?
    return attended, p_s_phi, likelihood

def nll_en_is_multi(x, trl, stiPar, config_pool, subject, condIdx=None, attention =False,T_update=True, savefile=True,return_nllk_only=True,attentionTestNormalisation=False):
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
    k1 = np.sort(np.concatenate(([-np.inf], x[:3], [np.inf])))  # boundaries for 4-point confidence report
    sig_x = 10**x[4]  # sensory noise
    alpha = 10**x[3]  # Dirichlet inference noise
    lapse = x[5]
    
    # # setup where we have no sensory noise
    # sig_x =  x[4]  # Sensory noise
    # alpha = 10 ** x[3]  # Dirichlet inference noise
    # lapse = x[5]
    
    # T= x[4] # as well as ptimizing lapse, optmiize temperature parameter. ################
    # if not T_update:
    #     T=1
    ##can't use the above if we are using x[4] for sig_x
    T=1.
    
    selection_threshold=0.15
    
    # this is for model attn where are also trying to optiMize the level at wich we switch between attn2 and attn3
    if len(x)==10:
        threshold=x[9]
    else: 
        threshold=selection_threshold # note differnet usage  depending whether doing likelihood test or just based on value of lowest probability
    
    ## new bounds added in for the 2nd model, to be optimised
    k2 = np.sort(np.concatenate(([-np.inf], x[6:9], [np.inf])))  # boundaries for 4-point confidence report (2nd model)
    
    #print("lapse", lapse, "temperature", T, "alpha", alpha)
    #print("lapse",lapse, "T", T)
    
    pMat = [None] * (len(config_pool))
    pMat_dec = [None] * (len(config_pool))
    pMat_conf = [None] * (len(config_pool))
    
    total_attended_states=0 # )just for invetigation, not used
    for idxconfig, config in enumerate(config_pool):
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
            # ##### EITHER JUST DO TWO STATES
            # attended_states=2 # for now we force to just take 2 states
            # attended, p_s_phi, likelihood = gmm_responsibilities_with_attention(x, 
            #                                             k=sim_lh.shape[-1],                                                                                         
            #                                             likelihoods=sim_lh,
            #                                             attended_states=attended_states)
        
            # ##### OR DO RELATIVE LIKLIHOOD TEST TO WORK OUT HOW MANY STATES TO COVER
            # ##### STARTS ASSUMING ONE AND INCREAES NUMBER OF ATTENED STATES IF CHI^2 TEST TELLS US THA INCREASE IN LIKELIHOOD
            # ##### FROM INCREASING NUMBER OF STATES IS "WORHT IT".  NOTE THAT HE P-VALUE THRESHOLD USED SHOULD PROBABLY BE A LEARNED INDIVIDUAL PARAMTER?  
            # ##### CURRENTLY JUST USING A SET VALUE
            # first=True
            # #for attended_states in range(1,sim_lh.shape[-1]): # compare attn1 and attn2
            # for attended_states in range(2,sim_lh.shape[-1]+1):    # compare attn2 and attn3
            #     attended, p_s_phi, likelihood = gmm_responsibilities_with_attention(x, 
            #                                                                                   k=sim_lh.shape[-1],                                                                                         
            #                                                                                   likelihoods=sim_lh,
            #                                                                                   attended_states=attended_states)    
            #     if first==True:
            #         chosen_attention=attended
            #         ll_restricted=np.log(np.max(likelihood, axis=-1))
            #         chosen_log_likelihood=ll_restricted
            #         used_states=attended_states * np.ones_like(chosen_log_likelihood)
            #         first = False
            #     else:
            #         ll_full=np.log(np.max(likelihood, axis=-1))
            #         lr_stat = 2 * (ll_full - chosen_log_likelihood)
                    
                    
            #         extra_dof =  attended_states*np.ones_like(chosen_log_likelihood) - used_states  
            #         p_value = 1 - chi2.cdf(lr_stat, extra_dof)
                    
            #         ## for each of the 10k*84 samples, where p_value less than threshold, 
            #         ## update the chosen attention to be the new attention and update the chosen_log_likelihood to be the new log likelihood
            #         ## else keep the previous attention. similarly, log_likelihood and used_states
            #         threshold=0.5 # 0.5
            #         tiled_p_value=np.tile(p_value[:,:,np.newaxis], (1,1,attended.shape[-1]))
            #         chosen_attention=np.where(tiled_p_value<threshold, attended, chosen_attention)
            #         chosen_log_likelihood=np.where (p_value<threshold, ll_full, chosen_log_likelihood)
            #         used_states=np.where(p_value<threshold, attended_states*np.ones_like(chosen_log_likelihood), used_states)
            # attended=chosen_attention
            
            ######## OR (experimantal) decide whether 2 or 3 attention accoridng to hte probability of the 3rd decision
            #very experimental
            # nomralize sim_lh into probability
            my_prob=sim_lh/np.repeat(np.sum(sim_lh, axis=2)[:,:,np.newaxis],sim_lh.shape[2],axis=2)
            min_prob=np.repeat(np.min(my_prob, axis=2)[:,:,np.newaxis],sim_lh.shape[2],axis=2)
            # get the minimum probability of each sample
            attended2, p_s_phi2, likelihood2 = gmm_responsibilities_with_attention(x, 
                                                                              k=sim_lh.shape[-1],                                                                                         
                                                                              likelihoods=sim_lh,
                                                                              attended_states=2)      
            attended3, p_s_phi3, likelihood3 = gmm_responsibilities_with_attention(x, 
                                                                              k=sim_lh.shape[-1],                                                                                         
                                                                              likelihoods=sim_lh,
                                                                              attended_states=3)      
            

            attended=np.where(min_prob<threshold, attended2, attended3)
            #################################################
            
            
            
            

            # now change likelihood into likelihood after attention cut down 
            sim_lh=sim_lh * attended
            sim_lh[sim_lh == 0] = np.finfo(float).eps  # prevent numerical issues 
            ### WE COULD REMOV THE UNATTENDED COLUMNS ENTIRELY, BUT DOESNT MAKE ANY DFFERENCE TO ENTROPY MEASURE, SO LEAVE IN
        total_attended_states+=np.sum(attended)   

        ############## expt to add T
        # take the likelihood and get log likelihood
        log_sim_lh= np.log(sim_lh)
        # weight LLK by temp parameter
        log_sim_lh *= T
        # convert back to likelihood
        sim_lh = np.exp(log_sim_lh)
        ######################################################
        
        if attentionTestNormalisation:
            ### alterntaive "Not normlised"
            sim_post = sim_lh / np.sum(unnorm_sim_lh, axis=2, keepdims=True)
        else:
            # Dirichlet decision noise
            sim_post = sim_lh / np.sum(sim_lh, axis=2, keepdims=True) ###############
        
        
        #sim_post = np.random.gamma(sim_post * alpha, scale=1)
        sim_post = gamma.rvs(sim_post * alpha, scale=1)
        sim_post /= np.sum(sim_post, axis=2, keepdims=True)
        if np.isnan(sim_post).any():
            print("warning sim_post has nan")


        sim_d = np.argmax(sim_post, axis=2) + 1  # category decision
        sim_post[sim_post == 0] = np.finfo(float).eps  # prevent numerical issues
        
       
        sim_c_en = 1 - (-np.sum(sim_post * np.log2(sim_post), axis=2) / np.log2(ncat))  # ENTROPY confidence

        ### for each sample, use different k  depending on which attention has been chosen (eg 2 or 3)
        num_attended= np.sum(attended, axis=2)
        k_ids=[k1,k2]
        sim_r=np.zeros_like(sim_d)
        for k,num_attended_states in enumerate([2,3]):
            # Get probability map given a set of parameter values
            sim_r_temp = np.digitize(sim_c_en, k_ids[k])  # categorical confidence report based on k
            sim_r_temp = np.clip(sim_r_temp, 1, 4)  # prevent numerical issues
            sim_r=np.where(num_attended==num_attended_states,sim_r_temp, sim_r)

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
                plot_model(temp_s,temp_sy,config,model_confidence, true_confidence,"EntropywithAttentionwithTemp",subject,k1, T=T, attended=attended[0,:,:],savefile=savefile)
            else: 
                plot_model(temp_s,temp_sy,config,model_confidence, true_confidence,"EntropywithAttention",subject,k1, attended=attended[0,:,:],savefile=savefile)
        else:    
            if "T" in locals() and T!=1:
                plot_model(temp_s,temp_sy,config,model_confidence, true_confidence,"EntropywithTemp",subject,k1, T=T,savefile=savefile)
            else: 
                plot_model(temp_s,temp_sy,config,model_confidence, true_confidence,"Entropy",subject,k1,savefile=savefile)
     
    print("average num total_attended_states", total_attended_states/ (attended.shape[0]*attended.shape[1]*len(config_pool)) )        
    # print("from 4*ntrl, number treated as attn3", (total_attended_states/ (attended.shape[0]*attended.shape[1]*len(config_pool))-2)*(attended.shape[1]*len(config_pool)) )
    
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
                    model="EntropywithAttentionwithTemp"
                else:
                    model="EntropywithAttention"
            
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

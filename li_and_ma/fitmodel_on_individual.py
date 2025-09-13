# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:50:04 2024

@author: pfkin


MATLAB's patternsearch function is replaced with scipy.optimize.minimize using the Nelder-Mead method, 
which is a good approximation for derivative-free optimizations.
A direct alternative to BADS now exists in Python - the pybad library. 
"""

import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.optimize import linprog  # For linear constraints in pattern search
import json
import time
from pybads import BADS

from nll_map_is import nll_map_is
from nll_en_is_withgamma import nll_en_is_withgamma
from nll_diff_is import nll_diff_is
from nll_en_is_multiattn import nll_en_is_multi

def fitmodel_on_individual(exp, subject, model_type, optimize=1,savefile=True):
    """
    Inputs:
    - exp: experiment to fit (1, 2, or 3)
    - subject: subject ID (1-13 in Exp1; 1-11 in Exp2 and Exp3)
    - model_type: 1 = Max, 2 = Difference, 3 = Entropy model
    - optimize: 0 = use pattern search, 1 = use other optimization (e.g. BADS)

    Outputs:
    - output: a dictionary with best-fit parameters, NLL, AIC, x0, and exit flag
    """

    
    print(f'Fitting exp{exp} sub{subject} model{model_type} ----- ')
    filename_data = f'li_and_ma/data/EXP{exp}/S{subject}_log.mat'
    
    # Load the data (requires scipy.io or h5py depending on file format)
    # Assuming the data is saved as a .mat file (MATLAB format)
    from scipy.io import loadmat
    data = loadmat(filename_data)
    trl = data['trl']
    # need to convert trl into dict
    trl = {
        'estD' : trl[0][0][2],
        'estC' : trl[0][0][1],
        'config' : trl[0][0][0],
        'target_x' : trl[0][0][3],
        'target_y' : trl[0][0][4],
        'theta' : trl[0][0][5]
        }
    
    
    
    # Select configurations to fit
    if exp in [1, 3]:
        configpool = list(range(1,5)) #was 1-5
    elif exp == 2:
        configpool = list(range(5,9)) # was 5 -9
    
    # Initialize experiment parameters
    stiPar = {
        'ux': np.array([[-96, 0, 96], [-128, 0, 128], [-96, -59, 96], [-96, 59, 96], 
                        [-64, 0, 64], [-51, 0, 51], [-64, -64, 64], [-64, 64, 64]]),
        'uy': np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], 
                        [37, -74, 37], [30, -59, 30], [37, 0, 37], [37, 0, 37]]),
        'sig_s': 64,
        'nr': 4,
        'ncat': 3
    }
    
    # Define the objective function based on model_type
    if model_type == 1:
        # Max model
        # lb = [0, 0, 0, -20, -20, 0]
        # ub = [1, 1, 1, 20, 20, 1]
        lb = [0, 0, 0, -20, .1, 0]
        ub = [1, 1, 1, 20, 3, .05]
        plb = [0.3, 0.7, 0.9, 1, 0.5, 0]
        pub = [0.7, 0.9, 1, 1.5, 1.5, 0.01]
        objFunc = lambda x: nll_map_is(x, trl, stiPar, configpool, subject,savefile=savefile)
        
    elif model_type == 2:
        # Difference model
        # lb = [0, 0, 0, -20, -20, 0]
        # ub = [1, 1, 1, 20, 20, 1]
        lb = [0, 0, 0, -20, .1, 0]
        ub = [1, 1, 1, 20, 3, .05]
        plb = [0, 0.2, 0.5, 1, 0.5, 0]
        pub = [0.2, 0.5, 1, 1.5, 1.5, 0.01]
        objFunc = lambda x: nll_diff_is(x, trl, stiPar, configpool, subject,savefile=savefile)
    
    # elif model_type == 3:
    #     # Entropy model with temperature
    #     # lb = [0, 0, 0, -20, -20, 0]
    #     # ub = [1, 1, 1, 20, 20, 1]
    #     lb = [0,   0,   0,  -20, .1,  0]
    #     ub = [1.2, 1.2, 1.2, 20, 3, .05]
        
    #     # plb = [0, 0.2, 0.7, 1, 0.5, 0] # orig
    #     # pub = [0.2, 0.7, 1, 1.5, 1.5, 0.2] # orig
    #     plb = [0.4, 0.6, 0.8, 1, 0.5, 0]  
    #     pub = [0.6, 0.8, 1, 1.5, 1.5, 0.01]
    #     objFunc = lambda x: nll_en_is(x, trl, stiPar, configpool, subject, attention=False,T_update=True, savefile=savefile)
        
    # elif model_type == 4:
    #     # Entropy model with attention
    #     # lb = [0, 0, 0, -20, -20, 0]
    #     # ub = [1, 1, 1, 20, 20, 1]
    #     lb = [0,   0,   0,  -20, .1,   0]
    #     ub = [1.2, 1.2, 1.2, 20, 3, .05]  #### note T is very constrained to near 1
        
    #     # plb = [0, 0.2, 0.7, 1, 0.5, 0] # orig
    #     # pub = [0.2, 0.7, 1, 1.5, 1.5, 0.2] # orig
    #     plb = [0.4, 0.6, 0.8, 1,   0.5, 0]  
    #     pub = [0.6, 0.8, 1,   1.5, 1.5, 0.01]
        
    #     objFunc = lambda x: nll_en_is(x, trl, stiPar, configpool, subject, attention=True,T_update=False, savefile=savefile)#,attentionTestNormalisation=True)
    
    # elif model_type == 5:
    #     # Entropy model no temp no attention
    #     # lb = [0, 0, 0, -20, -20, 0]
    #     # ub = [1, 1, 1, 20, 20, 1]
    #     lb = [0,   0,   0,  -20, .1,   0]
    #     ub = [1.2, 1.2, 1.2, 20, 3, .05]  #### note T is very constrained to near 1
        
    #     # plb = [0, 0.2, 0.7, 1, 0.5, 0] # orig
    #     # pub = [0.2, 0.7, 1, 1.5, 1.5, 0.2] # orig
    #     plb = [0.4, 0.6, 0.8, 1,   0.9, 0]  
    #     pub = [0.6, 0.8, 1,   1.5, 1.1, 0.01]
        
    #     objFunc = lambda x: nll_en_is(x, trl, stiPar, configpool, subject, attention=False,T_update=False, savefile=savefile)

    elif model_type == 6:
        # Difference model
        # lb = [0, 0, 0, -20, -20, 0]
        # ub = [1, 1, 1, 20, 20, 1]
        lb = [0, 0, 0, -20, .1, 0]
        ub = [1, 1, 1, 20, 3, .05]
        plb = [0, 0.2, 0.5, 1, 0.5, 0]
        pub = [0.2, 0.5, 1, 1.5, 1.5, 0.01]
        objFunc = lambda x: nll_diff_is(x, trl, stiPar, configpool, subject,savefile=savefile,renormed=True)    
 
    elif model_type == 7:
        # Entropy model with multi-attention
        # note that there are 3 more boundaries added into x0 (at the end), becuase we are operating 2 models
        # so x[0] to x[5] are the same as he other modesl, but we have additionl x[6:8]
        lb = [0,   0,   0,  -20, .1,   0, 0,   0,   0]
        ub = [1.2, 1.2, 1.2, 20, 3, .05,1.2, 1.2, 1.2]  #### note T is very constrained to near 1
        
        
        plb = [0.4, 0.6, 0.8, 1,   0.5, 0, 0.4, 0.6, 0.8]  
        pub = [0.6, 0.8, 1,   1.5, 1.5, 0.01, 0.6, 0.8, 1]
        
        objFunc = lambda x: nll_en_is_multi(x, trl, stiPar, configpool, subject, attention=True,T_update=False, savefile=savefile)#,attentionTestNormalisation=True)
      
    elif model_type == 8:
        # Entropy model with multi-attention, minimzing 
        # note that there are 3 more boundaries added into x0 (at the end), becuase we are operating 2 models
        # so x[0] to x[5] are the same as he other modesl, but we have additionl x[6:8]
        lb = [0,   0,   0,  -20, .1,   0, 0,   0,   0, 0]
        ub = [1.2, 1.2, 1.2, 20, 3, .05,1.2, 1.2, 1.2, 1]  
        
        
        plb = [0.4, 0.6, 0.8, 1,   0.9, 0, 0.4, 0.6, 0.8, .1]  
        pub = [0.6, 0.8, 1,   1.5, 1.1, 0.01, 0.6, 0.8, 1, .2]
        
        objFunc = lambda x: nll_en_is_multi(x, trl, stiPar, configpool, subject, attention=True,T_update=False, savefile=savefile)#,attentionTestNormalisation=True)
      
    elif model_type == 9:
        # Entropy model with gamma on structure
        # lb = [0, 0, 0, -20, -20, 0]
        # ub = [1, 1, 1, 20, 20, 1]
        lb = [0,   0,   0,  -20, .1,  0]
        ub = [1.2, 1.2, 1.2, 20, 3, .05]
 
    
        # plb = [0, 0.2, 0.7, 1, 0.5, 0] # orig
        # pub = [0.2, 0.7, 1, 1.5, 1.5, 0.2] # orig
        plb = [0.4, 0.6, 0.8, 1, 0.5, 0]  
        pub = [0.6, 0.8, 1, 1.5, 1.5, 0.01]
        objFunc = lambda x: nll_en_is_withgamma(x, trl, stiPar, configpool, subject, attention=True,T_update=False, savefile=savefile)

    # Generate initial parameter values
    if model_type==7:
        randseed = np.random.rand(len(plb))
        x0 = plb + randseed * (np.array(pub) - np.array(plb))
        x0[:3] = np.sort(x0[:3])
        x0[6:9] = np.sort(x0[6:9])
    elif model_type==8:
            randseed = np.random.rand(len(plb))
            x0 = plb + randseed * (np.array(pub) - np.array(plb))
            x0[:3] = np.sort(x0[:3])
            x0[6:9] = np.sort(x0[6:9])
    else:
        randseed = np.random.rand(len(plb))
        x0 = plb + randseed * (np.array(pub) - np.array(plb))
        x0[:3] = np.sort(x0[:3])


    # ### for investigations only - import previously optimised x0     
    # models=["Max", "Diff","EntropywithTemp","EntropywithAttention", "Entropy","Diff_Renormed","EntropywithMultiAttentionExperimental15", "EntropywithMultiAttentionOptimizeparameter"]
    # logdir="data/outputs/multiattn_run2/"
    # filename="NLLK_Subject_"+str(subject)+"_Model_"+ models[model-1] +".json"
    # path = logdir+filename
    # with open(path, "r") as file:
    #     x0=json.load(file)['x']
    # ######################
    
    # Optimization setup
    bounds = Bounds(lb, ub)
    #bounds = Bounds(lb[:5], ub[:5]) # if only passing x0 of size 5 (lapse excluded)
    #bounds = Bounds(list(lb[i] for i in [0,1,2,3,5]), list(ub[i] for i in [0,1,2,3,5])) # if only passing x0 of size 5 (sig_x excluded)   
    if optimize == 0:
        ### Use pattern search for optimization
        ### Equivalent to MATLAB's pattern search can be tricky in Python; using Nelder-Mead as an alternative
        result = minimize(objFunc, x0, method='Nelder-Mead', bounds=bounds,options={'disp': True, 'xatol':1., 'maxiter':200})
        #result = minimize(objFunc, x0[:5], method='Nelder-Mead', bounds=bounds,options={'disp': True, 'xatol':1., 'maxiter':200})
        #result = minimize(objFunc, x0[[0,1,2,3,5]], method='Nelder-Mead', bounds=bounds,options={'disp': True, 'xatol':1., 'maxiter':200})
        
        estX = result.x
        fval = result.fun
        status=result.status
    
    
    
    elif optimize == 1:
    # ## Use BADS for optimization; Download BADS here: https://github.com/lacerbi/bads
        pass    
        # nonbcon = @(x) x(:,1)>x(:,2) | x(:,2)>x(:,3);
        # [estX,fval,exitflag] = bads(objFunc,x0,lb,ub,plb,pub,nonbcon,options)
        bads = BADS(objFunc, x0, lb, ub, plb, pub, options={'max_iter': 75}) # just chosen 75 max iters because takes a long tim if allow default of 200*D.  and seems to have settled by then usually
        # bads = BADS(objFunc, x0, lb, ub, plb, pub, ) # force it to treat as a stochastic  fucntion
        
        result = bads.optimize()
        
        estX = result.x
        fval = result.fval
        status=result.success
    else:
        print("Optimizer not recognised")


    AIC = 2 * (len(x0) + fval)

    
    output = {
        'par': estX,
        'AIC': AIC,
        'nll': fval,
        'exitflag': status,
        'x0': x0
    }
    
    print(f'Done fitting Experiment {exp} Subject {subject} Model {model_type}')
    print(f'AIC score {AIC:.1f}')
    print(f'nll score {fval:.1f}')
    return output


if __name__ == "__main__":
  #time.sleep(60*40)
  #Experiment to fit (1,2 or 3)
  exp = 1
  for model in [1]: #[2,4,5,6,8]:
    for subject in [1,2,3,4,5,6,7,8,9,10,11,12,13]:
        #Subject to fit (There are 13 subjects in Experiment 1, 11 subjects in Experiment 2, and 11 subjects in Experiment 3)
        
        #Model to fit
        #1: Max model (with both sensory nosie and inference noise)
        #2: Diff model
        #3: Ent model. progrmamed to have T updating
        #4: Ent model with attention. prgramed to have T=1
        #5: straight entropy, no attention, no temperature
        #6: Diff Renormed
        #7: Ent model with multiattention
        #8: Ent model with multiattention, learning parameer x[9] which optiimses trade off beween atn2 and attn3
        #9 Ent model with attention including gamma weighting on structure
        #model = 4 #################################
        # optimizer to use
        optimize=1
        
        savefile=False # savefile =false so doesnt save all the "attempts" by the underlying model
        output = fitmodel_on_individual(exp, subject, model, optimize,savefile=savefile) 

        ###########################
        # save final optimised data

        models=["Max", "Diff","EntropywithTemp","EntropywithAttention", "Entropy","Diff_Renormed","EntropywithMultiAttentionExperimental15", "EntropywithMultiAttentionOptimizeparameter", "EntropywithSoftAttentionwithGamma_inf"]
        #where to save data
        print("nllk ", np.round(output['nll']))
        #logdir='data/Outputs/trained_for_jointcount_pybads_new_sig_x=0_Ti/'
        #logdir='data/Outputs/expt2_correct_pybads_sqrt_sig_x=0_Ta/'
        #logdir='data/Outputs/correct_pybads_sqrt_sig_x=0_T/'
        logdir="data/outputs/"
        #logdir='data/Outputs/expt2_correct_pybads_sqrt_Td/'
        filename="NLLK_Subject_"+str(subject)+"_Model_"+ models[model-1] +".json"
        
        metrics = {"NLLK": float(output['nll']), "x": output['par'].tolist(), "model": models[model-1], "ind":subject}
        print(metrics)
        path = logdir+filename
        try:
            with open(path, "w") as file:
                json.dump(metrics, file)         
                #utils.save_json(metrics, logdir + "metrics.json")
                #data_to_save=np.stack((x_middled_model,slide_model_confidence,x_middled_true,slide_true_confidence))
                #np.save(logdir+filename,data_to_save)
        except TypeError as e:
            # Catch the TypeError if an object is not JSON serializable
            print(f"JSON serialization failed: {e}")
        except ValueError as e:
            # Catch ValueError for other JSON-related issues (less common with dumping)
            print(f"Value error during JSON operation: {e}")
        except Exception as e:
            # Catch any other unexpected errors
            print(f"An unexpected error occurred: {e}")
        except Exception as e:
            print(f"something else went wrong: {e}") 
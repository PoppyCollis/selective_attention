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
from scipy.io import loadmat
import json
from pybads import BADS
# local imports
from nll_map_is import nll_map_is
from nll_diff_is import nll_diff_is

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
        configpool = list(range(1,5)) 
    elif exp == 2:
        configpool = list(range(5,9)) 
    
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
        lb = [0, 0, 0, -20, .1, 0]
        ub = [1, 1, 1, 20, 3, .05]
        plb = [0.3, 0.7, 0.9, 1, 0.5, 0]
        pub = [0.7, 0.9, 1, 1.5, 1.5, 0.01]
        objFunc = lambda x: nll_map_is(x, trl, stiPar, configpool, subject,savefile=savefile)
        
    elif model_type == 2:
        # Difference model
        lb = [0, 0, 0, -20, .1, 0]
        ub = [1, 1, 1, 20, 3, .05]
        plb = [0, 0.2, 0.5, 1, 0.5, 0]
        pub = [0.2, 0.5, 1, 1.5, 1.5, 0.01]
        objFunc = lambda x: nll_diff_is(x, trl, stiPar, configpool, subject,savefile=savefile)
        
    elif model_type == 3:
        # Entropy model
        lb = [0, 0, 0, -20, -20, 0]
        ub = [1, 1, 1, 20, 20, 1]
        lb = [0,   0,   0,  -20, .1,   0]
        ub = [1.2, 1.2, 1.2, 20, 3, .05]  #### note T is very constrained to near 1
    
    # Generate initial parameter values
    randseed = np.random.rand(len(plb))
    x0 = plb + randseed * (np.array(pub) - np.array(plb))
    x0[:3] = np.sort(x0[:3])

    # Optimization setup
    bounds = Bounds(lb, ub)
    if optimize == 0:
        # Use pattern search for optimization
        # Equivalent to MATLAB's pattern search can be tricky in Python; using Nelder-Mead as an alternative
        result = minimize(objFunc, x0, method='Nelder-Mead', bounds=bounds,options={'disp': True, 'xatol':1., 'maxiter':200})
        estX = result.x
        fval = result.fun
        status=result.status
        
    elif optimize == 1:
        bads = BADS(objFunc, x0, lb, ub, plb, pub, options={'max_iter': 75}) # just chosen 75 max iters because takes a long tim if allow default of 200*D
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

def main():
    exp = 1 # Experiment to fit (1,2,3)
    for model in [1]:
        for subject in [1,2,3,4,5,6,7,8,9,10,11,12,13]:
        
            # optimizer to use
            optimize=1
            savefile=False # savefile =false so doesnt save all the "attempts" by the underlying model
            output = fitmodel_on_individual(exp, subject, model, optimize,savefile=savefile) 

            models=["Max", "Diff"]
            print("nllk ", np.round(output['nll']))
            logdir="data/outputs/"
            filename="NLLK_Subject_"+str(subject)+"_Model_"+ models[model-1] +".json"
            metrics = {"NLLK": float(output['nll']), "x": output['par'].tolist(), "model": models[model-1], "ind":subject}
            print(metrics)
            path = logdir+filename
            try:
                with open(path, "w") as file:
                    json.dump(metrics, file)         
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


if __name__ == "__main__":
    main()
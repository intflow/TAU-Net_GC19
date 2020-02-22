# SPARSE_NMF Sparse NMF with beta-divergence reconstruction error, 
# L1 sparsity constraint, optimization in normalized basis vector space.
#
# [w, h, objective] = sparse_nmf(v)
#
# Inputs:
# v:  matrix to be factorized
# p: optional parameters
#     beta:     beta-divergence parameter (default: 1, i.e., KL-divergence)
#     cf:       cost function type (default: 'kl'; overrides beta setting)
#               'is': Itakura-Saito divergence
#               'kl': Kullback-Leibler divergence
#               'kl': Euclidean distance
#     sparsity: weight for the L1 sparsity penalty (default: 0)
#     max_iter: maximum number of iterations (default: 100)
#     conv_eps: threshold for early stopping (default: 0, 
#                                             i.e., no early stopping)
#     display:  display evolution of objective function (default: 0)
#     random_seed: set the random seed to the given value 
#                   (default: 1; if equal to 0, seed is not set)
#     init_w:   initial setting for W (default: random; 
#                                      either init_w or r have to be set)
#     r:        # basis functions (default: based on init_w's size;
#                                  either init_w or r have to be set)
#     init_h:   initial setting for H (default: random)
#     w_update_ind: set of dimensions to be updated (default: all)
#     h_update_ind: set of dimensions to be updated (default: all)
#
# Outputs:
# w: matrix of basis functions
# h: matrix of activations
# objective: objective function values throughout the iterations
#
#
#
# References: 
# J. Eggert and E. Korner, "Sparse coding and NMF," 2004
# P. D. O'Grady and B. A. Pearlmutter, "Discovering Speech Phones 
#   Using Convolutive Non-negative Matrix Factorisation
#   with a Sparseness Constraint," 2008
# J. Le Roux, J. R. Hershey, F. Weninger, "Sparse NMF ? half-baked or well 
#   done?," 2015
#
# This implementation follows the derivations in:
# J. Le Roux, J. R. Hershey, F. Weninger, 
# "Sparse NMF ? half-baked or well done?," 
# MERL Technical Report, TR2015-023, March 2015
#
# If you use this code, please cite:
# J. Le Roux, J. R. Hershey, F. Weninger, 
# "Sparse NMF ? half-baked or well done?," 
# MERL Technical Report, TR2015-023, March 2015
#   @TechRep{LeRoux2015mar,
#     author = {{Le Roux}, J. and Hershey, J. R. and Weninger, F.},
#     title = {Sparse {NMF} -?half-baked or well done?},
#     institution = {Mitsubishi Electric Research Labs (MERL)},
#     number = {TR2015-023},
#     address = {Cambridge, MA, USA},
#     month = mar,
#     year = 2015
#   }
#
############################################################################
#   Copyright (C) 2015 Mitsubishi Electric Research Labs (Jonathan Le Roux,
#                                         Felix Weninger, John R. Hershey)
#   Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
#############################################################################
#  
#   Matlab to Python Conversion worked by Kwang Myung Jeon. 2018
#
#############################################################################

import numpy as np
import scipy.io as sio

def sparse_nmf(v, max_iter=100, random_seed=1, sparsity=5, conv_eps=0.0001, cf='kl', \
               init_w=None, init_h=None, w_ind=None, h_ind=None,  r=100, display=1, cost_check=1) :


    try:
        [m, n] = v.shape
    except:
        m = len(v)
        n = 1
        v = v.reshape((m,1))


    if cf=='is':
        beta = 0
    elif cf=='kl':
        beta = 1
    elif cf=='ed':
        beta = 2

    if random_seed > 0:
        np.random.seed(random_seed)


    #initialize update indices of w and h
    if w_ind == None :
        w_ind = range(0,r)
        
    if h_ind == None :
        h_ind = range(0,r)

    update_w = np.sum(w_ind)
    update_h = np.sum(h_ind)

    if update_w > 0:
        w_ind_fix = range(0, w_ind[0])
    else:
        w_ind_fix = range(0,r)

    if update_h > 0:
        h_ind_fix = range(0, h_ind[0])
    else:
        h_ind_fix = range(0,r)


    #Define given basis and activations (Given + Random[To be learned])
    try:
        if init_w == None:
            w = np.random.rand(m, r)
    except:
        [m, z_i] = init_w.shape
        w = np.concatenate((init_w, np.random.rand(m, r-z_i)), axis = 1)

    try:
        if init_h == None: 
            h = np.random.rand(r, n)
    except:
        [z_i, n] = init_h.shape
        h = np.concatenate((init_h, np.random.rand(r-z_i, n)), axis = 0)

    # sparsity per matrix entry
    sparsity = np.ones([r, n]) * sparsity
    
    # Normalize the columns of W and rescale H accordingly
    wn = np.sqrt(np.sum(w ** 2, axis=0))
    w  = w / wn
    h  = (h.T * wn).T


    flr = 1e-5
    v_lambda = np.maximum(w @ h, flr)
    last_cost = float('inf')
    v= np.maximum(v, flr)

    objective = {'div':np.zeros([1,max_iter]), 'cost':np.zeros([1,max_iter])}

    div_beta = beta

    if display != 0:
        print('Performing sparse NMF with beta-divergence, beta=' + str(div_beta) + '\n')
    
    for it in range(0, max_iter):

        # H updates
        if update_h > 0:
            if div_beta == 1:
                dph = np.tile(np.sum(w[:,h_ind], axis=0).T, (n,1)).T + sparsity
                dph = np.maximum(dph, flr)
                dmh = w[:, h_ind].T @ (v / v_lambda)
                h[h_ind, :] = (h[h_ind, :] * dmh) / dph
            elif div_beta == 2:
                dph = w[:,h_ind].T @ v_lambda + sparsity
                dph = np.maximum(dph, flr)
                dmh = w[:,h_ind].T @ v
                h[h_ind, :] = h[h_ind, :] * dmh / dph
            else :
                dph = w[:,h_ind].T @ v_lambda**(div_beta - 1) + sparsity
                dph = np.maximum(dph, flr)
                dmh = w[:,h_ind].T @ (v * v_lambda**(div_beta - 2))
                h[h_ind, :] = h[h_ind, :] * dmh / dph                
            
            v_lambda = np.maximum(w @ h, flr)
        


        # W updates
        if update_w > 0 :
            if div_beta == 1:
                dpw = np.tile(np.sum(h[w_ind, :], axis=1), (m,1)) + \
                      (np.sum(((v / v_lambda) @ h[w_ind, :].T) * w[:,w_ind], axis=0).T * w[:,w_ind])
                dpw = np.maximum(dpw, flr)
                dmw = v / v_lambda @ h[w_ind, :].T \
                    + np.tile(np.sum((np.sum(h[w_ind, :], axis=1) * w[:,w_ind]).T, axis=1),(m,1)) * w[:,w_ind]
                w[:,w_ind] = w[:,w_ind] * dmw / dpw
            elif div_beta == 2:
                dpw = v_lambda @ h[w_ind, :].T + np.tile(np.sum(v @ h[w_ind, :].T * w[:,w_ind], axis=0),(m,1)) * w[:,w_ind]
                dpw = np.maximum(dpw, flr)
                dmw = v @ h[w_ind, :].T + np.tile(np.sum(v_lambda @ h[w_ind, :].T * w[:,w_ind], axis=0),(m,1)) * w[:,w_ind]
                w[:,w_ind] = w[:,w_ind] * dmw / dpw
            else :
                dpw = v_lambda ** (div_beta - 1) @ h[w_ind, :].T \
                    + np.tile(np.sum((v * v_lambda ** (div_beta - 2)) @ h[w_ind, :].T * w[:,w_ind], axis=0), (m,1)) * w[:,w_ind]
                dpw = np.maximum(dpw, flr)
                dmw = (v * v_lambda**(div_beta - 2)) @ h[w_ind, :].T \
                    + np.tile(np.sum(v_lambda**(div_beta - 1) @ h[w_ind, :].T * w[:,w_ind], axis=0), (m,1)) * w[:,w_ind]
                w[:,w_ind] = w[:,w_ind] * dmw / dpw
            
            # Normalize the columns of W
            w = w / np.sqrt(np.sum(w**2, axis=0))
            v_lambda = np.maximum(w @ h, flr)
        

        #compute the objective function
        if div_beta == 1:
            div = np.sum(np.sum(v * np.log(v / v_lambda) - v + v_lambda, axis=0), axis=0)
        elif div_beta == 2:
            div = np.sum(np.sum((v - v_lambda) ** 2, axis=0), axis=0)
        elif div_beta == 0:
            div = np.sum(np.sum(v / v_lambda - np.log( v / v_lambda) - 1, axis=0), axis=0) 
        else:
            div = np.sum(np.sum(v**div_beta + (div_beta - 1)@v_lambda**div_beta \
                - div_beta @ v ** v_lambda**(div_beta - 1), axis=0), axis=0) / (div_beta @ (div_beta - 1))
        
        cost = div + np.sum(np.sum(sparsity * h, axis=0), axis=0)

        if it == 0:
            objective = cost
        else :
            objective = np.append(objective, cost)

        if display != 0:
            print('iteration {0} div = {1} cost = {2}'.format(it, div, cost))
        
        # Convergence check
        if it > 1 and conv_eps > 0 :
            e = np.abs(cost - last_cost) / last_cost
            if (e < conv_eps) and cost_check:
                if display != 0:
                    print('Convergence reached, aborting iteration')
                
                break
            
        
        last_cost = cost
        
    
    if display != 0:
        print('\nMax Iteration reached, aborting iteration\n')


    return [w+flr, h+flr, objective]

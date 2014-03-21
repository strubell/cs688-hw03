'''
CS688 HW03: Monte Carlo De-noising

Functions facilitating Monte Carlo de-noising of images.

@author: Emma Strubell
'''
import numpy as np
from numpy.random import rand

def denoise_binary(w_l, w_p, image, original, num_iters=100, tol=1e-5, run_until_convergence=True):
    # initialize y to noisy data
    y = np.copy(image)
    rows, cols = image.shape
    im_not = np.logical_not(image)
    
    # duplicate matrix with empty borders
    with_border = np.zeros((rows+2,cols+2))
    with_border_not = np.zeros((rows+2,cols+2))
    
    neighbor_offsets = [[0,-1], [1,0], [0,1], [-1,0]]
    neighbors = [[None for j in range(rows)] for i in range(cols)]
    for i in range(cols):
        for j in range(rows):
            neighbors[i][j] = [(n[0]+i+1,n[1]+j+1) for n in neighbor_offsets]
    
    print "Using w_l = %g, w_p = %g" % (w_l, w_p)
    converged = False
    iter = 0
    mae = 0.0
    maes = np.zeros(500)
    while not converged:
        with_border[1:rows+1,1:cols+1]=y*w_p
        with_border_not[1:rows+1,1:cols+1]=np.logical_not(y)*w_p
        numerators = np.exp(np.sum(np.reshape([[with_border[n] + w_l*image[i,j] for n in neighbors[i][j]] for j in range(rows) for i in range(cols)], (rows,cols,4), order='F'), axis=2))
        denominators = numerators + np.exp(np.sum(np.reshape([[with_border_not[n] + w_l*im_not[i,j] for n in neighbors[i][j]] for j in range(rows) for i in range(cols)], (rows,cols,4), order='F'), axis=2)) 
        p_ys = numerators/denominators
        rands = rand(rows, cols)
        y = np.where(rands < p_ys, 1.0, 0.0)
        last_mae = mae
        mae = compute_MAE(y, original)
        maes[iter] = mae
        iter += 1
        converged = np.abs(mae-last_mae) < tol if run_until_convergence else iter == num_iters
        print "Iteration %d MAE=%g" % (iter, mae)
    return y, maes.trim()

def compute_MAE(image, original): return np.mean(np.abs(image-original))


'''
CS688 HW03: Monte Carlo De-noising

Functions facilitating Monte Carlo de-noising of images.

@author: Emma Strubell
'''
import numpy as np
from numpy.random import rand, normal

def denoise_binary(w_l, w_p, image, original, num_iters=100, tol=1e-6, run_until_convergence=True):
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
    neighbor_counts = np.sum(np.reshape([[1.0 if ( 0 < n[0] < rows+1 and 0 < n[1] < cols+1) else 0.0 for n in neighbors[i][j]] for j in range(rows) for i in range(cols)], (rows,cols,4), order='F'), axis=2)

    print "Using w_l = %g, w_p = %g" % (w_l, w_p)
    converged = False
    iter = 0
    mean_MAE = 0.0
    ys = np.zeros((500,rows,cols))
    maes = np.zeros(500)
    while not converged:
        with_border[1:rows+1,1:cols+1]=y*w_p
        with_border_not[1:rows+1,1:cols+1]=np.logical_not(y)*w_p
        numerators = np.exp(np.sum(np.reshape([[with_border[n] + w_l*image[i,j] for n in neighbors[i][j]] for j in range(rows) for i in range(cols)], (rows,cols,4), order='F'), axis=2))
        denominators = numerators + np.exp(np.sum(np.reshape([[with_border_not[n] + w_l*im_not[i,j] for n in neighbors[i][j]] for j in range(rows) for i in range(cols)], (rows,cols,4), order='F'), axis=2)) 
        p_ys = numerators/denominators
        rands = rand(rows, cols)
        ys[iter] = np.where(rands < p_ys, 1.0, 0.0)
        maes[iter] = compute_MAE(np.mean(ys[:iter+1],axis=0), original)
        if iter > 0:
            converged = np.abs(maes[iter]-maes[iter-1]) < tol if run_until_convergence else iter == num_iters
        iter += 1
        print "Iteration %d MAE=%g" % (iter, maes[iter-1])
    return np.mean(ys[:iter],axis=0), maes[:iter]

def denoise_greyscale(w_l, w_p, image, original, num_iters=100, tol=1e-6, run_until_convergence=True):
    # initialize y to noisy data
    y = np.copy(image)
    rows, cols = image.shape
    
    # duplicate matrix with empty borders
    with_border = np.zeros((rows+2,cols+2))
    with_border_not = np.zeros((rows+2,cols+2))
    
    neighbor_offsets = [[0,-1], [1,0], [0,1], [-1,0]]
    neighbors = [[None for j in range(rows)] for i in range(cols)]
    for i in range(cols):
        for j in range(rows):
            neighbors[i][j] = [(n[0]+i+1,n[1]+j+1) for n in neighbor_offsets]
    neighbor_counts = np.sum(np.reshape([[1.0 if ( 0 < n[0] < rows+1 and 0 < n[1] < cols+1) else 0.0 for n in neighbors[i][j]] for j in range(rows) for i in range(cols)], (rows,cols,4), order='F'), axis=2)
    norm = 1/(w_p*neighbor_counts + w_l)
    sigmas = np.sqrt(0.5*norm)
        
    print "Using w_l = %g, w_p = %g" % (w_l, w_p)
    converged = False
    iter = 0
    mean_MAE = 0.0
    ys = np.zeros((500,rows,cols))
    maes = np.zeros(500)
    while not converged:
        with_border[1:rows+1,1:cols+1]=y
        #p_ys = np.exp(-np.sum(np.reshape([[w_p*(with_border[n]-y[i,j])**2 + w_l*(image[i,j]-y[i,j])**2 for n in neighbors[i][j]] for j in range(rows) for i in range(cols)], (rows,cols,4), order='F'), axis=2))
        neighbor_sums = np.sum(np.reshape([[w_p*with_border[n] for n in neighbors[i][j]] for j in range(rows) for i in range(cols)], (rows,cols,4), order='F'), axis=2)
        #print neighbor_sums
        mus = norm*(w_l*image + neighbor_sums)
        rands = np.reshape(normal(size=rows*cols), (rows,cols))
        ys[iter] = mus + rands*sigmas
        maes[iter] = compute_MAE(np.mean(ys[:iter+1],axis=0), original)
        if iter > 0:
            converged = np.abs(maes[iter]-maes[iter-1]) < tol if run_until_convergence else iter == num_iters
        iter += 1
        print "Iteration %d MAE=%g" % (iter, maes[iter-1])
    return np.mean(ys[:iter],axis=0), maes[:iter]

def compute_MAE(image, original): return np.mean(np.abs(image-original))


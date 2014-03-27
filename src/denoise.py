'''
CS688 HW03: Monte Carlo De-noising

Functions facilitating Monte Carlo de-noising of images.

@author: Emma Strubell
'''
import numpy as np
from numpy.random import rand, normal

def denoise_binary_checkerboard(w_l, w_p, image, original, num_iters=100, tol=1e-4, run_until_convergence=True):
    # initialize y to noisy data
    rows, cols = image.shape
    im_not = np.logical_not(image)
    
    # for checkerboard
    grid = np.empty((2,rows,cols))
    coords = np.ogrid[0:rows, 0:cols]
    idxs = (coords[0] + coords[1]) % 2
    vals = np.array([0.0, 1.0])
    grid[0] = vals[idxs]
    grid[1] = np.logical_not(grid[0])
    
    # duplicate matrix with empty borders
    with_border = np.zeros((rows+2,cols+2))
    
    neighbor_offsets = [[0,-1], [1,0], [0,1], [-1,0]]
    neighbors = [[None for j in range(rows)] for i in range(cols)]
    for i in range(cols):
        for j in range(rows):
            neighbors[i][j] = [(n[0]+i+1,n[1]+j+1) for n in neighbor_offsets]
    neighbor_counts = np.sum(np.reshape([[1.0 if ( 0 < n[0] < rows+1 and 0 < n[1] < cols+1) else 0.0 for n in neighbors[i][j]] for j in range(rows) for i in range(cols)], (rows,cols,4), order='F'), axis=2)

    print "Using w_l = %g, w_p = %g" % (w_l, w_p)
    
    mean_MAE = 0.0
    ys = np.zeros((500,rows,cols))
    maes = np.zeros(500)
    ys[0] = np.copy(image)
    maes[0] = compute_MAE(ys[0], original)
    iter = 1
    checker = 0
    converged = False
    while not converged:
        # fill in center with new y vals
        with_border[1:rows+1,1:cols+1]=ys[iter-1]
        neighbor_counts_set = np.sum(np.reshape([[with_border[n[0],n[1]] for n in neighbors[i][j]] for j in range(rows) for i in range(cols)], (rows,cols,4), order='F'), axis=2)
        neighbor_counts_not = neighbor_counts - neighbor_counts_set
        numerators = np.exp(w_p*neighbor_counts_set + w_l*image)
        denominators = numerators + np.exp(w_p*neighbor_counts_not + w_l*im_not)
        p_ys = numerators/denominators
        rands = rand(rows, cols)
        ys[iter] = np.where(rands < p_ys, 1.0, 0.0)*grid[checker]
        ys[iter] += ys[iter-1]*grid[~checker]
        maes[iter] = compute_MAE(np.mean(ys[:iter+1],axis=0), original)
        converged = np.abs(maes[iter]-maes[iter-1]) < tol if run_until_convergence else iter == num_iters
        print "Iteration %d MAE=%g" % (iter, maes[iter])
        iter += 1
        checker = ~checker
    return np.mean(ys[:iter],axis=0), maes[:iter]

def denoise_greyscale_checkerboard(w_l, w_p, image, original, num_iters=100, tol=1e-5, run_until_convergence=True):
    # initialize y to noisy data
    rows, cols = image.shape
    
    # duplicate matrix with empty borders
    with_border = np.zeros((rows+2,cols+2))
    
    # for checkerboard
    grid = np.empty((2,rows,cols))
    coords = np.ogrid[0:rows, 0:cols]
    idxs = (coords[0] + coords[1]) % 2
    vals = np.array([0.0, 1.0])
    grid[0] = vals[idxs]
    grid[1] = np.logical_not(grid[0])
    
    neighbor_offsets = [[0,-1], [1,0], [0,1], [-1,0]]
    neighbors = [[None for j in range(rows)] for i in range(cols)]
    for i in range(cols):
        for j in range(rows):
            neighbors[i][j] = [(n[0]+i+1,n[1]+j+1) for n in neighbor_offsets]
    neighbor_counts = np.sum(np.reshape([[1.0 if ( 0 < n[0] < rows+1 and 0 < n[1] < cols+1) else 0.0 for n in neighbors[i][j]] for j in range(rows) for i in range(cols)], (rows,cols,4), order='F'), axis=2)
    norm = 1/(w_p*neighbor_counts + w_l)
    sigmas = np.sqrt(0.5*norm)
        
    print "Using w_l = %g, w_p = %g" % (w_l, w_p)
    
    ys = np.zeros((500,rows,cols))
    maes = np.zeros(500)
    ys[0] = np.copy(image)
    maes[0] = compute_MAE(ys[0], original)
    iter = 1
    converged = False
    checker = 0
    while not converged:
        with_border[1:rows+1,1:cols+1]=ys[iter-1]
        neighbor_sums = np.sum(np.reshape([[w_p*with_border[n] for n in neighbors[i][j]] for j in range(rows) for i in range(cols)], (rows,cols,4), order='F'), axis=2)
        mus = norm*(w_l*image + neighbor_sums)
        rands = np.reshape(normal(size=rows*cols), (rows,cols))
        ys[iter] = (mus + rands*sigmas)*grid[checker]
        ys[iter] += ys[iter-1]*grid[~checker]
        maes[iter] = compute_MAE(np.mean(ys[:iter+1],axis=0), original)
        converged = np.abs(maes[iter]-maes[iter-1]) < tol if run_until_convergence else iter == num_iters
        print "Iteration %d MAE=%g" % (iter, maes[iter])
        iter += 1
        checker = ~checker
    return np.mean(ys[:iter],axis=0), maes[:iter]

def denoise_greyscale_checkerboard2(w_l, w_p, image, original, num_iters=100, tol=1e-4, run_until_convergence=True):
    # initialize y to noisy data
    rows, cols = image.shape
    
    # duplicate matrix with empty borders
    with_border = np.zeros((rows+2,cols+2))
    with_border_im = np.zeros((rows+2,cols+2))
    with_border_im[1:rows+1,1:cols+1] = image
    
    # for checkerboard
    grid = np.empty((2,rows,cols))
    coords = np.ogrid[0:rows, 0:cols]
    idxs = (coords[0] + coords[1]) % 2
    vals = np.array([0.0, 1.0])
    grid[0] = vals[idxs]
    grid[1] = np.logical_not(grid[0])
    
    neighbor_offsets = [[0,-1], [1,0], [0,1], [-1,0]]
    neighbors = [[None for j in range(rows)] for i in range(cols)]
    for i in range(cols):
        for j in range(rows):
            neighbors[i][j] = [(n[0]+i+1,n[1]+j+1) for n in neighbor_offsets]
    
    # unique weight for each pairwise factor
    w_pijkl = np.reshape([[w_p/(0.01+(image[i,j]-with_border_im[n])**2) for n in neighbors[i][j]] for j in range(rows) for i in range(cols)], (rows,cols,4), order='F')    
    weight_sums = np.sum(w_pijkl*np.reshape([[1.0 if ( 0 < n[0] < rows+1 and 0 < n[1] < cols+1) else 0.0 for n in neighbors[i][j]] for j in range(rows) for i in range(cols)], (rows,cols,4), order='F'), axis=2)
    
    norm = 1/(weight_sums + w_l)
    sigmas = np.sqrt(0.5*norm)
    
    print "Using w_l = %g, w_p = %g" % (w_l, w_p)
    
    ys = np.zeros((500,rows,cols))
    maes = np.zeros(500)
    ys[0] = np.copy(image)
    maes[0] = compute_MAE(ys[0], original)
    iter = 1
    converged = False
    checker = 0
    while not converged:
        with_border[1:rows+1,1:cols+1]=ys[iter-1]
        neighbor_sums = np.sum(w_pijkl*np.reshape([[with_border[n] for n in neighbors[i][j]] for j in range(rows) for i in range(cols)], (rows,cols,4), order='F'), axis=2)
        mus = norm*(w_l*image + neighbor_sums)
        rands = np.reshape(normal(size=rows*cols), (rows,cols))
        ys[iter] = (mus + rands*sigmas)*grid[checker]
        ys[iter] += ys[iter-1]*grid[~checker]
        maes[iter] = compute_MAE(np.mean(ys[:iter+1],axis=0), original)
        converged = np.abs(maes[iter]-maes[iter-1]) < tol if run_until_convergence else iter == num_iters
        print "Iteration %d MAE=%g" % (iter, maes[iter])
        iter += 1
        checker = ~checker
    return np.mean(ys[:iter],axis=0), maes[:iter]

def compute_MAE(image, original): return np.mean(np.abs(image-original))


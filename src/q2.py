'''
CS688 HW03: Monte Carlo De-noising

Question 1: Monte Carlo De-noising for greyscale images (Gaussian CRF)

@author: Emma Strubell
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.random import rand
import denoise

orig_fname = "../data/swirl.txt"
noise_fname = "../data/swirl-noise.txt"

orig_data = np.loadtxt(orig_fname)
noise_data = np.loadtxt(noise_fname)

do_plot = True

# initialize w_p, w_l to (single) positive value each

# for first version
w_l = 200.0
w_p = 100.0

# for second version
w_l2 = 90000.0
w_p2 = 1000.0

y, maes = denoise.denoise_greyscale_checkerboard(w_l, w_p, noise_data, orig_data, run_until_convergence=False)
y2, maes2 = denoise.denoise_greyscale_checkerboard2(w_l2, w_p2, noise_data, orig_data, run_until_convergence=False)
print "Final MAE single pairwise weight:", maes[-1]
print "Final MAE variable pairwise weights:", maes2[-1]

if(do_plot):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title("Original data (greyscale)")
    ax1.imshow(orig_data, interpolation='nearest', cmap='gray')
      
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title("Noisy data (greyscale)")
    ax2.imshow(noise_data, interpolation='nearest', cmap='gray')
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.set_title("Denoised data single pairwise weight (greyscale) (MAE=%g)" % (maes[-1]))
    ax3.imshow(y, interpolation='nearest', cmap='gray')
    
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    ax5.set_title("Denoised data variable pairwise weights (greyscale) (MAE=%g)" % (maes2[-1]))
    ax5.imshow(y2, interpolation='nearest', cmap='gray')
    
#     fig4 = plt.figure()
#     ax4 = fig4.add_subplot(111)
#     ax4.set_title("Change in MAE until convergence (W^L=%g, W^P=%g)" % (w_l, w_p))
#     ax4.set_xlabel("Iteration")
#     ax4.set_ylabel("Mean absolute error")
#     ax4.plot(np.arange(maes.shape[0]), maes)
    
    plt.show()

'''
CS688 HW03: Monte Carlo De-noising

Question 1: Monte Carlo De-noising for binary images

@author: Emma Strubell
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.random import rand
import denoise

orig_fname = "../data/stripes.txt"
noise_fname = "../data/stripes-noise.txt"

orig_data = np.loadtxt(orig_fname)
noise_data = np.loadtxt(noise_fname)

do_plot = True

# initialize w_p, w_l to (single) positive value each

# over-smoothed
w_lo = 0.1
w_po = 100.0

# under-smoothed
w_lu = 5.0
w_pu = 1.0

# good
w_l = 5.0
w_p = 4.0

y, maes = denoise.denoise_binary_checkerboard(w_l, w_p, noise_data, orig_data, run_until_convergence=True)
yu, maesu = denoise.denoise_binary_checkerboard(w_lu, w_pu, noise_data, orig_data, run_until_convergence=False)
yo, maeso = denoise.denoise_binary_checkerboard(w_lo, w_po, noise_data, orig_data, run_until_convergence=False)
print "Final MAE:", maes[-1]

if(do_plot):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title("Original data (binary)")
    ax1.imshow(orig_data, interpolation='nearest', cmap='gray')
      
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title("Noisy data (binary)")
    ax2.imshow(noise_data, interpolation='nearest', cmap='gray')
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.set_title("Denoised data (binary) (good) (MAE=%g)" % (maes[-1]))
    ax3.imshow(y, interpolation='nearest', cmap='gray')
    
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    ax5.set_title("Denoised data (binary) (oversmoothed) (MAE=%g)" % (maeso[-1]))
    ax5.imshow(yo, interpolation='nearest', cmap='gray')
    
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(111)
    ax6.set_title("Denoised data (binary) (undersmoothed) (MAE=%g)" % (maesu[-1]))
    ax6.imshow(yu, interpolation='nearest', cmap='gray')
    
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.set_title("Change in MAE until convergence (W^L=%g, W^P=%g)" % (w_l, w_p))
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Mean absolute error")
    ax4.plot(np.arange(maes.shape[0]), maes)
    
    plt.show()

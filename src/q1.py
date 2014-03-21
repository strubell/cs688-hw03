'''
CS688 HW03: Monte Carlo De-noising

Question 1: Monte Carlo De-noising for Binary Images

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
#w_l = 0.0001
#w_p = 0.9

# under-smoothed
#w_l = 0.5
#w_p = 0.5

# good
w_l = 1.0
w_p = 20.0

y, maes = denoise.denoise_binary(w_l, w_p, noise_data, orig_data, run_until_convergence=True)
print "Final MAE:", maes[-1]

if(do_plot):
#     fig1 = plt.figure()
#     ax1 = fig1.add_subplot(111)
#     ax1.set_title("Original data (binary)")
#     ax1.imshow(orig_data, interpolation='nearest', cmap='Greys')
      
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title("Noisy data (binary)")
    ax2.imshow(noise_data, interpolation='nearest', cmap='Greys')
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.set_title("Denoised data (binary) (MAE=%g)" % (mae))
    ax3.imshow(y, interpolation='nearest', cmap='Greys')
    
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.set_title("Change in MAE until convergence (W^L=%g, W^P=%g)" % (w_l, w_p))
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Mean absolute error")
    ax4.plot(np.arange(maes.length), maes)
    
    plt.show()
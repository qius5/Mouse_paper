
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


#### Supplementary code for the paper: 
#### "Linking connectivity, dynamics and computations in recurrent neural networks"
#### F. Mastrogiuseppe and S. Ostojic (2018)

#### CODE 1a: spontaneous activity in rank-one networks: DMF theory (related to Fig. 1C)
#### This code computes the DMF solutions (and their stability) for increasing values of the random strength g
#### The overlap direction is defined along the unitary direction (rho = 0, see Methods)
#### Within the DMF theory, activity is then described in terms of mean (mu) and variance (delta) of x

#### Note that the Data/ folder is empty to begin; this code needs to be run with the flag doCompute = 1
#### at least once


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Import functions
from numba import jit
import matplotlib.pyplot as plt
import numpy as np

import fct_integrals as integ
import fct_facilities as fac
import fct_new_mf as mf


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Parameters

Sigma = 1.6 
SigmaW = 0.8 

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Compute
# We solve separately the DMF equations corresponding to stationary and chaotic states
# The DMF equations admit at maximum three different solutions, which can be reached starting from different i.c.
# We compute two variance values: the population variance delta0 and the individual one delta0I (see Methods)
# Within chaotic phases, the stationary fraction of the variance deltainf is computed as well (see Methods)

doCompute = 1

path_here = 'Data/'

ParVec = [Sigma, SigmaW]

if doCompute == True:

    # define points (k1, k2) on phase plane
    k1 = np.arange(-2.4,2.4,0.1)
    k2 = np.arange(-2.4,2.4,0.1)
    networksize=48
    X1, Y1 = np.meshgrid(k1, k2)
    print("look meshgrid X1")
    print(X1)
    print("look meshgrid Y1")
    print(Y1)
#    vecMatrix = np.zeros((networksize,networksize))    
#    dk1Matrix = np.zeros((networksize,networksize))
#    dk2Matrix = np.zeros((networksize,networksize))
#    dirMatrix = np.zeros((networksize,networksize))
    DX1, DY1 = mf.SolveVectorField([X1,Y1], ParVec)
    M = (np.hypot(DX1, DY1))
    DX1, DY1 = DX1/M, DY1/M
    print("look DX1")
    print(DX1)
    print("look DY1")
    print(DY1)
#    for i in range(networksize):
#       for j in range(networksize):
#        print(i, " ", j)
#        k1new = k1[i]
#        k2new = k1[j]
#        kvec = [k1new, k2new]
#        dk1dt, dk2dt = mf.SolveVectorField ( kvec, ParVec)
#        vecMatrix[i,j] = np.sqrt(dk1dt**2+dk2dt**2) 
##        dk1Matrix[i,j] = dk1dt
#        dk2Matrix[i,j] = dk2dt
#        dirMatrix[i,j] = np.hypot(dk1dt,dk2dt)
    # Store

    fac.Store( X1, 'X1.p', path_here)
    fac.Store( Y1, 'Y1.p', path_here)
    fac.Store( DX1, 'DX1.p', path_here)
    fac.Store( DY1, 'DY1.p', path_here)
    fac.Store( M, 'M.p', path_here)
#    fac.Store( vecMatrix, 'vecMatrix.p', path_here)
#    fac.Store( dk1Matrix, 'dk1Matrix.p', path_here)
#    fac.Store( dk2Matrix, 'dk2Matrix.p', path_here)
#    fac.Store( dirMatrix, 'dirMatrix.p', path_here)

else:

    # Retrieve

    X1 = fac.Retrieve('X1.p', path_here)
    Y1 = fac.Retrieve('Y1.p', path_here)
    DX1 = fac.Retrieve('DX1.p', path_here)
    DY1 = fac.Retrieve('DY1.p', path_here)
    M = fac.Retrieve('M.p', path_here)
    DX1, DX2 = DX1/M, DX2/M
#    vecMatrix = fac.Retrieve('vecMatrix.p', path_here)
#    dk1Matrix = fac.Retrieve('dk1Matrix.p', path_here)
#    dk2Matrix = fac.Retrieve('dk2Matrix.p', path_here)
#    dirMatrix = fac.Retrieve('dirMatrix.p', path_here)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Plot
fac.SetPlotParams()
dashes = [13, 13]

color_scs = '0.6'
color_s = '#4872A1'
color_c1 = '#C44343'
color_c2 = '#FF9999'
color_sim0 = '0'
color_sim1 = '0.5'

thr = 1e-10 # Plot chaotic solutions only when they are non-zero


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Plot K

fg = plt.figure()
ax0 = plt.axes(frameon = True)

# Stationary

#Q = plt.quiver(k1, k2, dk1Matrix, dk2Matrix, dirMatrix, pivot='mid', cmap=plt.cm.plasma)
#Q = plt.quiver(k1, k2, np.array(dk1Matrix), np.array(dk2Matrix), dirMatrix, cmap=plt.cm.plasma)
#Q = plt.quiver(X1, Y1, DX1, DY1, M, pivot='mid', cmap=plt.cm.plasma)
plt.streamplot(X1, Y1, DX1, DY1, density = [0.4, 0.8], color = M, cmap ='autumn') 
#Q = plt.quiver(k1*0.1, k2*0.1, dk1Matrix, dk2Matrix)
plt.colorbar()
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=4)

#plt.xlim(-2.5,2.5)

plt.legend(loc=1)

plt.xlabel('$\kappa_1$')
plt.ylabel('$\kappa_2$')
plt.savefig('k_2D_new.pdf')

plt.show()



#

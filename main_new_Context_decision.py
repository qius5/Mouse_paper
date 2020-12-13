
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
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import random

import fct_integrals as integ
import fct_facilities as fac
import fct_new_mf as mf


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Parameters
contextID = 2
SigmaPFC1PFC1 = 1.3
SigmaPFC1PFC2 = 1.2 
SigmaPFC1PFC3 = 2.1 
SigmaPFC1PFC4 = 1.8 
SigmaPFC2PFC1 = 1.6 
SigmaPFC2PFC2 = 0.4
SigmaPFC2PFC3 = 2.1 
SigmaPFC2PFC4 = 1.8 
SigmaPFC3PFC1 = 0.6 
SigmaPFC3PFC2 = 1.2 
SigmaPFC3PFC3 = 1.3 
SigmaPFC3PFC4 = 1.8 
SigmaPFC4PFC1 = 1.6 
SigmaPFC4PFC2 = 1.2 
SigmaPFC4PFC3 = 2.1 
SigmaPFC4PFC4 = 0.8
SigmaPFC1MD1 = 15
SigmaPFC2MD1 = 15
SigmaPFC3MD1 = -10.2
SigmaPFC4MD1 = -10.2
SigmaPFC1MD2 = -10.2
SigmaPFC2MD2 = -10.2
SigmaPFC3MD2 = 15
SigmaPFC4MD2 = 15
SigmaPFC1IA = 3 
SigmaPFC2IA = 0
SigmaPFC3IA = 0
SigmaPFC4IA = 0
SigmaPFC1IB = 0
SigmaPFC2IB = 3 
SigmaPFC3IB = 0
SigmaPFC4IB = 0
SigmaPFC1IC =  0
SigmaPFC2IC = 0
SigmaPFC3IC =3 
SigmaPFC4IC = 0
SigmaPFC1ID = 0
SigmaPFC2ID = 0
SigmaPFC3ID = 0
SigmaPFC4ID = 3
if contextID==1: 
 SigmaMD1PFC1 = 10 
 SigmaMD1PFC2 = 10 
 SigmaMD1PFC3 = -2.5
 SigmaMD1PFC4 = -2.5
 SigmaMD2PFC1 = 0 
 SigmaMD2PFC2 = 0
 SigmaMD2PFC3 = 0.0
 SigmaMD2PFC4 = 0.0
if contextID==2:
 SigmaMD1PFC1 = 0.0
 SigmaMD1PFC2 = 0.0
 SigmaMD1PFC3 = 0.0
 SigmaMD1PFC4 = 0.0
 SigmaMD2PFC1 = -2.5 
 SigmaMD2PFC2 = -2.5 
 SigmaMD2PFC3 = 10
 SigmaMD2PFC4 = 10
SigmaMD1MD1 = 0
SigmaMD1MD2 = 0 
SigmaMD2MD1 = 0
SigmaMD2MD2 = 0 



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Compute
# We solve separately the DMF equations corresponding to stationary and chaotic states
# The DMF equations admit at maximum three different solutions, which can be reached starting from different i.c.
# We compute two variance values: the population variance delta0 and the individual one delta0I (see Methods)
# Within chaotic phases, the stationary fraction of the variance deltainf is computed as well (see Methods)

doCompute = 1

path_here = 'Data/'

ParVec = [SigmaPFC1PFC1, SigmaPFC1PFC2, SigmaPFC1PFC3, SigmaPFC1PFC4, SigmaPFC2PFC1, SigmaPFC2PFC2, SigmaPFC2PFC3, SigmaPFC2PFC4, SigmaPFC3PFC1, SigmaPFC3PFC2, SigmaPFC3PFC3, SigmaPFC3PFC4, SigmaPFC4PFC1, SigmaPFC4PFC2, SigmaPFC4PFC3, SigmaPFC4PFC4, SigmaPFC1MD1, SigmaPFC2MD1, SigmaPFC3MD1, SigmaPFC4MD1, SigmaPFC1MD2, SigmaPFC2MD2, SigmaPFC3MD2, SigmaPFC4MD2, SigmaPFC1IA, SigmaPFC2IA, SigmaPFC3IA, SigmaPFC4IA, SigmaPFC1IB, SigmaPFC2IB, SigmaPFC3IB, SigmaPFC4IB, SigmaPFC1IC, SigmaPFC2IC, SigmaPFC3IC, SigmaPFC4IC, SigmaPFC1ID, SigmaPFC2ID, SigmaPFC3ID, SigmaPFC4ID, SigmaMD1PFC1, SigmaMD1PFC2, SigmaMD1PFC3, SigmaMD1PFC4, SigmaMD1MD1, SigmaMD1MD2, SigmaMD2PFC1, SigmaMD2PFC2, SigmaMD2PFC3, SigmaMD2PFC4, SigmaMD2MD1, SigmaMD2MD2  ]



if doCompute == True:

    # define points (k1, k2) on phase plane
    #kPFC = np.arange(-2.4,2.4,0.1)
    #kPFC = random.uniform(0,1)*2.4
    #kMD1 = random.uniform(0,1)*2.4
    #kMD2 = random.uniform(0,1)*2.4
    kPFC= -0.0
    kMD1= -0.0
    kMD2= -0.0
    networksize=48
    kvec = [ kPFC, kMD1, kMD2 ]
    tra_kPFC, tra_kMD1, tra_kMD2 = mf.SolveRank3Context1(kvec, ParVec, 1)
    #print(tra_kPFC)
    #print(tra_kMD1)
    #print(tra_kMD2)
    fac.Store( tra_kPFC, 'tra_kPFC.p', path_here)
    fac.Store( tra_kMD1, 'tra_kMD1.p', path_here)
    fac.Store( tra_kMD2, 'tra_kMD2.p', path_here)

else:

    # Retrieve

    tra_kPFC = fac.Retrieve('kPFC.p', path_here)
    tra_kMD1 = fac.Retrieve('kMD1.p', path_here)
    tra_kMD2 = fac.Retrieve('kMD2.p', path_here)


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
#ax0 = plt.axes(projection="3d")

# Stationary

#ax0.plot3D(tra_kPFC[0,:], tra_kMD1[0,:], tra_kMD2[0,:]) 
ax0.plot(tra_kPFC[0,:], color = 'black') 
ax0.plot(tra_kMD1[0,:], color = 'forestgreen') 
ax0.plot(tra_kMD2[0,:], color = 'lightgreen') 
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
#plt.locator_params(nbins=4)


plt.legend(loc=1)
ax0.set_ylim(-0.2,0.5)
ax0.set_xlabel('time')
ax0.set_ylabel('$\kappa$')
#ax0.set_zlabel('$\kappa_{MD2}$')
plt.savefig('k_3D_trajectories_new.pdf')

plt.show()



#

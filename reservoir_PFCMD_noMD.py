# -*- coding: utf-8 -*-
# (c) September 2020 Siwei Qiu, Brandeis.

"""Some reservoir tweaks are inspired by Nicola and Clopath, arxiv, 2016 and Miconi 2016."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# plt.ion()
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.io import savemat
from scipy import stats
import sys,shelve, tqdm, time
import plot_utils as pltu
from data_generator import data_generator
from plot_figures import *
import torch
cuda=1
class PFCMD():
    def __init__(self,PFC_G,PFC_G_off,learning_rate,
                    noiseSD,tauError,plotFigs=True,saveData=False):
        self.debug = False
        # random seed 17 can do the task, 47 has a bigger error at second switch, 167 has big peak
        #self.RNGSEED = 17
        #self.RNGSEED = 167
        self.RNGSEED = 51
        np.random.seed([self.RNGSEED])

        self.Nsub = 200                     # number of neurons per cue
        self.Ntasks = 2                     # Ambiguous variable name, replacing with appropriate ones below:  # number of contexts 
        self.Ncontexts = 2                  # number of contexts (match block or non-match block)
        self.Nblocks = 2                    # number of blocks
        self.Nmd    = 2                     # number of MD cells.
        self.xorTask = False                # use xor Task or simple 1:1 map task
        # self.xorTask = True               # use xor Task or simple 1:1 map task
        self.tactileTask = False             # Use the human tactile probabalistic task
        self.Ncues = 4 #self.Ncontexts *2   # number of input cues. Two for up cue, and two for down cue.
        self.Nneur = self.Nsub*(self.Ncues+1)# number of neurons
        if self.xorTask: self.inpsPerContext = 4# number of cue combinations per task
        else: self.inpsPerContext = 2
        self.Nout = 2                       # number of outputs
        self.tau = 0.02
        self.dt = 0.001
        self.tsteps = 200                   # number of timesteps in a trial
        self.cuesteps = 100                 # number of time steps for which cue is on
        self.noiseSD = noiseSD
        self.saveData = saveData

        self.learning_rate = learning_rate  # too high a learning rate makes the output weights
                                            #  change too much within a trial / training cycle,
                                            #  then the output interference depends
                                            #  on the order of cues within a cycle
                                            # typical values is 1e-5, can vary from 1e-4 to 1e-6
        self.tauError = tauError            # smooth the error a bit, so that weights don't fluctuate
        self.modular  = True                # Assumes PFC modules and pass input to only one module per tempral context.
        self.MDeffect = False                # whether to have MD present or not
        self.MDEffectType = 'submult'       # MD subtracts from across tasks and multiplies within task
        #self.MDEffectType = 'subadd'        # MD subtracts from across tasks and adds within task
        #self.MDEffectType = 'divadd'        # MD divides from across tasks and adds within task
        #self.MDEffectType = 'divmult'       # MD divides from across tasks and multiplies within task

        self.dirConn = False                # direct connections from cue to output, also learned
        self.outExternal = True             # True: output neurons are external to the PFC
                                            #  (i.e. weights to and fro (outFB) are not MD modulated)
                                            # False: last self.Nout neurons of PFC are output neurons
        self.outFB = False                  # if outExternal, then whether feedback from output to reservoir
        self.noisePresent = False           # add noise to all reservoir units

        self.positiveRates = True           # whether to clip rates to be only positive, G must also change
        
        self.MDlearn = False                # whether MD should learn
                                            #  possibly to make task representations disjoint (not just orthogonal)

        self.MDstrength = None              # if None, use wPFC2MD, if not None as below, just use context directly
        # self.MDstrength = 0.                # a parameter that controls how much the MD disjoints task representations.
        # self.MDstrength = 1.                # a parameter that controls how much the MD disjoints task representations.
                                            #  zero would be a pure reservoir, 1 would be full MDeffect
                                            # -1 for zero recurrent weights
        self.wInSpread = False              # Spread wIn also into other cue neurons to see if MD disjoints representations
        self.blockTrain = True              # first half of training is context1, second half is context2
      #  self.blockTrain = False # use different levels of association for multiple blocks training
        
        self.reinforce = False              # use reinforcement learning (node perturbation) a la Miconi 2017
        self.MDreinforce = False            #  instead of error-driven learning
        self.reinforceReservoir = False # learning on reservoir weights also?
        ## init weights: 
        self.wPFC2MD = np.zeros(shape=(self.Nmd,self.Nneur))
        self.wMD2MD = np.zeros(shape=(self.Nmd,self.Nmd))
                                            
        if self.reinforce:
            self.perturbProb = 50./self.tsteps
                                            # probability of perturbation of each output neuron per time step
            self.perturbAmpl = 10.          # how much to perturb the output by
            self.meanErrors = np.zeros(self.Ncontexts*self.inpsPerContext)
                                            # vector holding running mean error for each cue
            self.decayErrorPerTrial = 0.1   # how to decay the mean errorEnd by, per trial
            self.learning_rate *= 10        # increase learning rate for reinforce
            if self.reinforceReservoir:
                self.perturbProb /= 10

        self.depress = False                # a depressive term if there is pre-post firing
        self.multiAttractorReservoir = False# increase the reservoir weights within each cue
                                            #  all uniformly (could also try Hopfield style for the cue pattern)
        if self.outExternal:
            self.wOutMask = np.ones(shape=(self.Nout,self.Nneur))
            #self.wOutMask[ np.random.uniform( \
            #            size=(self.Nout,self.Nneur)) > 0.3 ] = 0.
            #                                # output weights sparsity, 30% sparsity

        for i in range(self.Nmd):
           for j in range(self.Nmd):
             if i==j:
              self.wMD2MD[i,j] = 0
             if i!=j:
              self.wMD2MD[i,j] = -5
        
        # for taski in np.arange(self.Nmd):
        #     self.wPFC2MD[taski,self.Nsub*taski*2:self.Nsub*(taski+1)*2] = 1./self.Nsub


        if self.MDEffectType == 'submult':
            # working!
            Gbase = 0.75                      # determines also the cross-task recurrence
            if self.MDstrength is None: MDval = 1.
            elif self.MDstrength < 0.: MDval = 0.
            else: MDval = self.MDstrength
            # subtract across tasks (task with higher MD suppresses cross-tasks)
            self.wMD2PFC = np.ones(shape=(self.Nneur,self.Nmd)) * (-10.) * MDval
            for taski in np.arange(self.Ncontexts):
                self.wMD2PFC[self.Nsub*2*taski:self.Nsub*2*(taski+1),taski] = 0.
            self.useMult = False
            # multiply recurrence within task, no addition across tasks
            ## choose below option for cross-recurrence
            ##  if you want "MD inactivated" (low recurrence) state
            ##  as the state before MD learning
            #self.wMD2PFCMult = np.zeros(shape=(self.Nneur,self.Nmd))
            # choose below option for cross-recurrence
            #  if you want "reservoir" (high recurrence) state
            #  as the state before MD learning (makes learning more difficult)
            self.wMD2PFCMult = np.ones(shape=(self.Nneur,self.Nmd)) \
                                * PFC_G_off/Gbase * (1-MDval)
            for taski in np.arange(self.Ncontexts):
                self.wMD2PFCMult[self.Nsub*2*taski:self.Nsub*2*(taski+1),taski]\
                            += PFC_G/Gbase * MDval
            # threshold for sharp sigmoid (0.1 width) transition of MDinp
            self.MDthreshold = 0.5 # Siwei modified from 0.4 to 5

        else:
            print('undefined inhibitory effect of MD')
            sys.exit(1)
        # With MDeffect = True and MDstrength = 0, i.e. MD inactivated
        #  PFC recurrence is (1+PFC_G_off)*Gbase = (1+1.5)*0.75 = 1.875
        # So with MDeffect = False, ensure the same PFC recurrence for the pure reservoir
        if not self.MDeffect: Gbase = 1.875

        # Choose G based on the type of activation function
        # unclipped activation requires lower G than clipped activation,
        #  which in turn requires lower G than shifted tanh activation.
        if self.positiveRates:
            self.G = Gbase
            self.tauMD = self.tau*3
        else:
            self.G = Gbase
            #self.MDthreshold = 0.4
            self.MDthreshold = 0.5
            self.tauMD = self.tau*10
            
        if self.MDeffect and self.MDlearn: # if MD is learnable, reset all weights to 0.
            # self.wMD2PFC *= 0.
            # self.wMD2PFCMult *= 0.
            #self.wPFC2MD = np.random.normal(size=(self.Nmd, self.Nneur))\
            #                *self.G/np.sqrt(self.Nsub*2)
            self.wPFC2MD = np.random.normal(size=(self.Nmd, self.Nneur))\
                            *self.G/np.sqrt(self.Nsub*2)
            #self.wPFC2MD = np.outer(self.MDrates, self.PFCrates)
            self.wPFC2MD -= np.mean(self.wPFC2MD,axis=1)[:,np.newaxis] # same as res rec, substract mean from each row.
            self.wMD2PFC = np.random.normal(size=(self.Nneur, self.Nmd))\
                            *self.G/np.sqrt(self.Nsub*2)
            self.wMD2PFC -= np.mean(self.wMD2PFC,axis=1)[:,np.newaxis] # same as res rec, substract mean from each row.
            #self.initial_norm_wPFC2MD = np.linalg.norm(self.wPFC2MD)
            #self.initial_norm_wMD2PFC = np.linalg.norm(self.wMD2PFC)

        self.MDpreTrace = np.zeros(shape=(self.Nneur))

        # Perhaps I shouldn't have self connections / autapses?!
        # Perhaps I should have sparse connectivity?
        self.Jrec = (np.random.normal(size=(self.Nneur, self.Nneur))+0*np.eye(self.Nneur))\
                        *self.G/np.sqrt(self.Nsub*2)
        if cuda:
            self.Jrec = torch.Tensor(self.Jrec).cuda()

        # if self.MDstrength < 0.: self.Jrec *= 0. # Ali commented this out. I'm setting MDstrength to None. Not sure if this is really asking if strength is ever negative
        if self.multiAttractorReservoir:
            for i in range(self.Ncues):
                self.Jrec[self.Nsub*i:self.Nsub*(i+1)] *= 2.
         
        # make mean input to each row zero,
        #  helps to avoid saturation (both sides) for positive-only rates.
        #  see Nicola & Clopath 2016
        # mean of rows i.e. across columns (axis 1),
        #  then expand with np.newaxis
        #   so that numpy's broadcast works on rows not columns
        if cuda:
          self.Jrec -= torch.mean(self.Jrec,axis=1)[:,np.newaxis]
        else:
          self.Jrec -= np.mean(self.Jrec,axis=1)[:,np.newaxis]
        #for i in range(self.Nsub):
        #    self.Jrec[i,:self.Nsub] -= np.mean(self.Jrec[i,:self.Nsub])
        #    self.Jrec[self.Nsub+i,self.Nsub:self.Nsub*2] -=\
        #        np.mean(self.Jrec[self.Nsub+i,self.Nsub:self.Nsub*2])
        #    self.Jrec[self.Nsub*2+i,self.Nsub*2:self.Nsub*3] -=\
        #        np.mean(self.Jrec[self.Nsub*2+i,self.Nsub*2:self.Nsub*3])
        #    self.Jrec[self.Nsub*3+i,self.Nsub*3:self.Nsub*4] -=\
        #        np.mean(self.Jrec[self.Nsub*3+i,self.Nsub*3:self.Nsub*4])

        # I don't want to have an if inside activation
        #  as it is called at each time step of the simulation
        # But just defining within __init__
        #  doesn't make it a member method of the class,
        #  hence the special self.__class__. assignment
        if self.positiveRates:
            # only +ve rates
            def activation(self,inp):
                return np.clip(np.tanh(inp),0,None)
                #return np.sqrt(np.clip(inp,0,None))
                #return (np.tanh(inp)+1.)/2.
        else:
            # both +ve/-ve rates as in Miconi
            def activation(self,inp):
                return np.tanh(inp)
        self.__class__.activation = activation

        #wIn = np.random.uniform(-1,1,size=(self.Nneur,self.Ncues))
        self.wIn = np.zeros((self.Nneur,self.Ncues))
        self.cueFactor = 0.75#1.5 Ali halved it when I added cues going to both PFC regions, i.e two copies of input. But now working ok even with only one copy of input.
        if self.positiveRates: lowcue,highcue = 0.5,1.
        else: lowcue,highcue = -1.,1
        for cuei in np.arange(self.Ncues):
            self.wIn[self.Nsub*cuei:self.Nsub*(cuei+1),cuei] = \
                    np.random.uniform(lowcue,highcue,size=self.Nsub) \
                            *self.cueFactor
            if self.wInSpread:
                # small cross excitation to half the neurons of cue-1 (wrap-around)
                if cuei == 0: endidx = self.Nneur
                else: endidx = self.Nsub*cuei
                self.wIn[self.Nsub*cuei - self.Nsub//2 : endidx,cuei] += \
                        np.random.uniform(0.,lowcue,size=self.Nsub//2) \
                                *self.cueFactor
                # small cross excitation to half the neurons of cue+1 (wrap-around)
                self.wIn[(self.Nsub*(cuei+1))%self.Nneur : \
                            (self.Nsub*(cuei+1) + self.Nsub//2 )%self.Nneur,cuei] += \
                        np.random.uniform(0.,lowcue,size=self.Nsub//2) \
                                *self.cueFactor

        # wDir and wOut are set in the main training loop
        if self.outExternal and self.outFB:
            self.wFB = np.random.uniform(-1,1,size=(self.Nneur,self.Nout))\
                            *self.G/np.sqrt(self.Nsub*2)*PFC_G

        self.cue_eigvecs = np.zeros((self.Ncues,self.Nneur))
        self.plotFigs = plotFigs
        self.cuePlot = (0,0)
                
        if self.saveData:
            self.fileDict = shelve.open('dataPFCMD/data_reservoir_PFC_MD'+\
                                    str(self.MDstrength)+\
                                    '_R'+str(self.RNGSEED)+\
                                    ('_xor' if self.xorTask else '')+'.shelve')
        
        self.meanAct = np.zeros(shape=(self.Ncontexts*self.inpsPerContext,\
                                    self.tsteps,self.Nneur))
        
        self.data_generator = data_generator(local_Ntrain = 10000)

    def sim_cue(self,taski,cuei,cue,target,MDeffect=True,
                    MDCueOff=False,MDDelayOff=False,
                    train=True,routsTarget=None):
        '''
        self.reinforce trains output weights
         using REINFORCE / node perturbation a la Miconi 2017.'''
        cues = np.zeros(shape=(self.tsteps,self.Ncues))
        # random initialization of input to units
        # very important to have some random input
        #  just for the xor task for (0,0) cue!
        #  keeping it also for the 1:1 task just for consistency
        xinp = np.random.uniform(0,0.1,size=(self.Nneur))
        #xinp = np.zeros(shape=(self.Nneur))
        xadd = np.zeros(shape=(self.Nneur))
        MDinp = np.zeros(shape=self.Nmd)
        MDinps = np.zeros(shape=(self.tsteps, self.Nmd))
        routs = np.zeros(shape=(self.tsteps,self.Nneur))
        MDouts = np.zeros(shape=(self.tsteps,self.Nmd))
        MDout =np.random.uniform(low=0.00, high = 1.00, size=(self.Nmd))
        outInp = np.zeros(shape=self.Nout)
        outs = np.zeros(shape=(self.tsteps,self.Nout))
        out = np.zeros(self.Nout)
        errors = np.zeros(shape=(self.tsteps,self.Nout))
        error_smooth = np.zeros(shape=self.Nout)
        if self.reinforce:
            HebbTrace = np.zeros(shape=(self.Nout,self.Nneur))
            if self.dirConn:
                HebbTraceDir = np.zeros(shape=(self.Nout,self.Ncues))
            if self.reinforceReservoir:
                if cuda:
                   HebbTraceRec = torch.Tensor(np.zeros(shape=(self.Nneur,self.Nneur))).cuda()
                else:
                   HebbTraceRec = np.zeros(shape=(self.Nneur,self.Nneur))
            if self.MDreinforce:
                HebbTraceMD = np.zeros(shape=(self.Nmd,self.Nneur))

        for i in range(self.tsteps):
            rout = self.activation(xinp)
            routs[i,:] = rout
            if self.outExternal:
                outAdd = np.dot(self.wOut,rout)

            if MDeffect:
                # MD decays 10x slower than PFC neurons,
                #  so as to somewhat integrate PFC input
                if self.positiveRates:
                    MDinp +=  self.dt/self.tauMD * \
                            ( -MDinp + np.dot(self.wPFC2MD,rout) + np.dot(self.wMD2MD, MDout))
                else: # shift PFC rates, so that mean is non-zero to turn MD on
                    MDinp +=  self.dt/self.tauMD * \
                            ( -MDinp + np.dot(self.wPFC2MD,(rout+1./2)) + np.dot(self.wMD2MD, (MDout+1./2)) )

                # MD off during cue or delay periods:
                if MDCueOff and i<self.cuesteps:
                    MDinp = np.zeros(self.Nmd)
                    #MDout /= 2.
                if MDDelayOff and i>self.cuesteps and i<self.tsteps:
                    MDinp = np.zeros(self.Nmd)

                # MD out either from MDinp or forced
                if self.MDstrength is not None:
                    MDout[taski] = 1.
                else:
                    MDout = np.clip((np.tanh( (MDinp-self.MDthreshold)/0.5 ) + 1.0) / 2. , 0, 1.5)#Siwei changed denominator from 0.1 to 10
                # if MDlearn then force "winner take all" on MD output
                if train and self.MDlearn:
                    MDout = np.clip((np.tanh((MDinp-self.MDthreshold)/0.5) + 1.0) / 2., 0, 1.5)
                    # winner take all on the MD
                    #  hardcoded for self.Nmd = 2
                    # Siwei get rid of the hard coding
                    #if MDinp[0] > MDinp[1]: MDout = np.array([1,0])
                    #else: MDout = np.array([0,1])

                MDouts[i,:] = MDout
                MDinps[i, :]= MDinp

                if self.useMult:
                    self.MD2PFCMult = np.dot(self.wMD2PFCMult,MDout)
                    if cuda:
                      xadd = (1.+self.MD2PFCMult) * torch.matmul(self.Jrec,torch.Tensor(rout).cuda()).detach().cpu().numpy()
                    else:
                      xadd = (1.+self.MD2PFCMult) * np.dot(self.Jrec,rout)
                else:
                    if cuda:
                      xadd = torch.matmul(self.Jrec,torch.Tensor(rout).cuda()).detach().cpu().numpy()
                    else:
                      xadd = np.dot(self.Jrec,rout)
                xadd += np.dot(self.wMD2PFC,MDout)

                if train and self.MDlearn:# and not self.MDreinforce:
                    # MD presynaptic traces filtered over 10 trials
                    # Ideally one should weight them with MD syn weights,
                    #  but syn plasticity just uses pre*post, but not actualy synaptic flow.
                    self.MDpreTrace += 1./self.tsteps/10. * \
                                        ( -self.MDpreTrace + rout )
                    # wPFC2MDdelta = 1e-4*np.outer(MDout-0.5,self.MDpreTrace-0.11) # Ali changed from 1e-4 and thresh from 0.13
                    wPFC2MDdelta = 1e-4*np.outer(MDout,self.MDpreTrace-0.11)-1e-4*0.0*self.wPFC2MD-1e-4*0.0*np.mean(self.wPFC2MD,axis=1)[:,np.newaxis]  # Ali changed from 1e-4 and thresh from 0.13
                    #wPFC2MDdelta = 1e-4*np.outer(MDout,self.MDpreTrace-0.11) # Ali changed from 1e-4 and thresh from 0.13
                    # wPFC2MDdelta *= self.wPFC2MD # modulate it by the weights to get supralinear effects. But it'll actually be sublinear because all values below 1
                    MDrange = 0.06
                    MDweightdecay = 1.#0.996
                    self.wPFC2MD = np.clip(self.wPFC2MD +wPFC2MDdelta ,  -MDrange ,MDrange ) # Ali lowered to 0.01 from 1. 
                    self.wMD2PFC = np.clip(self.wMD2PFC +wPFC2MDdelta.T,-MDrange ,MDrange ) # lowered from 10.
                    #self.wPFC2MD = np.clip(0.95*self.wPFC2MD +wPFC2MDdelta - 0.1*np.mean(self.wPFC2MD, axis=1)[:,np.newaxis],  -MDrange ,MDrange ) # Ali lowered to 0.01 from 1. 
                    #self.wMD2PFC = np.clip(0.95*self.wMD2PFC +wPFC2MDdelta.T-0.1*np.mean(self.wMD2PFC, axis=1)[:,np.newaxis],-MDrange ,MDrange ) # lowered from 10.
                    # self.wMD2PFCMult = np.clip(self.wMD2PFCMult+wPFC2MDdelta.T,0.,7./self.G) # ali removed all mult weights
            else:
                if cuda:
                   xadd = torch.matmul(self.Jrec,torch.Tensor(rout).cuda()).detach().cpu().numpy()
                else:
                   xadd = np.dot(self.Jrec,rout)

            if i < self.cuesteps:
                ## add an MDeffect on the cue
                #if MDeffect and useMult:
                #    xadd += self.MD2PFCMult * np.dot(self.wIn,cue)
                # baseline cue is always added
                xadd += np.dot(self.wIn,cue)
                cues[i,:] = cue
                if self.dirConn:
                    if self.outExternal:
                        outAdd += np.dot(self.wDir,cue)
                    else:
                        xadd[-self.Nout:] += np.dot(self.wDir,cue)

            if self.reinforce:
                # Exploratory perturbations a la Miconi 2017
                # Perturb each output neuron independently
                #  with probability perturbProb
                perturbationOff = np.where(
                        np.random.uniform(size=self.Nout)>=self.perturbProb )
                perturbation = np.random.uniform(-1,1,size=self.Nout)
                perturbation[perturbationOff] = 0.
                perturbation *= self.perturbAmpl
                outAdd += perturbation
            
                if self.reinforceReservoir:
                    perturbationOff = np.where(
                            np.random.uniform(size=self.Nneur)>=self.perturbProb )
                    perturbationRec = np.random.uniform(-1,1,size=self.Nneur)
                    perturbationRec[perturbationOff] = 0.
                    # shouldn't have MD mask on perturbations,
                    #  else when MD is off, perturbations stop!
                    #  use strong subtractive inhibition to kill perturbation
                    #   on task irrelevant neurons when MD is on.
                    #perturbationRec *= self.MD2PFCMult  # perturb gated by MD
                    perturbationRec *= self.perturbAmpl
                    xadd += perturbationRec
                
                if self.MDreinforce:
                    perturbationOff = np.where(
                            np.random.uniform(size=self.Nmd)>=self.perturbProb )
                    perturbationMD = np.random.uniform(-1,1,size=self.Nmd)
                    perturbationMD[perturbationOff] = 0.
                    perturbationMD *= self.perturbAmpl
                    MDinp += perturbationMD

            if self.outExternal and self.outFB:
                xadd += np.dot(self.wFB,out)
            xinp += self.dt/self.tau * (-xinp + xadd)
            
            if self.noisePresent:
                xinp += np.random.normal(size=(self.Nneur))*self.noiseSD \
                            * np.sqrt(self.dt)/self.tau
            
            if self.outExternal:
                outInp += self.dt/self.tau * (-outInp + outAdd)
                out = self.activation(outInp)                
            else:
                out = rout[-self.Nout:]
            error = out - target
            errors[i,:] = error
            outs[i,:] = out
            error_smooth += self.dt/self.tauError * (-error_smooth + error)
            
            if train:
                if self.reinforce:
                    # note: rout is the activity vector for previous time step
                    HebbTrace += np.outer(perturbation,rout)
                    if self.dirConn:
                        HebbTraceDir += np.outer(perturbation,cue)
                    if self.reinforceReservoir:
                        if cuda:
                             HebbTraceRec += torch.ger(torch.Tensor(perturbationRec).cuda(),torch.Tensor(rout).cuda())
                        else:
                             HebbTraceRec += np.outer(perturbationRec,rout)
                    if self.MDreinforce:
                        HebbTraceMD += np.outer(perturbationMD,rout)
                else:
                    # error-driven i.e. error*pre (perceptron like) learning
                    if self.outExternal:
                        self.wOut += -self.learning_rate \
                                        * np.outer(error_smooth,rout)
                        if self.depress:
                            self.wOut -= 10*self.learning_rate \
                                        * np.outer(out,rout)*self.wOut
                    else:
                        self.Jrec[-self.Nout:,:] += -self.learning_rate \
                                        * np.outer(error_smooth,rout)
                        if self.depress:
                            self.Jrec[-self.Nout:,:] -= 10*self.learning_rate \
                                        * np.outer(out,rout)*self.Jrec[-self.Nout:,:]
                    if self.dirConn:
                        self.wDir += -self.learning_rate \
                                        * np.outer(error_smooth,cue)
                        if self.depress:
                            self.wDir -= 10*self.learning_rate \
                                        * np.outer(out,cue)*self.wDir

        inpi = taski*self.inpsPerContext + cuei
        if train:
            if self.MDlearn: # after all Hebbian learning within trial and reinforce after trial, re-center MD2PFC and PFC2MD weights This will introduce 
                #synaptic competition both ways.
                #self.wMD2PFC = MDweightdecay* (self.wMD2PFC)
                #self.wPFC2MD = MDweightdecay* (self.wPFC2MD)
                #self.wPFC2MD /= np.linalg.norm(self.wPFC2MD)/ self.initial_norm_wPFC2MD
                #self.wMD2PFC /= np.linalg.norm(self.wMD2PFC)/ self.initial_norm_wMD2PFC
                
                self.wMD2PFC -= np.mean(self.wMD2PFC)
                self.wMD2PFC *= self.G/np.sqrt(self.Nsub*2)/np.std(self.wMD2PFC) # div weights by their std to get normalized dist, then mul it by desired std
                self.wPFC2MD -= np.mean(self.wPFC2MD)
                self.wPFC2MD *= self.G/np.sqrt(self.Nsub*2)/np.std(self.wPFC2MD) # div weights by their std to get normalized dist, then mul it by desired std
        if train and self.reinforce:
            # with learning using REINFORCE / node perturbation (Miconi 2017),
            #  the weights are only changed once, at the end of the trial
            # apart from eta * (err-baseline_err) * hebbianTrace,
            #  the extra factor baseline_err helps to stabilize learning
            #   as per Miconi 2017's code,
            #  but I found that it destabilized learning, so not using it.
            errorEnd = np.mean(errors*errors)
            if self.outExternal:
                self.wOut -= self.learning_rate * \
                        (errorEnd-self.meanErrors[inpi]) * \
                            HebbTrace #* self.meanErrors[inpi]
            else:
                self.Jrec[-self.Nout:,:] -= self.learning_rate * \
                        (errorEnd-self.meanErrors[inpi]) * \
                            HebbTrace #* self.meanErrors[inpi]
            if self.reinforceReservoir:
                self.Jrec -= self.learning_rate * \
                        (errorEnd-self.meanErrors[inpi]) * \
                            HebbTraceRec #* self.meanErrors[inpi]                
            if self.MDreinforce:
                self.wPFC2MD -= self.learning_rate * \
                        (errorEnd-self.meanErrors[inpi]) * \
                            HebbTraceMD #* self.meanErrors[inpi]                
                self.wMD2PFC -= self.learning_rate * \
                        (errorEnd-self.meanErrors[inpi]) * \
                            HebbTraceMD.T #* self.meanErrors[inpi]                
            if self.dirConn:
                self.wDir -= self.learning_rate * \
                        (errorEnd-self.meanErrors[inpi]) * \
                            HebbTraceDir #* self.meanErrors[inpi]

            # cue-specific mean error (low-pass over many trials)
            self.meanErrors[inpi] = \
                self.decayErrorPerTrial * self.meanErrors[inpi] + \
                (1.0 - self.decayErrorPerTrial) * errorEnd

        if train and self.outExternal:
            self.wOut *= self.wOutMask
        
        self.meanAct[inpi,:,:] += routs

        return cues, routs, outs, MDouts, MDinps, errors

    def get_cues_order(self,cues):
        cues_order = np.random.permutation(cues)
        return cues_order

    def get_cue_target(self,taski,cuei):
        cue = np.zeros(self.Ncues)
        if self.modular:
            inpBase = taski*2 # Ali turned off to stop encoding context at the inpuit layer
        else:
            inpBase = 2
        if cuei in (0,1):
            cue[inpBase+cuei] = 1. # so task is encoded in cue. taski shifts which set of cues to use. That's why number of cues was No_of_tasks *2
        elif cuei == 3:
            cue[inpBase:inpBase+2] = 1
        
        if self.xorTask:
            if cuei in (0,1):
                target = np.array((1.,0.))
            else:
                target = np.array((0.,1.))
        else:
            if cuei == 0: target = np.array((1.,0.))
            else: target = np.array((0.,1.))

        if self.tactileTask:
            cue = np.zeros(self.Ncues) #reset cue 
            cuei = np.random.randint(0,2) #up or down
            non_match = self.get_next_target(taski) #get a match or a non-match response from the data_generator class
            if non_match: #flip
                targeti = 0 if cuei ==1 else 1
            else:
                targeti = cuei 
            
            if self.modular:
                cue[inpBase+cuei] = 1. # Pass cue to the first PFC region 
            else:
                cue[inpBase+cuei] = 1. # Pass cue to the first PFC region 
                cue[cuei] = 1.         # Pass cue to the second PFC region
            
            target = np.array((1.,0.)) if targeti==0  else np.array((0.,1.))
        

        return cue, target

    def plot_column(self,fig,cues,routs,MDouts,outs,ploti=0):
        print('Plotting ...')
        cols=4
        if ploti==0:
            yticks = (0,1)
            ylabels=('Cues','PFC for cueA','PFC for cueB',
                        'PFC for cueC','PFC for cueD','PFC for rest',
                        'MD 1,2','Output 1,2')
        else:
            yticks = ()
            ylabels=('','','','','','','','')
        ax = fig.add_subplot(8,cols,1+ploti)
        ax.plot(cues,linewidth=pltu.plot_linewidth)
        ax.set_ylim([-0.1,1.1])
        pltu.beautify_plot(ax,x0min=False,y0min=False,
                xticks=(),yticks=yticks)
        pltu.axes_labels(ax,'',ylabels[0])
        ax = fig.add_subplot(8,cols,cols+1+ploti)
        ax.plot(routs[:,:10],linewidth=pltu.plot_linewidth)
        ax.set_ylim([-0.1,1.1])
        pltu.beautify_plot(ax,x0min=False,y0min=False,
                xticks=(),yticks=yticks)
        pltu.axes_labels(ax,'',ylabels[1])
        ax = fig.add_subplot(8,cols,cols*2+1+ploti)
        ax.plot(routs[:,self.Nsub:self.Nsub+10],
                    linewidth=pltu.plot_linewidth)
        ax.set_ylim([-0.1,1.1])
        pltu.beautify_plot(ax,x0min=False,y0min=False,
                xticks=(),yticks=yticks)
        pltu.axes_labels(ax,'',ylabels[2])
        if self.Ncues > 2:
            ax = fig.add_subplot(8,cols,cols*3+1+ploti)
            ax.plot(routs[:,self.Nsub*2:self.Nsub*2+10],
                        linewidth=pltu.plot_linewidth)
            ax.set_ylim([-0.1,1.1])
            pltu.beautify_plot(ax,x0min=False,y0min=False,
                    xticks=(),yticks=yticks)
            pltu.axes_labels(ax,'',ylabels[3])
            ax = fig.add_subplot(8,cols,cols*4+1+ploti)
            ax.plot(routs[:,self.Nsub*3:self.Nsub*3+10],
                        linewidth=pltu.plot_linewidth)
            ax.set_ylim([-0.1,1.1])
            pltu.beautify_plot(ax,x0min=False,y0min=False,
                    xticks=(),yticks=yticks)
            pltu.axes_labels(ax,'',ylabels[4])
            ax = fig.add_subplot(8,cols,cols*5+1+ploti)
            ax.plot(routs[:,self.Nsub*4:self.Nsub*4+10],
                        linewidth=pltu.plot_linewidth)
            ax.set_ylim([-0.1,1.1])
            pltu.beautify_plot(ax,x0min=False,y0min=False,
                    xticks=(),yticks=yticks)
            pltu.axes_labels(ax,'',ylabels[5])
        ax = fig.add_subplot(8,cols,cols*6+1+ploti)
        ax.plot(MDouts,linewidth=pltu.plot_linewidth)
        ax.set_ylim([-0.1,1.1])
        pltu.beautify_plot(ax,x0min=False,y0min=False,
                xticks=(),yticks=yticks)
        pltu.axes_labels(ax,'',ylabels[6])
        ax = fig.add_subplot(8,cols,cols*7+1+ploti)
        ax.plot(outs,linewidth=pltu.plot_linewidth)
        ax.set_ylim([-0.1,1.1])
        pltu.beautify_plot(ax,x0min=False,y0min=False,
                xticks=[0,self.tsteps],yticks=yticks)
        pltu.axes_labels(ax,'time (ms)',ylabels[7])
        fig.tight_layout()
        
        if self.saveData:
            d = {}
            # 1st column of all matrices is number of time steps
            # 2nd column is number of neurons / units
            d['cues'] = cues                # tsteps x 4
            d['routs'] = routs              # tsteps x 1000
            d['MDouts'] = MDouts            # tsteps x 2
            d['outs'] = outs                # tsteps x 2
            savemat('simData'+str(ploti), d)
        
        return ax

    def performance(self,cuei,outs,errors,target):
        meanErr = np.mean(errors[-100:,:]*errors[-100:,:])
        # endout is the mean of all end 100 time points for each output
        endout = np.mean(outs[-100:,:],axis=0)
        targeti = 0 if target[0]>target[1] else 1
        non_targeti = 1 if target[0]>target[1] else 0
        ## endout for targeti output must be greater than for the other
        ##  with a margin of 50% of desired difference of 1. between the two
        #if endout[targeti] > (endout[non_targeti]+0.5): correct = 1
        #else: correct = 0
        # just store the margin of error instead of thresholding it
        correct = endout[targeti] - endout[non_targeti]
        return meanErr, correct

    def do_test(self,Ntest,MDeffect,MDCueOff,MDDelayOff,
                    cueList,cuePlot,colNum,train=True):
        NcuesTest = len(cueList)
        MSEs = np.zeros(Ntest*NcuesTest)
        corrects = np.zeros(Ntest*NcuesTest)
        wOuts = np.zeros((Ntest,self.Nout,self.Nneur))
        self.meanAct = np.zeros(shape=(self.Ncontexts*self.inpsPerContext,\
                                        self.tsteps,self.Nneur))
        for testi in range(Ntest):
            if self.plotFigs: print(('Simulating test cycle',testi))
            cues_order = self.get_cues_order(cueList)
            for cuenum,(taski,cuei) in enumerate(cues_order):
                cue, target = self.get_cue_target(taski,cuei)
                cues, routs, outs, MDouts, MDinps, errors = \
                    self.sim_cue(taski,cuei,cue,target,
                            MDeffect,MDCueOff,MDDelayOff,train=train)
                MSEs[testi*NcuesTest+cuenum], corrects[testi*NcuesTest+cuenum] = \
                    self.performance(cuei,outs,errors,target)

                if cuePlot is not None:
                    if self.plotFigs and testi == 0 and taski==cuePlot[0] and cuei==cuePlot[1]:
                        ax = self.plot_column(self.fig,cues,routs,MDouts,outs,ploti=colNum)

            if self.outExternal:
                wOuts[testi,:,:] = self.wOut

        self.meanAct /= Ntest
        if self.plotFigs and cuePlot is not None:
            ax.text(0.1,0.4,'{:1.2f}$\pm${:1.2f}'.format(np.mean(corrects),np.std(corrects)),
                        transform=ax.transAxes)
            ax.text(0.1,0.25,'{:1.2f}$\pm${:1.2f}'.format(np.mean(MSEs),np.std(MSEs)),
                        transform=ax.transAxes)

        if self.saveData:
            # 1-Dim: numCycles * 4 cues/cycle i.e. 70*4=280
            self.fileDict['corrects'+str(colNum)] = corrects
            # at each cycle, a weights matrix 2x1000:
            # weights to 2 output neurons from 1000 cue-selective neurons
            # 3-Dim: 70 (numCycles) x 2 x 1000
            self.fileDict['wOuts'+str(colNum)] = wOuts
            #savemat('simDataTrials'+str(colNum), d)
        
        return MSEs,corrects,wOuts

    def get_cue_list(self,taski=None):
        if taski is not None:
            # (taski,cuei) combinations for one given taski
            cueList = np.dstack(( np.repeat(taski,self.inpsPerContext),
                                    np.arange(self.inpsPerContext) ))
        else:
            # every possible (taski,cuei) combination
            cueList = np.dstack(( np.repeat(np.arange(self.Ncontexts),self.inpsPerContext),
                                    np.tile(np.arange(self.inpsPerContext),self.Ncontexts) ))
        return cueList[0]
    
    def get_next_target(self, taski):
        
        return next(self.data_generator.task_data_gen[taski])

    def train(self,Ntrain):
        MDeffect = self.MDeffect
        if self.blockTrain:
            Nextra = Ntrain//5 #200            # add cycles to show if block1 learning is remembered
            Ntrain = Ntrain*self.Nblocks + Nextra
        else:
            Ntrain *= self.Nblocks
        wOuts = np.zeros(shape=(Ntrain,self.Nout,self.Nneur))
        wPFC2MDs = np.zeros(shape=(Ntrain,2,self.Nneur))
        wMD2PFCs = np.zeros(shape=(Ntrain,self.Nneur,2))
        wMD2PFCMults = np.zeros(shape=(Ntrain,self.Nneur,2))
        MDpreTraces = np.zeros(shape=(Ntrain,self.Nneur))
        #if self.MDlearn:
        #    wPFC2MDs = np.zeros(shape=(Ntrain,2,self.Nneur))
        #    wMD2PFCs = np.zeros(shape=(Ntrain,self.Nneur,2))
        #    wMD2PFCMults = np.zeros(shape=(Ntrain,self.Nneur,2))
        #    MDpreTraces = np.zeros(shape=(Ntrain,self.Nneur))
        
        wJrecs = np.zeros(shape=(Ntrain, 40, 40))
        # Reset the trained weights,
        #  earlier for iterating over MDeffect = False and then True
        if self.outExternal:
            self.wOut = np.random.uniform(-1,1,
                            size=(self.Nout,self.Nneur))/self.Nneur
            self.wOut *= self.wOutMask
        elif not MDeffect:
            self.Jrec[-self.Nout:,:] = \
                np.random.normal(size=(self.Nneur, self.Nneur))\
                            *self.G/np.sqrt(self.Nsub*2)
        # direct connections from cue to output,
        #  similar to having output neurons within reservoir
        if self.dirConn:
            self.wDir = np.random.uniform(-1,1,
                            size=(self.Nout,self.Ncues))\
                            /self.Ncues *1.5
        PFCrates = np.zeros( (Ntrain, self.tsteps, self.Nneur ) )
        MDinputs = np.zeros( (Ntrain, self.tsteps, self.Nmd) )
        MDrates  = np.zeros( (Ntrain, self.tsteps, self.Nmd) )
        Outrates = np.zeros( (Ntrain, self.tsteps, self.Nout  ) )
        Inputs   = np.zeros( (Ntrain, self.inpsPerContext))
        Targets =  np.zeros( (Ntrain, self.Nout))

        MSEs = np.zeros(Ntrain)
        for traini in tqdm.tqdm(range(Ntrain)):
            # if self.plotFigs: print(('Simulating training cycle',traini))
            
            ## reduce learning rate by *10 from 100th and 200th cycle
            #if traini == 100: self.learning_rate /= 10.
            #elif traini == 200: self.learning_rate /= 10.
            
            # if blockTrain,
            #  first half of trials is context1, second half is context2
            if self.blockTrain:
                taski = traini // ((Ntrain-Nextra)//self.Ncontexts)
                # last block is just the first context again
                if traini >= Ntrain-Nextra: taski = 0
                cueList = self.get_cue_list(taski)
            else:
                cueList = self.get_cue_list()
            cues_order = self.get_cues_order(cueList)
            taski,cuei = cues_order[0] # No need for this loop, just pick the first cue, this list is ordered randomly
            #print(taski)
            #print(cuei)
            cue, target = \
                self.get_cue_target(taski,cuei)
            #print(cue)
            #print(target)
            if self.debug == True:
                print('cue:', cue)
                print('target:', target)
            cues, routs, outs, MDouts, MDinps, errors = \
                self.sim_cue(taski,cuei,cue,target,MDeffect=MDeffect,
                train=True)

            PFCrates[traini, :, :] = routs
            MDinputs[traini, :, :] = MDinps
            MDrates [traini, :, :] = MDouts
            Outrates[traini, :, :] = outs
            Inputs  [traini, :]    = np.clip((cue[:2] + cue[2:]), 0., 1.) # go get the input going to either PFC regions. (but clip in case both regions receiving same input)
            Targets [traini, :]    = target

            MSEs[traini] += np.mean(errors*errors)
            if traini ==400:
                print('400 reached')
            wOuts[traini,:,:] = self.wOut
            if self.plotFigs and self.outExternal:
                if self.MDlearn:
                    wPFC2MDs[traini,:,:] = self.wPFC2MD
                    wMD2PFCs[traini,:,:] = self.wMD2PFC
                    wMD2PFCMults[traini,:,:] = self.wMD2PFCMult
                    MDpreTraces[traini,:] = self.MDpreTrace
                if self.reinforceReservoir:
                    wJrecs[traini,:,:] = self.Jrec[:40, 0:25:1000].detach().cpu().numpy() # saving the whole rec is too large, 1000*1000*2200
        self.meanAct /= Ntrain

        if self.saveData:
            self.fileDict['MSEs'] = MSEs
            self.fileDict['wOuts'] = wOuts


        if self.plotFigs:

            # plot output weights evolution
            
            weights= [wOuts, wPFC2MDs, wMD2PFCs,wMD2PFCMults,  wJrecs, MDpreTraces]
            rates =  [PFCrates, MDinputs, MDrates, Outrates, Inputs, Targets, MSEs]
            figMSE=plt.figure(figsize=(10,6), dpi=300)
            plt.plot(MSEs, 'gold', alpha=0.3)
            plt.plot(smooth(MSEs, 8), 'gold', linewidth= pltu.linewidth)
            plt.ylim(-0.01,0.55)
            plt.gca().spines['right'].set_color('none')
            plt.gca().spines['top'].set_color('none')
            figMSE.savefig('results/fig_correlation_{}.pdf'.format(time.strftime("%Y%m%d-%H%M%S")),
                    dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')

            print(len(PFCrates),len(PFCrates[0]))
            print(Inputs.shape)
            index_0_input=np.where(np.squeeze(Inputs[:,0]==0))
            index_1_input=np.where(np.squeeze(Inputs[:,0]==1))
            #print("what input raw data look like")
            #print(index_0_input[0])
            #print(index_1_input[0])
            #print(len(index_0_input[0]))
            #print(len(index_1_input[0]))
            index_cue1_input=np.where(index_0_input[0]<=1000)
            #print(index_cue1_input)
            index_cue2_input=np.where(index_1_input[0]<=1000)
            #print(index_cue2_input)
            index_cue3_input=np.where(np.logical_and(index_0_input[0]>1000, index_0_input[0]<=2000))
            #print(index_cue3_input)
            index_cue4_input=np.where(np.logical_and(index_1_input[0]>1000, index_1_input[0]<=2000))
            #print(index_cue4_input)
            #print("look at input index")
            #print(index_0_input[0][index_cue1_input[0][400]])
            #print(index_1_input[0][index_cue2_input[0][400]])
            #print(index_0_input[0][index_cue3_input[0][400]])
            #print(index_1_input[0][index_cue4_input[0][400]])
            # It is reasonable to take the 400th element of each input trial for the purpose of selecting cells.
            maxrate_PFC_cue1=[]
            maxrate_PFC_cue2=[]
            maxrate_PFC_cue3=[]
            maxrate_PFC_cue4=[]
            print(list(np.array(PFCrates[index_0_input[0][index_cue1_input[0][400]], :, 1]).reshape(200)))
            print(max(list(np.array(PFCrates[index_0_input[0][index_cue1_input[0][400]], :, 1]).reshape(200))))
            for i in range(self.Nneur):
              maxrate_PFC_cue1.append(max(list(np.array(PFCrates[index_0_input[0][index_cue1_input[0][400]], :, i]))))
              maxrate_PFC_cue2.append(max(list(np.array(PFCrates[index_1_input[0][index_cue2_input[0][400]], :, i]))))
              maxrate_PFC_cue3.append(max(list(np.array(PFCrates[index_0_input[0][index_cue3_input[0][400]], :, i]))))
              maxrate_PFC_cue4.append(max(list(np.array(PFCrates[index_1_input[0][index_cue4_input[0][400]], :, i]))))
            #temp_var1=np.array(PFCrates[index_0_input[0][index_cue1_input[0][400]], 91, :])
            #temp_var2=np.array(PFCrates[index_1_input[0][index_cue2_input[0][400]], 91, :])
            #temp_var3=np.array(PFCrates[index_0_input[0][index_cue3_input[0][400]], 91, :])
            #temp_var4=np.array(PFCrates[index_1_input[0][index_cue4_input[0][400]], 91, :])
            temp_var1=np.array(maxrate_PFC_cue1)
            temp_var2=np.array(maxrate_PFC_cue2)
            temp_var3=np.array(maxrate_PFC_cue3)
            temp_var4=np.array(maxrate_PFC_cue4)
            temp_var1=temp_var1.reshape(self.Nneur, 1)
            temp_var2=temp_var2.reshape(self.Nneur, 1)
            temp_var3=temp_var3.reshape(self.Nneur, 1)
            temp_var4=temp_var4.reshape(self.Nneur, 1)
            print("max value work?")
            #print(temp_var1)
            print(temp_var2)
            #print(temp_var3)
            #print(temp_var4)
            print("End max value work?")
          

            temp_context1_index=np.where(np.logical_and(temp_var1>0.3, temp_var2>0.3))
            temp_condition1=np.where(np.logical_and(temp_var3<0.1,temp_var4<0.1))
            print(temp_context1_index[0]) 
            temp_context2_index=np.where(np.logical_and(temp_var3>0.3, temp_var4>0.3))
            temp_condition2=np.where(np.logical_and(temp_var1<0.1,temp_var2<0.1))
            print(len(list(temp_context2_index[0]))) 
            set_context1_index=set(temp_context1_index[0])
            set_context2_index=set(temp_context2_index[0])
            set_condition1=set(temp_condition1[0])
            set_condition2=set(temp_condition2[0])
            context1_selective_index=list((set_context1_index-set_context2_index).intersection(set_condition1))
            context2_selective_index=list((set_context2_index-set_context1_index).intersection(set_condition2))
            set_context1_selective=set(context1_selective_index)
            set_context2_selective=set(context2_selective_index)
            fullset=set(list(range(self.Nneur)))
            set_all_context=set_context1_selective.union(set_context2_selective)
            # This will be important to plot figure 1d
            range_without_context=list(fullset-set_all_context)
            print("what we get after subtracting context index??")
            print(range_without_context)
            print("end of inspection")
            print(set_context1_index.intersection(set_context2_index))
            print(len(context1_selective_index))
            print(len(context2_selective_index))
            rule1_index=np.where(np.logical_and(temp_var1>0.3, temp_var3>0.3))
            rule2_index=np.where(np.logical_and(temp_var2>0.3, temp_var4>0.3))
            condition1=np.where(np.logical_and(temp_var2<0.1, temp_var4<0.1))
            condition2=np.where(np.logical_and(temp_var1<0.1, temp_var3<0.1))
            rule1_set_index=set(rule1_index[0])
            rule2_set_index=set(rule2_index[0])
            condition1_set=set(condition1[0])
            condition2_set=set(condition2[0])
            list_rule1_index=list((rule1_set_index-rule2_set_index).intersection(condition1_set))
            list_rule2_index=list((rule2_set_index-rule1_set_index).intersection(condition2_set))
     #       print(list_rule1_index)
     #       print(list_rule2_index)
            #print(rule1_index[0])
            #print(rule2_index[0])



            range_context1_test=np.where(np.logical_and(temp_var1>0.3, temp_var3<0.1))
            range_context2_test=np.where(np.logical_and(temp_var1<0.1, temp_var3>0.3))
            range_context1_test1=np.where(np.logical_and(temp_var2<0.1, temp_var4<0.1))
            range_context2_test1=np.where(np.logical_and(temp_var2<0.1, temp_var4<0.1))
            # prepare for cue 2 and cue 4
            range_context1_test2=np.where(np.logical_and(temp_var1<0.1, temp_var3<0.1))
            range_context2_test2=np.where(np.logical_and(temp_var1<0.1, temp_var3<0.1))
            range_context1_test3=np.where(np.logical_and(temp_var2>0.3, temp_var4<0.1))
            range_context2_test3=np.where(np.logical_and(temp_var2<0.1, temp_var4>0.3))
            #print("test range first:")
            #print(range_context1_test)
            #print(range_context2_test)
            #print(range_context1_test1)
            #print(range_context2_test1)
            #print(range_context1_test2)
            #print(range_context2_test2)
            #print(range_context1_test3)
            #print(range_context2_test3)
            set_context1_test=set(range_context1_test[0])
            set_context2_test=set(range_context2_test[0])
            set_context1_test1=set(range_context1_test1[0])
            set_context2_test1=set(range_context2_test1[0])
            set_context1_test2=set(range_context1_test2[0])
            set_context2_test2=set(range_context2_test2[0])
            set_context1_test3=set(range_context1_test3[0])
            set_context2_test3=set(range_context2_test3[0])
            print("Test raw data")
            #print(set_context1_test)
            #print(set_context2_test)
            #print(set_context1_test1)
            #print(set_context2_test1)
            #print(set_context1_test2)
            #print(set_context2_test2)
            #print(set_context1_test3)
            #print(set_context2_test3)
            print("end testing raw data")
            cue1_selective_index=list(set_context1_test.intersection(set_context1_test1))
            cue3_selective_index=list(set_context2_test.intersection(set_context2_test1))
            cue2_selective_index=list(set_context1_test3.intersection(set_context1_test2))
            cue4_selective_index=list(set_context2_test3.intersection(set_context2_test2))
            #range_cue2=range_cue1[(np.where(np.logical_and(range_cue1[0] > 200.0, range_cue1[0]<400.0)))]
            #print(range_context1)
            range_cue1_plot=cue1_selective_index
            range_cue2_plot=cue2_selective_index
            range_cue3_plot=cue3_selective_index
            range_cue4_plot=cue4_selective_index

     #       range_part1=np.where(np.logical_and(range_context1[0]>0.0, range_context1[0]<self.Nsub))
     #       range_part2=np.where(np.logical_and(range_context1[0]>self.Nsub, range_context1[0]<self.Nsub*2))
     #       range_part3=np.where(np.logical_and(range_context2[0]>self.Nsub*2, range_context2[0]<self.Nsub*3))
     #       range_part4=np.where(np.logical_and(range_context2[0]>self.Nsub*3, range_context2[0]<self.Nsub*4))
     #       print(range_part1[0])
     #       print(range_part2[0])
     #       print(range_part3[0])
     #       print(range_part4[0])
     #       range_cue1_plot=[]
     #       range_cue2_plot=[]
     #       range_cue3_plot=[]
     #       range_cue4_plot=[]
     #       for i in range_part1[0]:
     #         range_cue1_plot.append(range_context1[0][i])
     #       for i in range_part2[0]:
     #         range_cue2_plot.append(range_context1[0][i])
     #       for i in range_part3[0]:
     #         range_cue3_plot.append(range_context2[0][i])
     #       for i in range_part4[0]:
     #         range_cue4_plot.append(range_context2[0][i])
     #     following is 6 differnet clusters that we want
            #print("The following is 6 clusters")
            #print(range_cue1_plot)
            #print(range_cue2_plot)
            #print(range_cue3_plot)
            #print(range_cue4_plot)
            #print(list_rule1_index)
            #print(list_rule2_index)
            
            print("Take 6 cells to from 6 trace across trials")
            #print(range_cue1_plot[0])
            #print(range_cue2_plot[0])
            ##print(range_cue3_plot[0])
            #print(range_cue4_plot[0])
            print(list_rule1_index[0])
            print(list_rule2_index[0])
            print("start statistics count")
            cue_selective_number=len(range_cue1_plot)+len(range_cue2_plot)+len(range_cue3_plot)+len(range_cue4_plot)
            print("number of each type of cues:")
            print("cue 1 is ", len(range_cue1_plot), " | cue 2 is ", len(range_cue2_plot), " | cue 3 is ", len(range_cue3_plot), " | cue 4 is ", len(range_cue4_plot))
            print("Total number of cue-selective cells:")
            print(cue_selective_number)
            print("Total number of cue-invariant cells:")
            rule_number=len(list_rule1_index)+len(list_rule2_index)
            print(rule_number) 
            print("Total number of context-selective cells:")
            print(len(list(set_all_context)))
            print("Total number of non-coding cells:")
            print(self.Nneur-cue_selective_number-rule_number-len(list(set_all_context)))
            print("end statistics count")
            # Now we constrcut traces of firing rate for the particular neuron selected above:
            #vec1=np.array(PFCrates[:,95,range_cue1_plot[0]])
            #vec2=np.array(PFCrates[:,95,range_cue2_plot[0]])
            #vec3=np.array(PFCrates[:,95,range_cue3_plot[0]])
            #vec4=np.array(PFCrates[:,95,range_cue4_plot[0]])
            #vec5=np.array(PFCrates[:,95,list_rule1_index[0]])
            #vec6=np.array(PFCrates[:,95,list_rule2_index[0]])
            #vecall=np.concatenate((vec1, vec2, vec3, vec4, vec5, vec6), axis=0)
            #print("the concatenated vector is as follow:")
            #print(vecall)
            #print("end concatenated vector")
            #print("construct correlation matrix")
            #coefficient1_raw=np.corrcoef(vec1, vec2)
            #coefficient1=coefficient1_raw[0,1]
            #coefficient2_raw=np.corrcoef(vec1, vec3)
            #coefficient2=coefficient2_raw[0,1]
            #coefficient3_raw=np.corrcoef(vec1, vec4)
            #coefficient3=coefficient3_raw[0,1]
            #coefficient4_raw=np.corrcoef(vec1, vec5)
            #coefficient4=coefficient4_raw[0,1]
            #coefficient5_raw=np.corrcoef(vec1, vec6)
            #coefficient5=coefficient5_raw[0,1]
            #coefficient6_raw=np.corrcoef(vec2, vec3)
            #coefficient6=coefficient6_raw[0,1]
            #coefficient7_raw=np.corrcoef(vec2, vec4)
            #coefficient7=coefficient7_raw[0,1]
            #coefficient8_raw=np.corrcoef(vec2, vec5)
            #coefficient8=coefficient8_raw[0,1]
            #coefficient9_raw=np.corrcoef(vec2, vec6)
            #coefficient9=coefficient9_raw[0,1]
            #coefficient10_raw=np.corrcoef(vec3, vec4)
            #coefficient10=coefficient10_raw[0,1]
            #coefficient11_raw=np.corrcoef(vec3, vec5)
            #coefficient11=coefficient11_raw[0,1]
            #coefficient12_raw=np.corrcoef(vec3, vec6)
            #coefficient12=coefficient12_raw[0,1]
            #coefficient13_raw=np.corrcoef(vec4, vec5)
            #coefficient13=coefficient13_raw[0,1]
            #coefficient14_raw=np.corrcoef(vec4, vec6)
            #coefficient14=coefficient14_raw[0,1]
            #coefficient15_raw=np.corrcoef(vec5, vec6)
            #coefficient15=coefficient15_raw[0,1]
            #coefficient16_raw=np.corrcoef(vec1, vec1)
            #coefficient16=coefficient16_raw[0,1]
            #coefficient17_raw=np.corrcoef(vec2, vec2)
            #coefficient17=coefficient17_raw[0,1]
            #coefficient18_raw=np.corrcoef(vec3, vec3)
            #coefficient18=coefficient18_raw[0,1]
            #coefficient19_raw=np.corrcoef(vec4, vec4)
            #coefficient19=coefficient19_raw[0,1]
            #coefficient20_raw=np.corrcoef(vec5, vec5)
            #coefficient20=coefficient20_raw[0,1]
            #coefficient21_raw=np.corrcoef(vec6, vec6)
            #coefficient21=coefficient21_raw[0,1]
            #correlation_matrix=[[coefficient16, coefficient1, coefficient2, coefficient3, coefficient4,  coefficient5],
            #                    [coefficient1, coefficient17, coefficient6, coefficient7,  coefficient8,  coefficient9],
            #                    [coefficient2, coefficient6, coefficient18, coefficient10, coefficient11, coefficient12],
            #                    [coefficient3, coefficient7, coefficient10, coefficient19, coefficient13, coefficient14],
            #                    [coefficient4, coefficient8, coefficient11, coefficient13, coefficient20, coefficient15],
            #                    [coefficient5, coefficient9, coefficient12, coefficient14, coefficient15, coefficient21]]
            #print(np.array(correlation_matrix))
            #print(np.array(correlation_matrix).shape)
            #import seaborn as sns
            #figMatrix=plt.figure(figsize=(10,6), dpi=300)
            #im=plt.imshow(np.array(correlation_matrix), cmap='coolwarm', interpolation='nearest')
            #plt.colorbar(im)
            #figMatrix.savefig('results/fig_correlation_{}.pdf'.format(time.strftime("%Y%m%d-%H%M%S")),
            #        dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')
            #sns.heatmap(np.array(correlation_matrix))
            #plt.show()
            #range_cue2=range_cue1[(range_part2)]
            #print(range_cue2)
            #range_cue3=np.where(np.logical_and(temp1<0.3, temp2>0.25))
            #datamean1=np.mean(np.mean(PFCrates[:,:,np.array(range_cue1_plot[:5]).reshape(len(range_cue1_plot[:5]),1)], axis=1),axis=2)
            #datamean2=np.mean(np.mean(PFCrates[:,:,np.array(range_cue2_plot[:5]).reshape(len(range_cue2_plot[:5]),1)], axis=1),axis=2)
            #datamean3=np.mean(np.mean(PFCrates[:,:,np.array(range_cue3_plot[:5]).reshape(len(range_cue3_plot[:5]),1)], axis=1),axis=2)
            #datamean4=np.mean(np.mean(PFCrates[:,:,np.array(range_cue4_plot[:5]).reshape(len(range_cue4_plot[:5]),1)], axis=1),axis=2)
            #datamean5=np.mean(wMD2PFCs[:,np.array(range_cue1_plot).reshape(len(range_cue1_plot),1), 0],axis=1)
            #datamean6=np.mean(wMD2PFCs[:,np.array(range_cue3_plot).reshape(len(range_cue3_plot),1), 0],axis=1)
            #datamean7=np.mean(wMD2PFCs[:,np.array(range_cue1_plot).reshape(len(range_cue1_plot),1), 1],axis=1)
            #datamean8=np.mean(wMD2PFCs[:,np.array(range_cue3_plot).reshape(len(range_cue3_plot),1), 1],axis=1)
            #xaxisnumber=np.arange(Ntrain)
            #from scipy import stats
            #dataste5=stats.sem(np.array(wMD2PFCs[:,np.array(range_cue1_plot).reshape(len(range_cue1_plot),1), 0]).reshape(len(range_cue1_plot),Ntrain))
            #dataste6=stats.sem(np.array(wMD2PFCs[:,np.array(range_cue2_plot).reshape(len(range_cue2_plot),1), 0]).reshape(len(range_cue2_plot),Ntrain))
            #dataste7=stats.sem(np.array(wMD2PFCs[:,np.array(range_cue3_plot).reshape(len(range_cue3_plot),1), 0]).reshape(len(range_cue3_plot),Ntrain))
            #dataste8=stats.sem(np.array(wMD2PFCs[:,np.array(range_cue4_plot).reshape(len(range_cue4_plot),1), 0]).reshape(len(range_cue4_plot),Ntrain))
            #print(dataste5)
            #print(len(dataste5))
            #print(range_cue2)
              
      #      fig, axs = plt.subplots(2, 2, constrained_layout=True)
      #      #for i in range(5):
      #      axs[0,0].plot(PFCrates[52, :, range_cue2_plot[3]])
      #      axs[0,0].set_title('context 1')
      #      axs[0,0].set_xlabel('time (ms)')
      #      axs[0,0].set_ylabel('rate')
      #      axs[0,0].set_ylim([0,0.7])
      #      fig.suptitle('cue 2', fontsize=16)
      #      #for i in range(5):
      #      #  for j in range(20):
      #      axs[0,1].plot(PFCrates[50, :, range_cue2_plot[3]])
      #      axs[0,1].set_ylim([0,0.7])
      #      axs[1,0].plot(PFCrates[1050, :, range_cue2_plot[3]])
      #      axs[1,0].set_ylim([0,0.7])
      #      axs[1,1].plot(PFCrates[1052, :, range_cue2_plot[3]])
      #      axs[1,1].set_ylim([0,0.7])
      #      fig.savefig('results/fig_withintrial_{}.pdf'.format(time.strftime("%Y%m%d-%H%M%S")),
      #              dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')
                       
            #figContextPFC, axs = plt.subplots(2, 2, constrained_layout=True)
            #figContextPFC.suptitle('context cell', fontsize=16)
            #axs[0,0].plot(PFCrates[52, :, context1_selective_index[3]])
            #axs[0,0].set_xlabel('time (ms)')
            #axs[0,0].set_ylabel('rate')
            #axs[0,0].set_ylim([0,0.7])
            #axs[0,1].plot(PFCrates[50, :, context1_selective_index[3]])
            #axs[0,1].set_ylim([0,0.7])
            #axs[1,0].plot(PFCrates[1050, :, context1_selective_index[3]])
            #axs[1,0].set_ylim([0,0.7])
            #axs[1,1].plot(PFCrates[1052, :, context1_selective_index[3]])
            #axs[1,1].set_ylim([0,0.7])
            #figContextPFC.savefig('results/fig_context_PFC_{}.pdf'.format(time.strftime("%Y%m%d-%H%M%S")),
            #        dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')


       #     plt.subplot(611)
       #     plt.plot(PFCrates[900,91,:])
       #     plt.subplot(612)
       #     #for i in range_context1[0]:
       #     plt.plot(PFCrates[900, :, 30])
       #     plt.subplot(613)
       #     plt.plot(datamean1, 'bo', markersize=0.1)
       #     plt.subplot(614)
       #     plt.plot(datamean2, 'bo', markersize=0.1)
       #     plt.subplot(615)
       #     plt.plot(datamean3, 'bo', markersize=0.1)
       #     plt.subplot(616)
       #     plt.plot(datamean4, 'bo', markersize=0.1)
            plot_weights(self, weights)
            plot_rates(self, rates)
            
       #     plt.subplot(411)  
       #     plt.plot(datamean5, 'b', linewidth=2)
       #     plt.fill_between(xaxisnumber, np.squeeze(datamean5)-np.squeeze(dataste5), np.squeeze(datamean5)+np.squeeze(dataste5), color='blue', alpha=0.2)
       #     plt.ylim([-0.0628,0.0628])
       #     plt.subplot(412)  
       #     plt.plot(datamean6, 'r', linewidth=2)
       #     plt.fill_between(xaxisnumber, np.squeeze(datamean6)-np.squeeze(dataste6), np.squeeze(datamean6)+np.squeeze(dataste6), color='red', alpha=0.2)
       #     plt.ylim([-0.0628,0.0628])
       #     plt.subplot(413)  
       #     plt.plot(datamean7, 'b', linewidth=2)
       #     plt.fill_between(xaxisnumber, np.squeeze(datamean7)-np.squeeze(dataste7), np.squeeze(datamean7)+np.squeeze(dataste7), color='blue', alpha=0.2)
       #     plt.ylim([-0.0628,0.0628])
       #     plt.subplot(414)  
       #     plt.plot(datamean8, 'r', linewidth=2)
       #     plt.fill_between(xaxisnumber, np.squeeze(datamean8)-np.squeeze(dataste8), np.squeeze(datamean8)+np.squeeze(dataste8), color='red', alpha=0.2)
       #     plt.ylim([-0.0628,0.0628])
            
       #     print(Inputs[50:70])
       #     print(Inputs.shape)
       #     side_trial=Inputs[:,0]
       #     context_trials=np.zeros(shape=Ntrain)
            
            #figCue1=plt.figure(figsize=(10,6), dpi=300)
       #     figCue1, axs = plt.subplots(2, 2, constrained_layout=True)
       # #    axs[0,0].plot(np.array(PFCrates[index_0_input[0][index_cue1_input[0][400]], :, range_cue1_plot[2]]).reshape(200))       
       #     axs[0,0].plot(np.array(PFCrates[index_0_input[0][index_cue1_input[0][400]], :, 40]).reshape(200))       
       #     axs[0,0].set_ylim([0,1.1])
       # #    axs[0,1].plot(np.array(PFCrates[index_1_input[0][index_cue2_input[0][400]], :, range_cue1_plot[2]]).reshape(200))       
       #     axs[0,1].plot(np.array(PFCrates[index_1_input[0][index_cue2_input[0][400]], :, 40]).reshape(200))       
       #     axs[0,1].set_ylim([0,1.1])
       # #    axs[1,0].plot(np.array(PFCrates[index_0_input[0][index_cue3_input[0][400]], :, range_cue1_plot[2]]).reshape(200))       
       #     axs[1,0].plot(np.array(PFCrates[index_0_input[0][index_cue3_input[0][400]], :, 40]).reshape(200))       
       #     axs[1,0].set_ylim([0,1.1])
       # #    axs[1,1].plot(np.array(PFCrates[index_1_input[0][index_cue4_input[0][400]], :, range_cue1_plot[2]]).reshape(200))       
       #     axs[1,1].plot(np.array(PFCrates[index_1_input[0][index_cue4_input[0][400]], :, 40]).reshape(200))       
       #     axs[1,1].set_ylim([0,1.1])
       #     figCue1.savefig('results/fig_1c_cue1_{}.pdf'.format(time.strftime("%Y%m%d-%H%M%S")),
       #             dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')
            #here we plot for rule
       #     figRule1, axs = plt.subplots(2, 2, constrained_layout=True)
       #     figRule1.suptitle("Rule cell example", fontsize=16)
       #     # tried 3
       #     axs[0,0].plot(np.array(PFCrates[index_0_input[0][index_cue1_input[0][400]], :, list_rule1_index[0]]).reshape(200), color='black')       
       #     axs[0,0].set_ylim([0,1.1])
       #     axs[0,1].plot(np.array(PFCrates[index_1_input[0][index_cue2_input[0][400]], :, list_rule1_index[0]]).reshape(200), color='black')       
       #     axs[0,1].set_ylim([0,1.1])
       #     axs[1,0].plot(np.array(PFCrates[index_0_input[0][index_cue3_input[0][400]], :, list_rule1_index[0]]).reshape(200), color='black')       
       #     axs[1,0].set_ylim([0,1.1])
       #     axs[1,1].plot(np.array(PFCrates[index_1_input[0][index_cue4_input[0][400]], :, list_rule1_index[0]]).reshape(200), color='black')       
       #     axs[1,1].set_ylim([0,1.1])
       #     figRule1.savefig('results/fig_1c_rule_{}.pdf'.format(time.strftime("%Y%m%d-%H%M%S")),
       #             dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')
            #Here we plot for thalamus cell
       #     figMD1, axs = plt.subplots(2, 2, constrained_layout=True)
       #     figMD1.suptitle("thalamus cell example", fontsize=16)
       #     axs[0,0].plot(np.array(MDrates[index_0_input[0][index_cue1_input[0][400]], :, 0]).reshape(200), color='green')       
       #     axs[0,0].set_ylim([0,1.1])
       #     axs[0,1].plot(np.array(MDrates[index_1_input[0][index_cue2_input[0][400]], :, 0]).reshape(200), color='green')       
       #     axs[0,1].set_ylim([0,1.1])
       #     axs[1,0].plot(np.array(MDrates[index_0_input[0][index_cue3_input[0][400]], :, 0]).reshape(200), color='green')       
       #     axs[1,0].set_ylim([0,1.1])
       #     axs[1,1].plot(np.array(MDrates[index_1_input[0][index_cue4_input[0][400]], :, 0]).reshape(200), color='green')       
       #     axs[1,1].set_ylim([0,1.1])
       #     figMD1.savefig('results/fig_1c_MD1_{}.pdf'.format(time.strftime("%Y%m%d-%H%M%S")),
       #             dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')
            #Plot figure 2c
       #     figMD1, axs = plt.subplots(2, 2, constrained_layout=True)
       #     figMD1.suptitle("thalamus cell transition", fontsize=16)
       #     axs[0,0].plot(np.array(MDrates[1, :, 0]).reshape(200), color='green')       
       #     axs[0,0].set_ylim([0,1.1])
       #     axs[0,1].plot(np.array(MDrates[12, :, 0]).reshape(200), color='green')       
       #     axs[0,1].set_ylim([0,1.1])
       #     axs[1,0].plot(np.array(MDrates[15, :, 0]).reshape(200), color='green')       
       #     axs[1,0].set_ylim([0,1.1])
       #     axs[1,1].plot(np.array(MDrates[20, :, 0]).reshape(200), color='green')       
       #     axs[1,1].set_ylim([0,1.1])
       #     figMD1.savefig('results/fig_2c_MD1_{}.pdf'.format(time.strftime("%Y%m%d-%H%M%S")),
       #             dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')
            #Plot figure 2c mean
            figMD1, axs = plt.subplots(3, 2, constrained_layout=True)
            figMD1.suptitle("thalamus cell transition mean rate", fontsize=16)
            a=np.mean(np.array(MDrates[1, :, 0]))       
            b=np.mean(np.array(MDrates[12, :, 0]))       
            c=np.mean(np.array(MDrates[26, :, 0]))       
            d=np.mean(np.array(MDrates[50, :, 0]))     
            trialnum=['12', '26', '50']  
            barlist=[b,c,d]
            axs[0,0].bar(trialnum, barlist)       
            axs[0,0].set_ylim([0,1.1])
            a=np.mean(np.array(MDrates[1, :, 1]))       
            b=np.mean(np.array(MDrates[13, :, 1]))       
            c=np.mean(np.array(MDrates[27, :, 1]))       
            d=np.mean(np.array(MDrates[51, :, 1]))     
            trialnum=['13', '27', '51']  
            barlist=[b,c,d]
            axs[0,1].bar(trialnum, barlist)       
            axs[0,1].set_ylim([0,1.1])
            a=np.mean(np.array(MDrates[950, :, 0]))       
            b=np.mean(np.array(MDrates[995, :, 0]))       
            c=np.mean(np.array(MDrates[1037, :, 0]))       
            d=np.mean(np.array(MDrates[1069, :, 0]))     
            trialnum=['995', '1037', '1069']  
            barlist=[b,c,d]
            axs[1,0].bar(trialnum, barlist)       
            axs[1,0].set_ylim([0,1.1])
            a=np.mean(np.array(MDrates[950, :, 1]))       
            b=np.mean(np.array(MDrates[996, :, 1]))       
            c=np.mean(np.array(MDrates[1036, :, 1]))       
            d=np.mean(np.array(MDrates[1066, :, 1]))     
            trialnum=['996', '1036', '1066']  
            barlist=[b,c,d]
            axs[1,1].bar(trialnum, barlist)       
            axs[1,1].set_ylim([0,1.1])
            a=np.mean(np.array(MDrates[1950, :, 0]))       
            b=np.mean(np.array(MDrates[2001, :, 0]))       
            c=np.mean(np.array(MDrates[2035, :, 0]))       
            d=np.mean(np.array(MDrates[2050, :, 0]))     
            trialnum=['2001', '2035', '2050']  
            barlist=[b,c,d]
            axs[2,0].bar(trialnum, barlist)       
            axs[2,0].set_ylim([0,1.1])
            a=np.mean(np.array(MDrates[1950, :, 1]))       
            b=np.mean(np.array(MDrates[2002, :, 1]))       
            c=np.mean(np.array(MDrates[2036, :, 1]))       
            d=np.mean(np.array(MDrates[2051, :, 1]))     
            trialnum=['2002', '2036', '2051']  
            barlist=[b,c,d]
            axs[2,1].bar(trialnum, barlist)       
            axs[2,1].set_ylim([0,1.1])
            figMD1.savefig('results/fig_2c_MD_bar_{}.pdf'.format(time.strftime("%Y%m%d-%H%M%S")),
                    dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')

            extremetemp_cue1=np.array(PFCrates[index_0_input[0][index_cue1_input[0][400]],195,range_cue4_plot])
            extremetemp_cue2=np.array(PFCrates[index_1_input[0][index_cue2_input[0][400]],195,range_cue4_plot])
            extremetemp_cue3=np.array(PFCrates[index_0_input[0][index_cue3_input[0][400]],195,range_cue4_plot])
            extremetemp_cue4=np.array(PFCrates[index_1_input[0][index_cue4_input[0][400]],195,range_cue4_plot])
            extremecue4=np.where(np.logical_and(extremetemp_cue4>0.3, extremetemp_cue3<0.1))
            
  #          for i in range(1000):
  #             context_trials[i+1000]=1
            accuracy_array=np.arange(200)
            accuracy_array1=np.arange(200)
            MD_accuracy_array=np.arange(200)
            MD_accuracy_array1=np.arange(200)
            from sklearn.svm import SVC
            from sklearn.metrics import classification_report, confusion_matrix
            from sklearn.model_selection import cross_val_score
            #This for loop is for data of rule
     #       for i in range(200):
     #         #PFCdataset=np.array(PFCrates[:, i, :]).reshape(Ntrain, self.Nneur)
     #         #PFCdataset=np.array(PFCrates[:, i, range_without_context]).reshape(Ntrain, len(range_without_context))
     #         PFCdataset=np.array(PFCrates[:, i, range_cue4_plot]).reshape(Ntrain, len(range_cue4_plot))
     #         #PFCdataset=np.array(PFCrates[:, i, extremecue4[0]]).reshape(Ntrain, len(extremecue4[0]))
     #         #PFCdataset=np.array(PFCrates[:, i, list_rule2_index]).reshape(Ntrain, len(list_rule2_index))
     #         #PFCdataset=np.array(PFCrates[:, i, context2_selective_index]).reshape(Ntrain, len(context2_selective_index))
     #        #PFCdataset=np.array(MDrates[:, i, 1]).reshape(Ntrain, 1)
     #         PFCdata_binary=np.where(PFCdataset<0.2, 0, 1)
     #       #print(side_trial)
     #       #print(PFCdata_binary[10, 1:500])
     #         X_train, X_test, y_train, y_test = train_test_split(PFCdata_binary, side_trial,test_size=0.2) 
     #         svclassifier=SVC(kernel='sigmoid')
     #         svclassifier.fit(X_train,y_train)
     #         y_pred=svclassifier.predict(X_test)
     #         cm=confusion_matrix(y_test,y_pred)
     #         print(confusion_matrix(y_test, y_pred))
     #         print(classification_report(y_test, y_pred))
     #         accuracy_confusian=float(cm.diagonal().sum())/len(y_test)*100
     #         accuracy=cross_val_score(svclassifier, X_train, y_train, scoring='accuracy', cv=10).mean()*100
     #         print("Accuracy is: ", accuracy)
     #         print("Accuracy using confusion matrix is: ", accuracy_confusian)
     #         accuracy_array[i]=accuracy
     #         accuracy_array1[i]=accuracy_confusian
            #print(cue)
     #       np.savetxt('accuracy_PFC_transient_rule.csv', [accuracy_array], delimiter=' ', fmt='%f')
            # add code for rule classification of MD cells here
     #       for i in range(200):
     #         #PFCdataset=np.array(PFCrates[:, i, :]).reshape(Ntrain, self.Nneur)
     #         #PFCdataset=np.array(PFCrates[:, i, range_without_context]).reshape(Ntrain, len(range_without_context))
     #         #PFCdataset=np.array(PFCrates[:, i, range_cue4_plot]).reshape(Ntrain, len(range_cue4_plot))
     #         #PFCdataset=np.array(PFCrates[:, i, extremecue4[0]]).reshape(Ntrain, len(extremecue4[0]))
     #         #PFCdataset=np.array(PFCrates[:, i, list_rule2_index]).reshape(Ntrain, len(list_rule2_index))
     #         #PFCdataset=np.array(PFCrates[:, i, context2_selective_index]).reshape(Ntrain, len(context2_selective_index))
     #         PFCdataset=np.array(MDrates[:, i, :]).reshape(Ntrain, 2)
     #         PFCdata_binary=np.where(PFCdataset<0.2, 0, 1)
            #print(side_trial)
            #print(PFCdata_binary[10, 1:500])
     #         X_train, X_test, y_train, y_test = train_test_split(PFCdata_binary, side_trial,test_size=0.2) 
     #         svclassifier=SVC(kernel='sigmoid')
     #         svclassifier.fit(X_train,y_train)
     #         y_pred=svclassifier.predict(X_test)
     #         cm=confusion_matrix(y_test,y_pred)
     #         print(confusion_matrix(y_test, y_pred))
     #         print(classification_report(y_test, y_pred))
     #         accuracy_confusian=float(cm.diagonal().sum())/len(y_test)*100
     #         accuracy=cross_val_score(svclassifier, X_train, y_train, scoring='accuracy', cv=10).mean()*100
     #         print("Accuracy is: ", accuracy)
     #         print("Accuracy using confusion matrix is: ", accuracy_confusian)
     #         MD_accuracy_array[i]=accuracy
     #         MD_accuracy_array1[i]=accuracy_confusian
            #print(cue)
     #       np.savetxt('accuracy_MD_transient_rule.csv', [accuracy_array], delimiter=' ', fmt='%f')
     #       accuracy_running_mean=[]
     #       accuracy_running_ste=[]
     #       xaxis=[]
     #       MD_accuracy_running_mean=[]
     #       MD_accuracy_running_ste=[]
            from scipy import stats
     #       for i in range(20):
     #           for j in range(10):
     #             accuracy_running_mean.append(np.mean(accuracy_array[i*10:(i+1)*10]))
     #             accuracy_running_ste.append(stats.sem(accuracy_array[i*10:(i+1)*10]))
     #             MD_accuracy_running_mean.append(np.mean(MD_accuracy_array[i*10:(i+1)*10]))
     #             MD_accuracy_running_ste.append(stats.sem(MD_accuracy_array[i*10:(i+1)*10]))
     #             xaxis.append(i*10)
     #       figRuleClassify=plt.figure(figsize=(10,6), dpi=300)
     #       plt.plot(xaxis, accuracy_running_mean, color="black")
     #       plt.plot(xaxis, MD_accuracy_running_mean, color="green")
     #       plt.fill_between(np.array(xaxis), np.array(accuracy_running_mean)-np.array(accuracy_running_ste), np.array(accuracy_running_mean)+np.array(accuracy_running_ste), color="black", alpha=0.2)
     #       plt.fill_between(np.array(xaxis), np.array(MD_accuracy_running_mean)-np.array(MD_accuracy_running_ste), np.array(MD_accuracy_running_mean)+np.array(MD_accuracy_running_ste), color="green", alpha=0.2)
            #plt.plot(accuracy_array1)
     #       plt.title("PFC context classification")
     #       figRuleClassify.savefig('results/fig_PFCMD_rule_classification_{}.pdf'.format(time.strftime("%Y%m%d-%H%M%S")),
     #               dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')
            
            # end rule classification for MD cells 
 #           fig4=plt.figure(figsize=(10,6), dpi=300)
 #           plt.plot(accuracy_array)
 #           plt.plot(accuracy_array1)
 #           plt.title("PFC rule classification")
 #           fig4.savefig('results/fig_rule_PFC_accuracy_{}.pdf'.format(time.strftime("%Y%m%d-%H%M%S")),
 #                   dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')
            #end for rule
            #This for loop is for data of context
   #         for i in range(200):
   #           #PFCdataset=np.array(PFCrates[:, i, :]).reshape(Ntrain, self.Nneur)
   #           #PFCdataset=np.array(PFCrates[:, i, range_without_context]).reshape(Ntrain, len(range_without_context))
   #           PFCdataset=np.array(PFCrates[:, i, range_cue4_plot]).reshape(Ntrain, len(range_cue4_plot))
   #           #PFCdataset=np.array(PFCrates[:, i, extremecue4[0]]).reshape(Ntrain, len(extremecue4[0]))
   #           #PFCdataset=np.array(PFCrates[:, i, list_rule2_index]).reshape(Ntrain, len(list_rule2_index))
   #           #PFCdataset=np.array(PFCrates[:, i, context2_selective_index]).reshape(Ntrain, len(context2_selective_index))
   #           #PFCdataset=np.array(MDrates[:, i, 1]).reshape(Ntrain, 1)
   #           PFCdata_binary=np.where(PFCdataset<0.2, 0, 1)
   #         #print(side_trial)
   #         #print(PFCdata_binary[10, 1:500])
   #           X_train, X_test, y_train, y_test = train_test_split(PFCdata_binary, context_trials,test_size=0.2) 
   #           svclassifier=SVC(kernel='sigmoid')
   #           svclassifier.fit(X_train,y_train)
   #           y_pred=svclassifier.predict(X_test)
   #           cm=confusion_matrix(y_test,y_pred)
   #           print(confusion_matrix(y_test, y_pred))
   #           print(classification_report(y_test, y_pred))
   #           accuracy_confusian=float(cm.diagonal().sum())/len(y_test)*100
   #           accuracy=cross_val_score(svclassifier, X_train, y_train, scoring='accuracy', cv=10).mean()*100
   #           print("Accuracy for context is: ", accuracy)
   #           print("Accuracy using confusion matrix is: ", accuracy_confusian)
   #           accuracy_array[i]=accuracy
   #           accuracy_array1[i]=accuracy_confusian
   #         #print(cue)
   #         np.savetxt('accuracy_PFC_context.csv', [accuracy_array], delimiter=' ', fmt='%f')
         # add the computation of context classification of MD cells here:
            #This for loop is for data of context
   #         for i in range(200):
   #           #PFCdataset=np.array(PFCrates[:, i, :]).reshape(Ntrain, self.Nneur)
   #           #PFCdataset=np.array(PFCrates[:, i, range_without_context]).reshape(Ntrain, len(range_without_context))
   #           #PFCdataset=np.array(PFCrates[:, i, range_cue4_plot]).reshape(Ntrain, len(range_cue4_plot))
   #           #PFCdataset=np.array(PFCrates[:, i, extremecue4[0]]).reshape(Ntrain, len(extremecue4[0]))
   #           #PFCdataset=np.array(PFCrates[:, i, list_rule2_index]).reshape(Ntrain, len(list_rule2_index))
   #           #PFCdataset=np.array(PFCrates[:, i, context2_selective_index]).reshape(Ntrain, len(context2_selective_index))
   #           PFCdataset=np.array(MDrates[:, i, :]).reshape(Ntrain, 2)
   #           PFCdata_binary=np.where(PFCdataset<0.2, 0, 1)
   #         #print(side_trial)
   #         #print(PFCdata_binary[10, 1:500])
   #           X_train, X_test, y_train, y_test = train_test_split(PFCdata_binary, context_trials,test_size=0.2) 
  #            svclassifier=SVC(kernel='sigmoid')
  #            svclassifier.fit(X_train,y_train)
  #            y_pred=svclassifier.predict(X_test)
  #            cm=confusion_matrix(y_test,y_pred)
  #            print(confusion_matrix(y_test, y_pred))
  #            print(classification_report(y_test, y_pred))
  #            accuracy_confusian=float(cm.diagonal().sum())/len(y_test)*100
  #            accuracy=cross_val_score(svclassifier, X_train, y_train, scoring='accuracy', cv=10).mean()*100
  #            print("Accuracy for context is: ", accuracy)
  #            print("Accuracy using confusion matrix is: ", accuracy_confusian)
  #            MD_accuracy_array[i]=accuracy
  #            MD_accuracy_array1[i]=accuracy_confusian
            #print(cue)
  #          np.savetxt('accuracy_MD_context.csv', [accuracy_array], delimiter=' ', fmt='%f')
         # end of computation of context classification 
  #          accuracy_running_mean=[]
  #          accuracy_running_ste=[]
  #          xaxis=[]
  #          MD_accuracy_running_mean=[]
  #          MD_accuracy_running_ste=[]
            from scipy import stats
  #          for i in range(20):
  #              for j in range(10):
  #                accuracy_running_mean.append(np.mean(accuracy_array[i*10:(i+1)*10]))
  #                accuracy_running_ste.append(stats.sem(accuracy_array[i*10:(i+1)*10]))
  #                MD_accuracy_running_mean.append(np.mean(MD_accuracy_array[i*10:(i+1)*10]))
  #                MD_accuracy_running_ste.append(stats.sem(MD_accuracy_array[i*10:(i+1)*10]))
  #                xaxis.append(i*10)
  #          figContextClassify=plt.figure(figsize=(10,6), dpi=300)
  #          plt.plot(xaxis, accuracy_running_mean, color="black")
  #          plt.plot(xaxis, MD_accuracy_running_mean, color="green")
  #          plt.fill_between(np.array(xaxis), np.array(accuracy_running_mean)-np.array(accuracy_running_ste), np.array(accuracy_running_mean)+np.array(accuracy_running_ste), color="black", alpha=0.2)
  #          plt.fill_between(np.array(xaxis), np.array(MD_accuracy_running_mean)-np.array(MD_accuracy_running_ste), np.array(MD_accuracy_running_mean)+np.array(MD_accuracy_running_ste), color="green", alpha=0.2)
            #plt.plot(accuracy_array1)
  #          plt.title("PFC context classification")
  #          figContextClassify.savefig('results/fig_PFC_context_classification_{}.pdf'.format(time.strftime("%Y%m%d-%H%M%S")),
  #                  dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')
            #end for context

            #copy end here
           # fig5=plt.figure(figsize=(10,6), dpi=300)
           # #for i in range(990, 1010):
           # plt.plot(np.array(PFCrates[995,:,range_cue1_plot[3]]).reshape(200,1), color='red')
           # plt.plot(np.array(PFCrates[1001,:,range_cue1_plot[3]]).reshape(200,1), color='black')
           # plt.plot(np.array(PFCrates[1004,:,range_cue1_plot[3]]).reshape(200,1), color='yellow')
           # fig5.savefig('results/fig_transition_PFC_{}.pdf'.format(time.strftime("%Y%m%d-%H%M%S")),
           #         dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')
            #fig6=plt.figure(figsize=(10,6), dpi=300)
         #   fig6, axs = plt.subplots(1, 2, constrained_layout=True)
         #   axs[0].plot(np.array(MDrates[995,:,0]).reshape(200,1), color='forestgreen', alpha=0.1)
         #   axs[0].plot(np.array(MDrates[1037,:,0]).reshape(200,1), color='forestgreen', alpha=0.5)
         #   axs[0].plot(np.array(MDrates[1069,:,0]).reshape(200,1), color='forestgreen')
         #   axs[1].plot(np.array(MDrates[996,:,1]).reshape(200,1), color='lightgreen', alpha=0.1)
         #   axs[1].plot(np.array(MDrates[1036,:,1]).reshape(200,1), color='lightgreen', alpha=0.5)
         #   axs[1].plot(np.array(MDrates[1066,:,1]).reshape(200,1), color='lightgreen')
         #   axs[0].set_ylim(0,1)
         #   axs[1].set_ylim(0,1)
         #   fig6.savefig('results/fig_thlamus_transition_PFC_{}.pdf'.format(time.strftime("%Y%m%d-%H%M%S")),
         #           dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')

         #   figMD3block, axs = plt.subplots(1, 2, constrained_layout=True)
         #   axs[0].plot(np.array(MDrates[2001,:,0]).reshape(200,1), color='forestgreen', alpha=0.1)
         #   axs[0].plot(np.array(MDrates[2035,:,0]).reshape(200,1), color='forestgreen', alpha=0.5)
         #   axs[0].plot(np.array(MDrates[2050,:,0]).reshape(200,1), color='forestgreen')
         #   axs[1].plot(np.array(MDrates[2002,:,1]).reshape(200,1), color='lightgreen', alpha=0.1)
         #   axs[1].plot(np.array(MDrates[2036,:,1]).reshape(200,1), color='lightgreen', alpha=0.5)
         #   axs[1].plot(np.array(MDrates[2051,:,1]).reshape(200,1), color='lightgreen')
         #   axs[0].set_ylim(0,1)
         #   axs[1].set_ylim(0,1)
         #   figMD3block.savefig('results/fig_thlamus_transition_3rdblock_{}.pdf'.format(time.strftime("%Y%m%d-%H%M%S")),
         #           dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')

         #   figMD1block, axs = plt.subplots(1, 2, constrained_layout=True)
         #   axs[0].plot(np.array(MDrates[12,:,0]).reshape(200,1), color='forestgreen', alpha=0.1)
         #   axs[0].plot(np.array(MDrates[26,:,0]).reshape(200,1), color='forestgreen', alpha=0.5)
         #   axs[0].plot(np.array(MDrates[50,:,0]).reshape(200,1), color='forestgreen')
         #   axs[1].plot(np.array(MDrates[13,:,1]).reshape(200,1), color='lightgreen', alpha=0.1)
         #   axs[1].plot(np.array(MDrates[27,:,1]).reshape(200,1), color='lightgreen', alpha=0.5)
         #   axs[1].plot(np.array(MDrates[51,:,1]).reshape(200,1), color='lightgreen')
         #   axs[0].set_ylim(0,1)
         #   axs[1].set_ylim(0,1)
         #   figMD1block.savefig('results/fig_thlamus_transition_1stblock_{}.pdf'.format(time.strftime("%Y%m%d-%H%M%S")),
         #           dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')
            

            self.fig3.savefig('results/fig_weights_{}.png'.format(time.strftime("%Y%m%d-%H%M%S")),
                    dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')
            self.figOuts.savefig('results/fig_behavior_{}.png'.format(time.strftime("%Y%m%d-%H%M%S")),
                   dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')
            self.figRates.savefig('results/fig_rates_{}.png'.format(time.strftime("%Y%m%d-%H%M%S")),
                    dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')

        ## MDeffect and MDCueOff
        #MSE,_,_ = self.do_test(20,self.MDeffect,True,False,
        #                        self.get_cue_list(),None,2)

        #return np.mean(MSE)

    def taskSwitch2(self,Nblock):
        if self.plotFigs:
            self.fig = plt.figure(figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth*1.5),
                                facecolor='w')
        task1Cues = self.get_cue_list(0)
        task2Cues = self.get_cue_list(1)
        self.do_test(Nblock,self.MDeffect,True,False,
                    task1Cues,task1Cues[0],0,train=True)
        self.do_test(Nblock,self.MDeffect,False,False,
                    task2Cues,task2Cues[0],1,train=True)
        
        if self.plotFigs:
            self.fig.tight_layout()
            self.fig.savefig('results/fig_plasticPFC2Out_{}.png'.format(time.strftime("%Y%m%d-%H%M%S")),
                        dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')

    def taskSwitch3(self,Nblock,MDoff=True):
        if self.plotFigs:
            self.fig = plt.figure(figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth*1.5),
                                facecolor='w')
        task1Cues = self.get_cue_list(0)
        task2Cues = self.get_cue_list(1)
        # after learning, during testing the learning rate is low, just performance tuning
        self.learning_rate /= 100.
        MSEs1,_,wOuts1 = self.do_test(Nblock,self.MDeffect,False,False,\
                            task1Cues,task1Cues[0],0,train=True)
        if MDoff:
            self.learning_rate *= 100.
            MSEs2,_,wOuts2 = self.do_test(Nblock,self.MDeffect,MDoff,False,\
                                task2Cues,task2Cues[0],1,train=True)
            self.learning_rate /= 100.
        else:
            MSEs2,_,wOuts2 = self.do_test(Nblock,self.MDeffect,MDoff,False,\
                                task2Cues,task2Cues[0],1,train=True)
        MSEs3,_,wOuts3 = self.do_test(Nblock,self.MDeffect,False,False,\
                            task1Cues,task1Cues[0],2,train=True)
        self.learning_rate *= 100.
        
        if self.plotFigs:
            self.fig.tight_layout()
            self.fig.savefig('results/fig_plasticPFC2Out_{}.png'.format(time.strftime("%Y%m%d-%H%M%S")),
                        dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')

            # plot the evolution of mean squared errors over each block
            fig2 = plt.figure(figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth),
                                facecolor='w')
            ax2 = fig2.add_subplot(111)
            ax2.plot(MSEs1,'-,r')
            #ax2.plot(MSEs2,'-,b')
            ax2.plot(MSEs3,'-,g')

            # plot the evolution of different sets of weights
            fig2 = plt.figure(figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth),
                                facecolor='w')
            ax2 = fig2.add_subplot(231)
            ax2.plot(np.reshape(wOuts1[:,:,:self.Nsub*2],(Nblock,-1)))
            ax2.set_ylim((-0.1,0.1))
            ax2 = fig2.add_subplot(232)
            ax2.plot(np.reshape(wOuts2[:,:,:self.Nsub*2],(Nblock,-1)))
            ax2.set_ylim((-0.1,0.1))
            ax2 = fig2.add_subplot(233)
            ax2.plot(np.reshape(wOuts3[:,:,:self.Nsub*2],(Nblock,-1)))
            ax2.set_ylim((-0.1,0.1))
            ax2 = fig2.add_subplot(234)
            ax2.plot(np.reshape(wOuts1[:,:,self.Nsub*2:self.Nsub*4],(Nblock,-1)))
            ax2.set_ylim((-0.1,0.1))
            ax2 = fig2.add_subplot(235)
            ax2.plot(np.reshape(wOuts2[:,:,self.Nsub*2:self.Nsub*4],(Nblock,-1)))
            ax2.set_ylim((-0.1,0.1))
            ax2 = fig2.add_subplot(236)
            ax2.plot(np.reshape(wOuts3[:,:,self.Nsub*2:self.Nsub*4],(Nblock,-1)))
            ax2.set_ylim((-0.1,0.1))

    def test(self,Ntest):
        if self.plotFigs:
            self.fig = plt.figure(figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth*1.5),
                                facecolor='w')
            # self.fig2 = plt.figure(figsize=(pltu.columnwidth,pltu.columnwidth),
            #                     facecolor='w')
        cues = self.get_cue_list()
        
        # after learning, during testing the learning rate is low, just performance tuning
        self.learning_rate /= 100.
        
        self.do_test(Ntest,self.MDeffect,False,False,cues,(0,0),0)
        if self.plotFigs:
            axs = self.fig.get_axes() #self.fig2.add_subplot(111)
            ax = axs[0]
            # plot mean activity of each neuron for this taski+cuei
            #  further binning 10 neurons into 1
            ax.plot(np.mean(np.reshape(\
                                np.mean(self.meanAct[0,:,:],axis=0),\
                            (self.Nneur//10,10)),axis=1),',-r')
        if self.saveData:
            self.fileDict['meanAct0'] = self.meanAct[0,:,:]
        self.do_test(Ntest,self.MDeffect,False,False,cues,(0,1),1)
        if self.plotFigs:
            # plot mean activity of each neuron for this taski+cuei
            ax.plot(np.mean(np.reshape(\
                                np.mean(self.meanAct[1,:,:],axis=0),\
                            (self.Nneur//10,10)),axis=1),',-b')
            ax.set_xlabel('neuron #')
            ax.set_ylabel('mean rate')
        if self.saveData:
            self.fileDict['meanAct1'] = self.meanAct[1,:,:]

        if self.xorTask:
            self.do_test(Ntest,self.MDeffect,True,False,cues,(0,2),2)
            self.do_test(Ntest,self.MDeffect,True,False,cues,(0,3),3)
        else:
            self.do_test(Ntest,self.MDeffect,True,False,cues,(1,0),2)
            self.do_test(Ntest,self.MDeffect,True,False,cues,(1,1),3)
            #self.learning_rate *= 100
            ## MDeffect and MDCueOff
            #self.do_test(Ntest,self.MDeffect,True,False,cues,self.cuePlot,2)
            ## MDeffect and MDDelayOff
            ## network doesn't (shouldn't) learn this by construction.
            #self.do_test(Ntest,self.MDeffect,False,True,cues,self.cuePlot,3)
            ## back to old learning rate
            #self.learning_rate *= 100.
        
        if self.plotFigs:
            self.fig.tight_layout()
            self.fig.savefig('results/fig_plasticPFC2Out_{}.png'.format(time.strftime("%Y%m%d-%H%M%S")),
                        dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')
            # self.fig2.tight_layout()

    def load(self,filename):
        d = shelve.open(filename) # open
        if self.outExternal:
            self.wOut = d['wOut']
        else:
            self.Jrec[-self.Nout:,:] = d['JrecOut']
        if self.dirConn:
            self.wDir = d['wDir']
        d.close()
        return None

    def save(self):
        if self.outExternal:
            self.fileDict['wOut'] = self.wOut
        else:
            self.fileDict['JrecOut'] = self.Jrec[-self.Nout:,:]
        if self.dirConn:
            self.fileDict['wDir'] = self.wDir

if __name__ == "__main__":
    #PFC_G = 1.6                    # if not positiveRates
    PFC_G = 1.2
    PFC_G_off = 1.5
    learning_rate = 5e-6
    learning_cycles_per_task = 1000
    Ntest = 20
    Nblock = 70
    noiseSD = 1e-3
    tauError = 0.001
    reLoadWeights = False
    saveData = True #not reLoadWeights
    plotFigs = True#not saveData
    pfcmd = PFCMD(PFC_G,PFC_G_off,learning_rate,
                    noiseSD,tauError,plotFigs=plotFigs,saveData=saveData)
    if not reLoadWeights:
        t = time.perf_counter()
        pfcmd.train(learning_cycles_per_task)
        print('training_time', (time.perf_counter() - t)/60, ' minutes')

        if saveData:
            pfcmd.save()
        # save weights right after training,
        #  since test() keeps training on during MD off etc.
        # pfcmd.test(Ntest) # Ali turned test off for now.
        print('total_time', (time.perf_counter() - t)/60, ' minutes')
    else:
        pfcmd.load(filename)
        # all 4cues in a block
        pfcmd.test(Ntest)
        
        #pfcmd.taskSwitch2(Nblock)
        
        # task switch
        #pfcmd.taskSwitch3(Nblock,MDoff=True)
        
        # control experiment: task switch without turning MD off
        # also has 2 cues in a block, instead of 4 as in test()
        #pfcmd.taskSwitch3(Nblock,MDoff=False)
#    print(len(PFCrates),len(PFCrates[0]))
    figs = list(map(plt.figure, plt.get_fignums()))
    current_sizes = [(fi.canvas.height(), fi.canvas.width()) for fi in figs] #list of tuples height, width
    from data_generator import move_figure
    # move_figure(figs[0],col=4, position='bottom')
    # move_figure(figs[1],col=2, position='top')
    # move_figure(figs[2],col=0, position='bottom')
    # move_figure(figs[3],col=4, position='top')
    # move_figure(figs[4],col=3, position='bottom')
    # move_figure(figs[6],col=4, position='top')

    if pfcmd.saveData:
        pfcmd.fileDict.close()
    
    plt.show()
    
    # plt.close('all')

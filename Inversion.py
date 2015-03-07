#!/usr/bin/env python

import desdb
import numpy as np
import esutil
import pyfits
import sys
import healpy as hp

#from mpi4py import MPI
#import mpifunctions
#import DBfunctions

import numpy as np
import scipy.spatial
import scipy.interpolate
from sklearn.neighbors import NearestNeighbors
import numpy.lib.recfunctions as recfunctions

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import testInference_galmag as huff


def MagNoise(m, z=30):
    e = np.power( np.power(10.0, (z-m)/2.5 ), -0.5) * np.random.randn(len(m)) * 3
    b = (np.pi/2 + np.arctan(m-z)) / (np.pi * 5)
    m = m + e + b
    return m

def RandomType(n, min, max, kind, extra=None):
    if kind=='uniform':
        return np.random.uniform(min, max, n)
    elif kind=='gaussian':
        return min + np.random.randn(n) * 4
    elif kind=='beta':
        if extra==None:
            extra = [5,3]
        return min + np.random.beta(extra[0],extra[1], n) * (max-min)
    elif kind=='power':
        if extra==None:
            extra = 2
        return min + np.random.power(extra, n) * (max-min)



def GetBalrogTruth(n, min, max, kind, truthkey, extra=None):
    balrog_input = RandomType(n, min, max, kind, extra=extra)
    truth = np.zeros(n, dtype=[(truthkey,np.float64)])
    truth[truthkey] = balrog_input
    return truth

def GetBalrogObserved(a, falloff, wfalloff, simkey, truthkey):
    all = np.copy(a)
    detprob = np.exp( -(all[truthkey] - 16)/2.5)
    detr = np.random.random(len(all))
    keep = (detr < detprob)
    detected = all[keep]

    m = MagNoise(detected[truthkey])
    detected = recfunctions.append_fields(detected, simkey, m)
    return detected

def GetDESTruth(n, min, max, kind, truthkey, extra=None):
    dist = RandomType(n, min, max, kind, extra=extra)
    des = np.zeros(n, dtype=[(truthkey,np.float64)])
    des[truthkey] = dist
    return des


class BalrogInference(object):

    def __init__(self, config, data, truth):
        self.TruthColumns = config['truth_columns']
        self.TruthBins = config['truth_bins']

        self.ObsColumns = config['obs_columns']
        self.ObsBins = config['obs_bins']
        
        self.data = data
        self.truth = truth
        self.TransferMatrix = self.Data2TransferMatrix()


    def Data2TransferMatrix(self):
        SingleDimBinIndex = np.zeros( (len(self.data),len(self.TruthColumns)), dtype=np.int64 )
        DimFactor = np.ones( len(self.TruthColumns), dtype=np.int64)
        outside = np.zeros(len(self.data), dtype=np.bool_)
        NTruthBins = 1
        for i in range(len(self.TruthColumns)):
            neg = -(i+1)
            NTruthBins = NTruthBins * (len(self.TruthBins[neg])-1)
            SingleDimBinIndex[:,i] = np.digitize(self.data[self.TruthColumns[neg]], bins=self.TruthBins[neg]) - 1
            outside = (outside) | ( (SingleDimBinIndex[:,i]==-1) | (SingleDimBinIndex[:,i]==(len(self.TruthBins[neg])-1)) )
            if i > 0:
                DimFactor[i] = (len(self.TruthBins[neg+1]) - 1) * DimFactor[i-1]

        DimFactor = np.reshape(DimFactor, (1,len(DimFactor)) )
        BinIndex = np.sum( SingleDimBinIndex*DimFactor, axis=-1 )
        d = recfunctions.append_fields(self.data, 'BinIndex', BinIndex)[-outside]
        d = np.sort(d, order='BinIndex')
        binindex = np.arange(1, NTruthBins, 1)
        splitat = np.searchsorted(d['BinIndex'], binindex)
        BalrogByTruthIndex = np.split(d, splitat)
        self.NObserved = np.zeros( len(BalrogByTruthIndex) )

        NObsBins = 1
        SingleDimBinSize = []
        for i in range(len(self.ObsColumns)):
            NObsBins = NObsBins * (len(self.ObsBins[i]) - 1)
            SingleDimBinSize.append( np.diff(self.ObsBins[i]) )
        BinVolume = 1.0
        inc = 0
        for i in range(len(SingleDimBinSize)):
            BinVolume = BinVolume * SingleDimBinSize[i]
            BinVolume = np.expand_dims(BinVolume, axis=-1)
        BinVolume = BinVolume.flatten()
        
        TransferMatrix = np.zeros( (NObsBins,NTruthBins) )
        for i in range(len(BalrogByTruthIndex)):
            ThisTruth = np.zeros( (len(BalrogByTruthIndex[i]), len(self.ObsColumns)) )
            for j in range(len(self.ObsColumns)):
                ThisTruth[:,j] = (BalrogByTruthIndex[i][self.ObsColumns[j]])

            hist, edge = np.histogramdd(ThisTruth, bins=self.ObsBins)
            #hist = hist / (BinVolume * len(ThisTruth))
            hist = hist / len(ThisTruth)
            self.NObserved[i] = len(ThisTruth)
            hist1d = hist.flatten()
            TransferMatrix[:, i] = hist1d
           

        return TransferMatrix


    def WindowFunction(self):
        CanHist = np.zeros( (len(self.truth),len(self.TruthColumns)) )
        for i in range(len(self.TruthColumns)):
            CanHist[:, i] = self.truth[ self.TruthColumns[i] ]
        TruthHist, edge = np.histogramdd(CanHist, bins=self.TruthBins)
        self.TruthHist = TruthHist.flatten()

        return self.NObserved / self.TruthHist



if __name__=='__main__': 

    binsize = 0.2
    simkey = 'magauto'
    truthkey = 'mag'

    balrog_min = 16
    balrog_max = 28
    obs_min = 14
    obs_max = 29

    config = {'obs_columns': [simkey],
              'obs_bins': [np.arange(obs_min,obs_max,binsize)],
              'truth_columns': [truthkey],
              'truth_bins': [np.arange(balrog_min,balrog_max,binsize)]
             }

    n = 1e7
    ndes = 1e6
    falloff = 20
    wfalloff = 0.1


    #truth = GetBalrogTruth(n, balrog_min, balrog_max, 'gaussian')
    #truth = GetBalrogTruth(n, balrog_min, config['truth_bins'][0][-1], 'power', extra=2)
    truth = GetBalrogTruth(n, balrog_min, balrog_max, 'power', truthkey, extra=2)
    observed = GetBalrogObserved(truth, falloff, wfalloff, simkey, truthkey)
    #des_truth = GetDESTruth(n, balrog_min, balrog_max, 'power', extra=3)
    #des_truth = GetDESTruth(n, balrog_min, config['truth_bins'][0][-1], 'power', extra=3)
    des_truth = GetDESTruth(ndes, balrog_min, balrog_max, 'power', truthkey, extra=3)
    des_observed = GetBalrogObserved(des_truth, falloff, wfalloff, simkey, truthkey)

    '''
    binsize = 0.2
    simkey = 'data'
    truthkey = 'data_truth'
    balrog_min = 15
    balrog_max = 25.8
    obs_min = 15
    obs_max = 28
    config = {'obs_columns': [simkey],
              'obs_bins': [np.arange(obs_min,obs_max,binsize)],
              'truth_columns': [truthkey],
              'truth_bins': [np.arange(balrog_min,balrog_max,binsize)]
             }

    # Generate a simulated simulated truth catalog.
    truth = huff.generateTruthCatalog(n_obj=1000000, slope=2.500, downsample=True)
    observed = huff.applyTransferFunction(truth, blend_fraction=0)
    des_truth = huff.generateTruthCatalog(n_obj=1000000, slope=2.50, downsample=False)
    des_observed = huff.applyTransferFunction(des_truth, blend_fraction=0)
    print len(truth)==len(des_truth)
    '''

    BalrogObject = BalrogInference(config, observed, truth)
    tm = BalrogObject.TransferMatrix
    window = BalrogObject.WindowFunction()
    des_hist, edge = np.histogram(des_observed[simkey], bins=config['obs_bins'][0])
    c = (config['truth_bins'][0][1:]+config['truth_bins'][0][:-1]) / 2.0
    wind = scipy.interpolate.interp1d(c, window)

    
    reg1 = 1.0e-12
    reg2 = np.power(10.0, -8)
    #reg1 = 0
    #reg2 = 0

    prior = np.zeros(tm.shape[1])
    prior_cov_inv = reg2 * np.identity(tm.shape[1])
   
    '''
    prior_cov_inv = np.zeros( (tm.shape[1],tm.shape[1]) )
    prior_cov_inv[0,0] = -1
    prior_cov_inv[-1,-1] = 1
    for i in range(prior_cov_inv.shape[0]-1):
        prior_cov_inv[i, i+1] = 1
        prior_cov_inv[i+1, i] = -1
    
    #prior_cov_inv = np.dot(tm*window, prior_cov_inv)
    #prior_cov_inv = np.dot(np.transpose(prior_cov_inv), prior_cov_inv)
    #prior_cov_inv = np.dot(np.transpose(prior_cov_inv), prior_cov_inv) / 100000
    '''

    cc = (config['obs_bins'][0][1:]+config['obs_bins'][0][:-1]) / 2.0
    cut = (cc > c[0]) & (cc < c[-1])
    factor = np.sum(des_hist[cut] / wind(cc[cut]) )
    #prior_cov_inv = np.dot(np.transpose(prior_cov_inv), prior_cov_inv) / (factor*factor)
    prior_cov_inv = np.dot(np.transpose(prior_cov_inv), prior_cov_inv) / (factor)

    '''
    c_obs = (config['obs_bins'][0][1:]+config['obs_bins'][0][:-1]) / 2.0
    prior = des_hist / wind(c_obs)
    #prior = des_hist
    prior_cov_inv = np.linalg.inv( np.diag(prior) )
    print prior.shape, des_hist.shape
    '''

    descov = np.diag(des_hist + reg1)
    descov_inv = np.linalg.inv(descov)
    rcr_inv = np.linalg.inv( np.dot( np.transpose(tm), np.dot(descov_inv,tm) ) + prior_cov_inv)

    #rcd = np.dot(np.transpose(tm), np.dot(descov_inv, des_hist)) + np.dot(prior_cov_inv, prior)
    #des_corr = np.dot(rcr_inv, rcd)
    #des_recon = des_corr / window

    tm_inv = np.dot(rcr_inv, np.dot(np.transpose(tm), descov_inv))
    des_corr = np.dot(tm_inv, des_hist)
    des_recon = des_corr / window
    
    #cut = (c > 24)
    #print c[cut], wind(c[cut]), des_hist[cut], des_recon[cut]
    #print c, wind(c), des_hist, des_recon
  
    shot = np.diag(des_recon)
    get_back = np.dot(tm_inv, tm)
    shot_rec = np.dot(get_back, np.dot(shot, np.transpose(get_back)))
    shot_recon = np.diag(shot_rec)
    
    #leakage_recon = np.dot( tm_inv, np.dot(tm, window*des_recon) - des_hist ) / window - des_recon
    leakage_recon = np.dot( (get_back - np.identity(len(des_recon))), des_recon)
    #leakage_recon = np.dot( (get_back - np.identity(len(des_recon))), des_recon) / window
    #leakage_recon = np.dot( (get_back - np.identity(len(des_recon))), des_recon)
    #leakage_recon = np.dot( (get_back - np.diag(get_back)), des_recon)
    #leakage_recon = np.dot( np.transpose(get_back - np.diag(get_back)), des_recon)

    #err_recon = np.sqrt( shot_recon + np.power(leakage_recon,2.0) )
    err_recon = leakage_recon

    '''
    des_hist, edge = np.histogram(des_observed[simkey], bins=config['obs_bins'][0])
    des_cov = np.diag(des_hist)
    des_invcov = np.linalg.inv(des_cov)
    print des_invcov
    '''
    
    plt.figure(1)
    #tm = Balrog2TransferMatrix(observed, config['truth_columns'], config['truth_bins'], config['obs_columns'], config['obs_bins'])
    im = plt.imshow(tm, origin='lower', extent=[balrog_min,balrog_max,obs_min,obs_max], interpolation='nearest')
    plt.plot( [balrog_min,balrog_max],[balrog_min,balrog_max], color='black' )
    plt.colorbar()
 
    plt.figure(2)
    #bins = np.arange(14, 30, binsize)

    bins = config['truth_bins'][0]
    center = (bins[1:]+bins[:-1])/2.0
    bt, bins = np.histogram(truth[truthkey], bins=bins)
    bo, bins = np.histogram(observed[simkey], bins=bins)
    dt, bins = np.histogram(des_truth[truthkey], bins=bins)
    do, bins = np.histogram(des_observed[simkey], bins=bins)

    plt.plot(center, bt, color='blue', label='Sim truth')
    plt.plot(center, bo, color='green', label='Sim observed')
    plt.plot(center, dt, color='red', label='Data truth')
    plt.plot(center, do, color='magenta', label='Data observed')

    #leakage_recon = np.dot( (get_back - np.identity(len(des_recon))), dt) / window
    #leakage_recon = np.dot( (get_back - np.identity(len(des_recon))), dt)

    c = (config['truth_bins'][0][1:]+config['truth_bins'][0][:-1])/2.0
    print len(c), len(des_corr)
    #plt.plot(c, des_corr, color='black', label='DES no window correction')
    plt.scatter(c, des_recon, color='black', label='Reconstruction', s=3)
    plt.scatter(c, prior, color='cyan', label='prior', s=3)

    #plt.errorbar(c, des_recon, yerr=err_recon, color='black', fmt='o', markersize=3,label='Reconstruction')
    plt.errorbar(c, des_recon, yerr=np.sqrt(shot_recon), yerrcolor='black', fmt=None, markersize=3,label='shot noise')
    plt.errorbar(c, des_recon, yerr=leakage_recon, yerrcolor='cyan', fmt=None, markersize=3,label='leakage')

    plt.legend(loc='best', ncol=2)
    #plt.legend(loc='upper left')
    #plt.ylim([1,1e5])
    plt.ylim([1,1e7])
    #plt.xlim([16,28])
    plt.xlim([16,28])
    plt.yscale('log')
    #plt.xscale('log')
    plt.show()

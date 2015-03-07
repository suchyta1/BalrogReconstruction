#!/usr/bin/env python

import desdb
import numpy as np
import esutil
import pyfits
import healpy as hp

import sys
import os
import subprocess

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
import emcee
import time


def MagNoise(m, z=30):
    e = np.power( np.power(10.0, (z-m)/2.5 ), -0.5) * np.random.randn(len(m)) * 1
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


def AutoBin(data, columns):
    #defaultNbins = 40
    #defaultsize = 0.25
    defaultsize = 0.4
    bins = []
    for col in columns:
        min = np.amin(data[col])
        max = np.amax(data[col])
        #bins.append( np.linspace(min, max, defaultNbins) )
        bins.append( np.arange(min, max+defaultsize, defaultsize) )
    return bins



class MCMCReconstruction(object):

    def __init__(self, Balrog, Measured, logL, nWalkers=1000, reg=1.0e-10, truth=None):
        self.Balrog = Balrog
        self.Measured = Measured
        self.BuildMeasuredHist()

        self.StartGuess = np.random.rand(nWalkers, self.Balrog.TransferMatrix.shape[1]) + np.reshape(self.Balrog.TruthHistogram1D, (1, len(self.Balrog.TruthHistogram1D)))
        self.nWalkers = nWalkers
        self.nParams = self.Balrog.TransferMatrix.shape[1]

        self.reg = reg * np.identity(self.Balrog.TransferMatrix.shape[0])
        self.CovTruth = np.diag(self.Balrog.TruthHistogram1D) 
        self.CovObs = np.dot(self.Balrog.TransferMatrix, np.dot(self.Balrog.Window*self.CovTruth, np.transpose(self.Balrog.TransferMatrix)))
        self.iCovObs = np.linalg.inv(self.CovObs + self.reg)
        #CovObs = np.dot(BalrogObject.TransferMatrix, np.dot(BalrogObject.Window*CovTruth, np.transpose(BalrogObject.TransferMatrix))) + np.diag(do)
        #CovObs = np.dot(BalrogObject.TransferMatrix, np.dot(CovObs, np.transpose(BalrogObject.TransferMatrix))) #+ np.diag(do)
        #CovObs = np.dot(BalrogObject.TransferMatrix, np.dot(CovObs, np.transpose(BalrogObject.TransferMatrix))) + np.diag(do)
        self.Sampler = emcee.EnsembleSampler(self.nWalkers, self.nParams, logL, args=[self])

        if truth!=None:
            self.Truth = truth
            self.BuildTruthHist()


    def BuildMeasuredHist(self):
        Measured = np.zeros( (len(self.Measured), len(self.Balrog.MeasuredColumns)) )
        for j in range(len(self.Balrog.MeasuredColumns)):
            Measured[:,j] = self.Measured[self.Balrog.MeasuredColumns[j]]
        self.MeasuredHistogramND, edge = np.histogramdd(Measured, bins=self.Balrog.MeasuredBins)
        self.MeasuredHistogram1D = self.MeasuredHistogramND.flatten() 

    def BuildTruthHist(self):
        Truth = np.zeros( (len(self.Truth), len(self.Balrog.TruthColumns)) )
        for j in range(len(self.Balrog.TruthColumns)):
            Truth[:,j] = self.Truth[self.Balrog.TruthColumns[j]]
        self.TruthHistogramND, edge = np.histogramdd(Truth, bins=self.Balrog.TruthBins)
        self.TruthHistogram1D = self.TruthHistogramND.flatten() 
        
    def PlotMeasuredHistogram1D(self, ax, where=None, plotkwargs={}):
        ax = PlotHist(self.Balrog, ax, des=self, kind='Measured', where=where, plotkwargs=plotkwargs)
        return ax

    def PlotTruthHistogram1D(self, ax, where=None, plotkwargs={}):
        ax = PlotHist(self.Balrog, ax, des=self, kind='Truth', where=where, plotkwargs=plotkwargs)
        return ax

    def ClearChain(self):
        self.Sampler.reset()

    def BurnIn(self, n):
        self.ChainPos, self.ChainProb, self.ChainState = self.Sampler.run_mcmc(self.StartGuess, n)
        self.ClearChain()

    def Sample(self, n, pos=None):
        if pos==None:
            pos = self.ChainPos
        self.ChainPos, self.ChainProb, self.ChainState = self.Sampler.run_mcmc(pos, n)
    
    def GetReconstruction(self, chainstart=0, chainend=None):
        if chainend==None:
            chainend = self.Sampler.chain.shape[1]
        subchain = self.Sampler.chain[:, chainstart:chainend, :]
        subchain = np.transpose(subchain, (2,0,1))
        subchain = np.reshape(subchain, (subchain.shape[0], subchain.shape[1]*subchain.shape[2]))
        avg = np.average(subchain, axis=-1)
        std = np.std(subchain, axis=-1)
        return avg, std

    def DefaultReconstruct(self, burnin=1000, steps=1000):
        self.BurnIn(burnin)
        self.Sample(steps)
        self.ReconHistogram1D, self.ReconHistogramd1DErr = self.GetReconstruction()


    def PlotReconHistogram1D(self, ax, where=None, plotkwargs={}, chainstart=None, chainend=None):
        self.ReconHistogram1D, self.ReconHistogram1DErr = self.GetReconstruction(chainstart=chainstart, chainend=chainend)
        dims = []
        for i in range(len(self.Balrog.TruthColumns)):
            dims.append(len(self.Balrog.TruthBins[i])-1)
        dims = tuple(dims)
        self.ReconHistogramND = np.reshape(self.ReconHistogram1D, dims)
        self.ReconHistogramNDErr = np.reshape(self.ReconHistogram1DErr, dims)
        ax = PlotHist(self.Balrog, ax, des=self, kind='Reconstructed', where=where, plotkwargs=plotkwargs)
        return ax

    def PlotChain(self, ax, chainnum, chainstart=0, chainend=None, plotkwargs={}):
        if chainend==None:
            chainend = self.Sampler.chain.shape[1]
        for i in range(nWalkers):
            ax.plot(np.arange(steps), ReconObject.Sampler.chain[i, chainstart:chainend, chainnum], **plotkwargs)
        return ax


    def PlotAllChains(self, chainstart=0, chainend=None, tmp='tmp', out='chains.pdf', plotkwargs={}):
        if not os.path.exists(tmp):
            os.makedirs(tmp)

        files = []
        for i in range(self.nParams):
            fig = plt.figure(1)
            ax = fig.add_subplot(1,1, 1)
            self.PlotChain(ax, i, chainstart=chainstart, chainend=chainend, plotkwargs=plotkwargs)
            file = os.path.join(tmp, '%i.pdf'%i)
            plt.savefig(file)
            plt.close()
            files.append(file)

        gs = ['gs', '-dBATCH', '-dNOPAUSE', '-q', '-sDEVICE=pdfwrite', '-sOutputFile=%s'%(out)]
        for file in files:
            gs.append(file)
        cmd = subprocess.list2cmdline(gs)
        os.system(cmd)
        os.system('rm -r %s'%tmp )


class BalrogLikelihood(object):

    def __init__(self, truthcat, observedcat, truthcolumns=['mag'], truthbins=None, measuredcolumns=['mag_auto'], measuredbins=None, domatch=False, matchedon='balrog_index'):
        self.FullTruth = truthcat
        self.MatchedOn = matchedon
        if domatch:
            pass
        else:
            self.Matched = observedcat

        self.TruthColumns = truthcolumns
        self.TruthBins = truthbins
        if self.TruthBins is None:
            self.TruthBins = AutoBin(self.FullTruth, self.TruthColumns)

        self.MeasuredColumns = measuredcolumns
        self.MeasuredBins = measuredbins
        if self.MeasuredBins is None:
            self.MeasuredBins = AutoBin(self.Matched, self.MeasuredColumns)
   
        self.BuildTruthHist()
        self.BuildMeasuredHist()
        self.BuildTransferMatrix()
        self.BuildWindowFunction()


    def BuildTruthHist(self):
        Truth = np.zeros( (len(self.FullTruth), len(self.TruthColumns)) )
        for j in range(len(self.TruthColumns)):
            Truth[:,j] = self.FullTruth[self.TruthColumns[j]]
        self.TruthHistogramND, edge = np.histogramdd(Truth, bins=self.TruthBins)
        self.TruthHistogram1D = self.TruthHistogramND.flatten() 

    def BuildMeasuredHist(self):
        Measured = np.zeros( (len(self.Matched), len(self.MeasuredColumns)) )
        for j in range(len(self.MeasuredColumns)):
            Measured[:,j] = self.Matched[self.MeasuredColumns[j]]
        self.MeasuredHistogramND, edge = np.histogramdd(Measured, bins=self.MeasuredBins)
        self.MeasuredHistogram1D = self.MeasuredHistogramND.flatten() 

    def BuildTransferMatrix(self):
        SingleDimBinIndex = np.zeros( (len(self.Matched),len(self.TruthColumns)), dtype=np.int64 )
        DimFactor = np.ones( len(self.TruthColumns), dtype=np.int64)
        outside = np.zeros(len(self.Matched), dtype=np.bool_)
        NTruthBins = 1
        for i in range(len(self.TruthColumns)):
            neg = -(i+1)
            NTruthBins = NTruthBins * (len(self.TruthBins[neg])-1)
            SingleDimBinIndex[:,i] = np.digitize(self.Matched[self.TruthColumns[neg]], bins=self.TruthBins[neg]) - 1
            outside = (outside) | ( (SingleDimBinIndex[:,i]==-1) | (SingleDimBinIndex[:,i]==(len(self.TruthBins[neg])-1)) )
            if i > 0:
                DimFactor[i] = (len(self.TruthBins[neg+1]) - 1) * DimFactor[i-1]

        DimFactor = np.reshape(DimFactor, (1,len(DimFactor)) )
        BinIndex = np.sum( SingleDimBinIndex*DimFactor, axis=-1 )
        d = recfunctions.append_fields(self.Matched, 'BinIndex', BinIndex)[-outside]
        #d = recfunctions.append_fields(self.Matched, 'BinIndex', BinIndex)
        d = np.sort(d, order='BinIndex')
        binindex = np.arange(1, NTruthBins, 1)
        splitat = np.searchsorted(d['BinIndex'], binindex)
        BalrogByTruthIndex = np.split(d, splitat)
        self.NObserved = np.zeros( len(BalrogByTruthIndex) )

        NObsBins = 1
        SingleDimBinSize = []
        for i in range(len(self.MeasuredColumns)):
            NObsBins = NObsBins * (len(self.MeasuredBins[i]) - 1)
            SingleDimBinSize.append( np.diff(self.MeasuredBins[i]) )
        BinVolume = 1.0
        inc = 0
        for i in range(len(SingleDimBinSize)):
            BinVolume = BinVolume * SingleDimBinSize[i]
            BinVolume = np.expand_dims(BinVolume, axis=-1)
        BinVolume = BinVolume.flatten()
        
        self.TransferMatrix = np.zeros( (NObsBins,NTruthBins) )
        for i in range(len(BalrogByTruthIndex)):
            ThisTruth = np.zeros( (len(BalrogByTruthIndex[i]), len(self.MeasuredColumns)) )
            for j in range(len(self.MeasuredColumns)):
                ThisTruth[:,j] = (BalrogByTruthIndex[i][self.MeasuredColumns[j]])

            hist, edge = np.histogramdd(ThisTruth, bins=self.MeasuredBins)
            nhist = hist / len(ThisTruth)
            self.NObserved[i] = len(ThisTruth)
            hist1d = nhist.flatten()
            self.TransferMatrix[:, i] = hist1d
           

    def BuildWindowFunction(self):
        CanHist = np.zeros( (len(self.FullTruth),len(self.TruthColumns)) )
        for i in range(len(self.TruthColumns)):
            CanHist[:, i] = self.FullTruth[ self.TruthColumns[i] ]
        TruthHist, edge = np.histogramdd(CanHist, bins=self.TruthBins)
        self.TruthHist = TruthHist.flatten()
        self.Window = self.NObserved / self.TruthHist

    def PlotTransferMatrix(self, fig, ax, truthwhere=None, measuredwhere=None, plotkwargs={}):
        #im = ax.imshow(BalrogObject.TransferMatrix, origin='lower', extent=[BalrogObject.TruthBins[0][0],BalrogObject.TruthBins[0][-1], BalrogObject.MeasuredBins[0][0],BalrogObject.MeasuredBins[0][-1]], interpolation='nearest')
        #plt.plot( [BalrogObject.TruthBins[0][0],BalrogObject.TruthBins[0][-1]],[BalrogObject.TruthBins[0][0],BalrogObject.TruthBins[0][-1]], color='black' )
        cax = ax.imshow(BalrogObject.TransferMatrix, origin='lower', interpolation='nearest', **plotkwargs)
        cbar = fig.colorbar(cax)


    def PlotTruthHistogram1D(self, ax, where=None, plotkwargs={}):
        ax = PlotHist(self, ax, kind='Truth', where=where, plotkwargs=plotkwargs) 
        return ax

    def PlotMeasuredHistogram1D(self, ax, where=None, plotkwargs={}):
        ax = PlotHist(self, ax, kind='Measured', where=where, plotkwargs=plotkwargs)
        return ax


def PlotHist(BalrogObject, ax, kind='Truth', des=None, where=None, plotkwargs={}):
    if kind.upper() in ['TRUTH', 'RECONSTRUCTED']:
        columns = BalrogObject.TruthColumns
        bins = BalrogObject.TruthBins
        if des is None:
            h = BalrogObject.TruthHistogramND
        else:
            if kind.upper()=='TRUTH':
                h = des.TruthHistogramND
            else:
                h = des.ReconHistogramND
                hh = des.ReconHistogramNDErr
    elif kind.upper()=='MEASURED':
        columns = BalrogObject.MeasuredColumns
        bins = BalrogObject.MeasuredBins
        if des is None:
            h = BalrogObject.MeasuredHistogramND
        else:
            h = des.MeasuredHistogramND

    if where=='flat':
        hist = h.flatten() 
        c = np.arange(len(hist))
        ax.scatter(c, hist, **plotkwargs) 
    else:
        if where==None:
            where = [0]*len(columns)
            where[-1] = None
        ws = [None]*len(columns)
        for i in range(len(columns)):
            if where[i]==None:
                ws[i] = ':'
                bins = bins[i]
                xlabel = columns[i]
            else:
                ws[i] = '%i' %(where[i])
        exec "hist = h[%s]" %(', '.join(ws))
        c = (bins[1:]+bins[:-1]) / 2
        if kind.upper()!='RECONSTRUCTED':
            ax.plot(c, hist, **plotkwargs) 
        else:
            exec "histerr = hh[%s]" %(', '.join(ws))
            ax.errorbar(c, hist, histerr, **plotkwargs)

    return ax


def ObjectLogL(Truth, ReconObject):
    if np.sum(Truth < 0) > 0:
        return -np.inf
    TruthObs = np.dot(ReconObject.Balrog.TransferMatrix, ReconObject.Balrog.Window*Truth)
    delta = TruthObs - ReconObject.MeasuredHistogram1D
    logL = -0.5 * np.dot(np.transpose(delta), np.dot(ReconObject.iCovObs,delta))
    return logL


def LogL(ThisWalker, TransferMatrix, Window, Observed, iCov):
    t1 = time.time()
    neg = (ThisWalker < 0)
    if np.sum(neg) > 0:
        return -np.inf

    ThisWalkerObs = np.dot(TransferMatrix, Window*ThisWalker)
    delta = ThisWalkerObs - Observed
    np.seterr(all='raise')
    try:
        logL = -0.5 * np.dot(np.transpose(delta), np.dot(iCov,delta))
    except:
        logL = -np.inf
    np.seterr(all='warn')
    t2 = time.time()
    #print t2 - t1
    return logL

    

def LogLike(Truth, TransferMatrix, Window, Observed):
    t1 = time.time()

    if np.sum(Truth < 0) > 0:
        return -np.inf
    if np.sum(Truth)==0:
        return -np.inf
    
    logarg = np.dot(TransferMatrix, np.power(Truth*Window, 2.0)/np.sum(Truth))
    if np.sum(logarg < 0) > 0:
        return -np.inf
    detpiece = np.dot(np.transpose(Observed), np.log(logarg))

    unWindow = np.zeros(len(window))
    cut = (Window==1)
    unWindow[-cut] = (1.0-Window[-cut]) * np.log(1.0-Window[-cut])
    undetpiece = np.sum(Truth*unWindow)

    logL = detpiece + undetpiece
    #logL = detpiece

    t2 = time.time()
    #print t2-t1
    return logL


def Extrap1D(interp, xs, left, right):
    xrange = interp.x
    val = np.zeros(len(xs))
    for i in range(len(xs)):
        if xs[i] < xrange[0]:
            val[i] = left
        elif xs[i] > xrange[1]:
            val[i] = right
        else:
            val[i] = interp(xs[i])
    return val


def GetSample(kind='suchyta'):
    nbalrog = 1e6
    ndes = 1e6

    if kind=='suchyta':
        simkey = 'mag_auto'
        truthkey = 'mag'
        balrog_min = 16
        balrog_max = 28
        falloff = 20
        wfalloff = 0.1

        truth = GetBalrogTruth(nbalrog, balrog_min, balrog_max, 'power', truthkey, extra=2)
        observed = GetBalrogObserved(truth, falloff, wfalloff, simkey, truthkey)
        des_truth = GetDESTruth(ndes, balrog_min, balrog_max, 'power', truthkey, extra=3)
        des_observed = GetBalrogObserved(des_truth, falloff, wfalloff, simkey, truthkey)

    elif kind=='huff':
        # Generate a simulated simulated truth catalog.
        simkey = 'data'
        truthkey = 'data_truth'
        truth = huff.generateTruthCatalog(n_obj=nbalrog, slope=2.500, downsample=True)
        observed = huff.applyTransferFunction(truth, blend_fraction=0)
        des_truth = huff.generateTruthCatalog(n_obj=ndes, slope=2.50, downsample=False)
        des_observed = huff.applyTransferFunction(des_truth, blend_fraction=0)

    return truth, observed, des_truth, des_observed, [truthkey], [simkey]


if __name__=='__main__': 
    
    truth, observed, des_truth, des_observed, truthcolumns, measuredcolumns = GetSample(kind='suchyta')
    BalrogObject = BalrogLikelihood(truth, observed, truthcolumns=truthcolumns, measuredcolumns=measuredcolumns)

    fig = plt.figure(1)
    ax = fig.add_subplot(1,1, 1)
    BalrogObject.PlotTransferMatrix(fig, ax)


    nWalkers = 1000
    burnin = 5000
    steps = 1000
    ReconObject = MCMCReconstruction(BalrogObject, des_observed, ObjectLogL, truth=des_truth, nWalkers=nWalkers)
    ReconObject.BurnIn(burnin)
    ReconObject.Sample(steps)
    print np.average(ReconObject.Sampler.acceptance_fraction)


    fig = plt.figure(2)
    ax = fig.add_subplot(1,1, 1)
    BalrogObject.PlotTruthHistogram1D(ax=ax, plotkwargs={'label':'Balrog Truth', 'color':'Red'})
    BalrogObject.PlotMeasuredHistogram1D(ax=ax, plotkwargs={'label':'Balrog Observed', 'color':'Pink'})
    ReconObject.PlotTruthHistogram1D(ax=ax, plotkwargs={'label':'Data Truth', 'color':'Blue'})
    ReconObject.PlotMeasuredHistogram1D(ax=ax, plotkwargs={'label':'Data Observed', 'color':'LightBlue'})
    ReconObject.PlotReconHistogram1D(ax=ax, plotkwargs={'label':'Data Reconstructed', 'color':'black', 'fmt':'o', 'markersize':3})
    ax.legend(loc='best', ncol=2)
    ax.set_yscale('log')

    ReconObject.PlotAllChains(plotkwargs={'color':'black', 'linewidth':0.005})

    chains = [1, 10, -3, -2]
    fig = plt.figure(3, figsize=(16,6))
    for i in range(len(chains)):
        ax = fig.add_subplot(1,len(chains), i)
        ReconObject.PlotChain(ax, chains[i], plotkwargs={'color':'black', 'linewidth':0.005})
    fig.tight_layout()
    
    plt.show()


    '''
    nWalkers = 1000
    burnin = 5000
    steps = 1000
    w_int = scipy.interpolate.interp1d(c, BalrogObject.Window)
    Wint = Extrap1D(w_int, c, w_int(c[0]), w_int(c[-1]))
    o_int = scipy.interpolate.interp1d(cc, BalrogObject.MeasuredHistogram1D)
    Oint = Extrap1D(o_int, c, o_int(cc[0]), o_int(cc[-1]))
    start = Oint / Wint
    sampler = emcee.EnsembleSampler(nWalkers, BalrogObject.TransferMatrix.shape[1], LogL, args=[BalrogObject.TransferMatrix, BalrogObject.Window, do, iCov])
    #start = 100*np.random.rand(nWalkers, BalrogObject.TransferMatrix.shape[1]) + np.reshape(start, (1, len(start)))
    start = np.random.rand(nWalkers, BalrogObject.TransferMatrix.shape[1]) + np.reshape(BalrogObject.TruthHistogram1D, (1, len(BalrogObject.TruthHistogram1D)))
    pos, prob, state = sampler.run_mcmc(start, burnin)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, steps, rstate0=state)

    instate = np.loadtxt('chain.dat')
    print instate.shape
    pos, prob, state = sampler.run_mcmc(instate, steps)
    print pos.shape
    np.savetxt('chain.dat', pos)
    '''
#!/usr/bin/env python

import desdb
import numpy as np
import esutil
import pyfits
import sys
import healpy as hp
import numpy.lib.recfunctions as recfunctions

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mpi4py import MPI
import mpifunctions
import DBfunctions
import MCMC
import healpyfunctions



def GetAllViaTileQuery(select):
    cur = desdb.connect()
    if MPI.COMM_WORLD.Get_rank()==0:
        arr = cur.quick('SELECT unique(tilename) from balrog_%s_truth_%s' %(select['table'],select['bands'][0]), array=True)
        tiles = arr['tilename']
    else:
        tiles = None

    tiles = mpifunctions.Scatter(tiles)
    arr = []
    for tile in tiles:
        where = "where tilename='%s'"%(tile)
        truth, sim, nosim = DBfunctions.GetBalrog(select, truthwhere=where, simwhere=where)
        des = DBfunctions.GetDES(select, where=where)
        arr.append([truth, sim, nosim, des])
  
    arr = MPI.COMM_WORLD.gather(arr, root=0)
    if MPI.COMM_WORLD.Get_rank()==0:
        newarr = []
        for i in range(len(arr)):
            if len(arr[i])==0:
                continue
            
            for j in range(len(arr[i])):
                for k in range(len(arr[i][j])):
                    if len(newarr) < len(arr[i][j]):
                        newarr.append(arr[i][j][k])
                    else:
                        newarr[k] = recfunctions.stack_arrays( (newarr[k], arr[i][j][k]), usemask=False)

    else:
        newarr = [None]*4

    return newarr


def Modestify(data, byband='i'):
    modest = np.zeros(len(data), dtype=np.int32)

    galcut = (data['flags_%s'%(byband)] <=3) & -( ((data['class_star_%s'%(byband)] > 0.3) & (data['mag_auto_%s'%(byband)] < 18.0)) | ((data['spread_model_%s'%(byband)] + 3*data['spreaderr_model_%s'%(byband)]) < 0.003) | ((data['mag_psf_%s'%(byband)] > 30.0) & (data['mag_auto_%s'%(byband)] < 21.0)))
    modest[galcut] = 1

    starcut = (data['flags_%s'%(byband)] <=3) & ((data['class_star_%s'%(byband)] > 0.3) & (data['mag_auto_%s'%(byband)] < 18.0) & (data['mag_psf_%s'%(byband)] < 30.0) | (((data['spread_model_%s'%(byband)] + 3*data['spreaderr_model_%s'%(byband)]) < 0.003) & ((data['spread_model_%s'%(byband)] +3*data['spreaderr_model_%s'%(byband)]) > -0.003)))
    modest[starcut] = 3

    neither = -(galcut | starcut)
    modest[neither] = 5

    data = recfunctions.append_fields(data, 'modtype_%s'%(byband), modest)
    print len(data), np.sum(galcut), np.sum(starcut), np.sum(neither)
    return data



def StarGalaxyRecon(truth, matched, des, band):
    #BalrogObject = MCMC.BalrogLikelihood(truth, matched, truthcolumns=['objtype_%s'%(band), 'mag_%s'%(band)], truthbins=[np.arange(0.5, 5, 2.0), np.arange(17.5,27,0.25)], measuredcolumns=['modtype_%s'%(band), 'mag_auto_%s'%(band)], measuredbins=[np.arange(0.5, 7, 2.0), np.arange(17.5,27,0.25)])
    BalrogObject = MCMC.BalrogLikelihood(truth, matched, truthcolumns=['objtype_%s'%(band), 'mag_%s'%(band)], truthbins=[np.arange(0.5, 5, 2.0), np.arange(17.5,25,0.5)], measuredcolumns=['modtype_%s'%(band), 'mag_auto_%s'%(band)], measuredbins=[np.arange(0.5, 7, 2.0), np.arange(17.5,25,0.5)])
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1, 1)
    BalrogObject.PlotTransferMatrix(fig, ax)
    plt.savefig('TransferMatrixSG-%s-sva1v2.png'%(band))

    nWalkers = 1000
    #burnin = 5000
    burnin = 10000
    steps = 1000
    ReconObject = MCMC.MCMCReconstruction(BalrogObject, des, MCMC.ObjectLogL, truth=truth, nWalkers=nWalkers, reg=1.0e-10)
    ReconObject.BurnIn(burnin)
    ReconObject.Sample(steps)
    print np.average(ReconObject.Sampler.acceptance_fraction)


    fig = plt.figure(2)
    ax = fig.add_subplot(1,1, 1)
    where = [0, None]
    BalrogObject.PlotTruthHistogram1D(where=where, ax=ax, plotkwargs={'label':'BT-G', 'color':'Blue'})
    BalrogObject.PlotMeasuredHistogram1D(where=where, ax=ax, plotkwargs={'label':'BO-G', 'color':'Red'})
    ReconObject.PlotMeasuredHistogram1D(where=where, ax=ax, plotkwargs={'label':'DO-G', 'color':'Gray'})
    ReconObject.PlotReconHistogram1D(where=where, ax=ax, plotkwargs={'label':'DR-G', 'color':'black', 'fmt':'o', 'markersize':3})
    ax.legend(loc='best', ncol=2)
    #ax.set_yscale('log')
    #plt.savefig('ReconstructedHistogramsG.png')

    #fig = plt.figure(3)
    #ax = fig.add_subplot(1,1, 1)
    where = [1, None]
    BalrogObject.PlotTruthHistogram1D(where=where, ax=ax, plotkwargs={'label':'BT-S', 'color':'Blue', 'ls':'dashed'})
    BalrogObject.PlotMeasuredHistogram1D(where=where, ax=ax, plotkwargs={'label':'BO-S', 'color':'Red', 'ls':'dashed'})
    ReconObject.PlotMeasuredHistogram1D(where=where, ax=ax, plotkwargs={'label':'DO-S', 'color':'Gray', 'ls':'dashed'})
    ReconObject.PlotReconHistogram1D(where=where, ax=ax, plotkwargs={'label':'DR-S', 'color':'black', 'fmt':'*', 'markersize':3})
    ax.legend(loc='best', ncol=2)
    ax.set_yscale('log')
    plt.savefig('ReconstructedHistogramsSG-%s-sva1v2.png'%(band))


    #ReconObject.PlotAllChains(plotkwargs={'color':'black', 'linewidth':0.005})
    chains = [1, 10, -3, -2]
    fig = plt.figure(4, figsize=(16,6))
    for i in range(len(chains)):
        ax = fig.add_subplot(1,len(chains), i+1)
        ReconObject.PlotChain(ax, chains[i], plotkwargs={'color':'black', 'linewidth':0.005})
    fig.tight_layout()
    plt.savefig('chainsSG-%s-sva1v2.png'%(band))






if __name__=='__main__': 

    select = {'table': 'sva1v2',
              'des': 'sva1_coadd_objects',
              'bands': ['i'],
              'truth': ['balrog_index', 'mag', 'ra', 'dec', 'objtype'],
              'sim': ['mag_auto', 'flux_auto', 'fluxerr_auto', 'flags', 'spread_model', 'spreaderr_model', 'class_star', 'mag_psf', 'alphawin_j2000', 'deltawin_j2000']
             }

    band = select['bands'][0]
    bi = 'balrog_index_%s' %(band)
    truth, matched, nosim, des = GetAllViaTileQuery(select)
    if MPI.COMM_WORLD.Get_rank()==0:
        print len(truth), len(nosim), len(matched), len(des)

        matched = Modestify(matched, byband=band)
        des = Modestify(des, byband=band)
        StarGalaxyRecon(truth, matched, des, band)
    
    '''
    truth, matched, nosim, des = healpyfunctions.ScatterByHealPixel(truth, matched, nosim, des, band, nside=256)
    if MPI.COMM_WORLD.Get_rank()==0:
        print truth.shape, truth[0].dtype.names
        print truth[0].shape
    '''

    '''
    truth = mpifunctions.scatter(truth)
    if MPI.COMM_WORLD.Get_rank()==0:
        print '0'
        print truth
        print len(truth)
    '''

         



    """
        cut = (truth['mag_%s'%(band)] > 17) & (truth['mag_%s'%(band)] < 30)
        truth = truth[cut]

        #p = np.float64( np.power(truth['mag_%s'%(band)]-24.5, 2.0) )
        keep = np.ones(len(truth), dtype=np.bool_)
        mid = (truth['mag_%s'%(band)] > 22) & (truth['mag_%s'%(band)] < 25)
        r = np.random.rand(np.sum(mid))
        keep[mid] = (r < 0.33)
        high = (truth['mag_%s'%(band)] > 25)
        r = np.random.rand(np.sum(high))
        keep[high] = (r < 0.03)
        truth = truth[keep]
        matched = matched[ np.in1d(matched[bi], truth[bi]) ]

        '''
        p = np.float64( 1.0/np.power(truth['mag_%s'%(band)]-17.0, 2.0) + 1.0/np.power(30.0-truth['mag_%s'%(band)], 2.0) )
        p = p / np.sum(p)
        print np.sum(p), p.shape
        half = np.random.choice(len(truth), size=len(truth)/4, replace=False, p=p)
        truth = truth[half]
        matched = matched[ np.in1d(matched[bi], truth[bi]) ]
        '''
        
        truth = truth[ -np.in1d(truth[bi],nosim[bi]) ]
        matched = matched[ -np.in1d(matched[bi],nosim[bi]) ]

        matched = np.sort(matched, order=[bi])
        diff = (np.diff(matched[bi]) == 0)
        cut = np.zeros(len(matched), dtype=np.bool_)
        cut[:-1] = (diff | cut[:-1])
        cut[1:] = (diff | cut[1:])
        dup = matched[cut][bi]
        matched = matched[-cut]
        truth = truth[ -np.in1d(truth[bi],dup) ]

        '''
        cut = np.zeros(len(matched), dtype=np.bool_)
        c = matched['fluxerr_auto_%s'%(band)] > 0
        cut[c] = (matched['flux_auto_%s'%(band)][c] / matched['fluxerr_auto_%s'%(band)][c]) > 5
        matched = matched[cut]
        '''
    """

        
    """
        #BalrogObject = MCMC.BalrogLikelihood(truth, matched, truthcolumns=['mag_i'], truthbins=[np.arange(16,25,0.25)], measuredcolumns=['mag_auto_i'], measuredbins=[np.arange(16,25,0.25)])
        #BalrogObject = MCMC.BalrogLikelihood(truth, matched, truthcolumns=['mag_g'], truthbins=[np.arange(17.5,27,0.25)], measuredcolumns=['mag_auto_g'], measuredbins=[np.arange(17.5,27,0.25)])
        #BalrogObject = MCMC.BalrogLikelihood(truth, matched, truthcolumns=['mag_r'], truthbins=[np.arange(17.5,27,0.25)], measuredcolumns=['mag_auto_r'], measuredbins=[np.arange(17.5,27,0.25)])
        BalrogObject = MCMC.BalrogLikelihood(truth, matched, truthcolumns=['mag_r'], truthbins=[np.arange(17.5,27,0.25)], measuredcolumns=['mag_auto_r'], measuredbins=[np.arange(17.5,27,0.25)])
        fig = plt.figure(1)
        ax = fig.add_subplot(1,1, 1)
        BalrogObject.PlotTransferMatrix(fig, ax)
        plt.savefig('TransferMatrix.png')

        nWalkers = 1000
        burnin = 5000
        steps = 1000
        ReconObject = MCMC.MCMCReconstruction(BalrogObject, des, MCMC.ObjectLogL, truth=truth, nWalkers=nWalkers, reg=1.0e-10)
        ReconObject.BurnIn(burnin)
        ReconObject.Sample(steps)
        print np.average(ReconObject.Sampler.acceptance_fraction)


        fig = plt.figure(2)
        ax = fig.add_subplot(1,1, 1)
        BalrogObject.PlotTruthHistogram1D(ax=ax, plotkwargs={'label':'Balrog Truth', 'color':'Blue'})
        BalrogObject.PlotMeasuredHistogram1D(ax=ax, plotkwargs={'label':'Balrog Observed', 'color':'Cyan'})
        #ReconObject.PlotTruthHistogram1D(ax=ax, plotkwargs={'label':'Data Truth', 'color':'Blue'})
        ReconObject.PlotMeasuredHistogram1D(ax=ax, plotkwargs={'label':'Data Observed', 'color':'Gray'})
        ReconObject.PlotReconHistogram1D(ax=ax, plotkwargs={'label':'Data Reconstructed', 'color':'black', 'fmt':'o', 'markersize':3})
        ax.legend(loc='best', ncol=2)
        ax.set_yscale('log')
        plt.savefig('ReconstructedHistograms.png')

        #ReconObject.PlotAllChains(plotkwargs={'color':'black', 'linewidth':0.005})
        chains = [1, 10, -3, -2]
        fig = plt.figure(3, figsize=(16,6))
        for i in range(len(chains)):
            ax = fig.add_subplot(1,len(chains), i+1)
            ReconObject.PlotChain(ax, chains[i], plotkwargs={'color':'black', 'linewidth':0.005})
        fig.tight_layout()
        plt.savefig('chains.png')


        #plt.show()
    """

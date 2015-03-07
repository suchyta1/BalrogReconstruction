#!/usr/bin/env python

import desdb
import numpy as np
import esutil
import pyfits
import sys
import healpy as hp
import numpy.lib.recfunctions as recfunctions

from mpi4py import MPI
import mpifunctions
import DBfunctions
import MCMC

import numpy as np
import scipy.spatial
from sklearn.neighbors import NearestNeighbors

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


if __name__=='__main__': 

    select = {'table': 'sva1v2',
              'des': 'sva1_coadd_objects',
              'bands': ['i'],
              'truth': ['balrog_index', 'mag', 'ra', 'dec'],
              'sim': ['mag_auto']
             }

    truth, matched, nosim, des = GetAllViaTileQuery(select)
    if MPI.COMM_WORLD.Get_rank()==0:
        print len(truth), len(nosim), len(matched), len(des)

        BalrogObject = MCMC.BalrogLikelihood(truth, matched, truthcolumns=['mag_i'], truthbins=[np.arange(16,25,0.25)], measuredcolumns=['mag_auto_i'], measuredbins=[np.arange(16,25,0.25)])
        fig = plt.figure(1)
        ax = fig.add_subplot(1,1, 1)
        BalrogObject.PlotTransferMatrix(fig, ax)

        nWalkers = 1000
        burnin = 5000
        steps = 1000
        ReconObject = MCMC.MCMCReconstruction(BalrogObject, des, MCMC.ObjectLogL, truth=truth, nWalkers=nWalkers)
        ReconObject.BurnIn(burnin)
        ReconObject.Sample(steps)
        print np.average(ReconObject.Sampler.acceptance_fraction)


        fig = plt.figure(2)
        ax = fig.add_subplot(1,1, 1)
        BalrogObject.PlotTruthHistogram1D(ax=ax, plotkwargs={'label':'Balrog Truth', 'color':'Red'})
        BalrogObject.PlotMeasuredHistogram1D(ax=ax, plotkwargs={'label':'Balrog Observed', 'color':'Pink'})
        #ReconObject.PlotTruthHistogram1D(ax=ax, plotkwargs={'label':'Data Truth', 'color':'Blue'})
        ReconObject.PlotMeasuredHistogram1D(ax=ax, plotkwargs={'label':'Data Observed', 'color':'LightBlue'})
        ReconObject.PlotReconHistogram1D(ax=ax, plotkwargs={'label':'Data Reconstructed', 'color':'black', 'fmt':'o', 'markersize':3})
        ax.legend(loc='best', ncol=2)
        ax.set_yscale('log')

        #ReconObject.PlotAllChains(plotkwargs={'color':'black', 'linewidth':0.005})
        chains = [1, 10, -3, -2]
        fig = plt.figure(3, figsize=(16,6))
        for i in range(len(chains)):
            ax = fig.add_subplot(1,len(chains), i+1)
            ReconObject.PlotChain(ax, chains[i], plotkwargs={'color':'black', 'linewidth':0.005})
        fig.tight_layout()


        plt.show()

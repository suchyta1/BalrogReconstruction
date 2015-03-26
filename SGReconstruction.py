#!/usr/bin/env python

import numpy as np
import numpy.lib.recfunctions as recfunctions
from mpi4py import MPI
import os
import sys
import pyfits
import esutil
import healpy as hp
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mpifunctions
import DBfunctions
import MCMC
import healpyfunctions
import PCA



def Modestify(data, byband='i'):
    modest = np.zeros(len(data), dtype=np.int32)

    galcut = (data['flags_%s'%(byband)] <=3) & -( ((data['class_star_%s'%(byband)] > 0.3) & (data['mag_auto_%s'%(byband)] < 18.0)) | ((data['spread_model_%s'%(byband)] + 3*data['spreaderr_model_%s'%(byband)]) < 0.003) | ((data['mag_psf_%s'%(byband)] > 30.0) & (data['mag_auto_%s'%(byband)] < 21.0)))
    modest[galcut] = 1

    starcut = (data['flags_%s'%(byband)] <=3) & ((data['class_star_%s'%(byband)] > 0.3) & (data['mag_auto_%s'%(byband)] < 18.0) & (data['mag_psf_%s'%(byband)] < 30.0) | (((data['spread_model_%s'%(byband)] + 3*data['spreaderr_model_%s'%(byband)]) < 0.003) & ((data['spread_model_%s'%(byband)] +3*data['spreaderr_model_%s'%(byband)]) > -0.003)))
    modest[starcut] = 3

    neither = -(galcut | starcut)
    modest[neither] = 5

    data = recfunctions.append_fields(data, 'modtype_%s'%(byband), modest, usemask=False)
    return data



def StarGalaxyRecon(truth, matched, des, band, truthcolumns, truthbins, measuredcolumns, measuredbins, nWalkers=1000, burnin=5000, steps=1000, out='SGPlots', hpfield='hpIndex'):
    if hpfield is None:
        index = -1
    else:
        index = truth[hpfield][0]
    #index = truth[hpfield][0]

    tmfile = os.path.join(out, 'TransferMatrix-%s-%i.png' %(band,index))
    reconfile = os.path.join(out, 'ReconstructedHistogram-%s-%i.png' %(band,index))
    chainfile = os.path.join(out, 'Chains-%s-%i.png' %(band,index))
    burnfile = os.path.join(out, 'Burnin-%s-%i.png' %(band,index))



    BalrogObject = MCMC.BalrogLikelihood(truth, matched, truthcolumns=truthcolumns, truthbins=truthbins, measuredcolumns=measuredcolumns, measuredbins=measuredbins)
    #ReconObject = MCMC.MCMCReconstruction(BalrogObject, des, MCMC.ObjectLogL, truth=truth, nWalkers=nWalkers, reg=1.0e-10)
    #ReconObject = MCMC.MCMCReconstruction(BalrogObject, des, MCMC.ObjectLogL, truth=truth, nWalkers=nWalkers, reg=1.0e-10, samplelog=True)

    fig = plt.figure()
    ax = fig.add_subplot(1,1, 1)
    BalrogObject.PlotTransferMatrix(fig, ax)
    plt.savefig(tmfile)


    ReconObject = MCMC.MCMCReconstruction(BalrogObject, des, MCMC.ObjectLogThing, truth=truth, nWalkers=nWalkers, reg=1.0e-10)
    ReconObject.BurnIn(burnin)

    c = len(truthbins[-1]) - 2
    s = c * 2
    chains = [c-6, c-3, c, s-6, s-3, s]
    
    '''
    #chains = [1, 10, -2]
    fig = plt.figure(figsize=(16,6))
    for i in range(len(chains)):
        ax = fig.add_subplot(1,len(chains), i+1)
        ReconObject.PlotChain(ax, chains[i], plotkwargs={'color':'black', 'linewidth':0.03}, twocolor=True)
    fig.tight_layout()
    plt.savefig(burnfile)
    plt.close(fig)
    '''

    ReconObject.Sample(steps)
    print index, np.average(ReconObject.Sampler.acceptance_fraction)
    acceptance = np.average(ReconObject.Sampler.acceptance_fraction)



    fig = plt.figure(figsize=(16,6))
    for i in range(len(chains)):
        ax = fig.add_subplot(1,len(chains), i+1)
        ReconObject.PlotChain(ax, chains[i], plotkwargs={'color':'black', 'linewidth':0.03}, twocolor=True)
    fig.tight_layout()
    plt.savefig(chainfile)
    plt.close(fig)

    where = [0, None]
    galaxy_balrog_obs_center, galaxy_balrog_obs = BalrogObject.ReturnHistogram(kind='Measured', where=where)
    galaxy_balrog_truth_center, galaxy_balrog_truth = BalrogObject.ReturnHistogram(kind='Truth', where=where)
    galaxy_recon_obs_center, galaxy_recon_obs = ReconObject.ReturnHistogram(kind='Measured', where=where)
    galaxy_recon_truth_center, galaxy_recon_truth, galaxy_recon_trutherr = ReconObject.ReturnHistogram(kind='RECONSTRUCTED', where=where)

    where = [1, None]
    star_balrog_obs_center, star_balrog_obs = BalrogObject.ReturnHistogram(kind='Measured', where=where)
    star_balrog_truth_center, star_balrog_truth = BalrogObject.ReturnHistogram(kind='Truth', where=where)
    star_recon_obs_center, star_recon_obs = ReconObject.ReturnHistogram(kind='Measured', where=where)
    star_recon_truth_center, star_recon_truth, star_recon_trutherr = ReconObject.ReturnHistogram(kind='RECONSTRUCTED', where=where)

    where = [2, None]
    neither_balrog_obs_center, neither_balrog_obs = BalrogObject.ReturnHistogram(kind='Measured', where=where)

    balrog_obs = [galaxy_balrog_obs, star_balrog_obs, neither_balrog_obs, galaxy_balrog_obs_center]
    balrog_truth = [galaxy_balrog_truth, star_balrog_truth, galaxy_balrog_truth_center]
    recon_obs = [galaxy_recon_obs, star_recon_obs, galaxy_recon_obs_center]
    recon_truth = [galaxy_recon_truth, star_recon_truth, galaxy_recon_trutherr, star_recon_trutherr, galaxy_recon_truth_center]

    return balrog_obs, balrog_truth, recon_obs, recon_truth, index, len(truth), acceptance


'''
    fig = plt.figure()
    ax = fig.add_subplot(1,1, 1)
    BalrogObject.PlotTransferMatrix(fig, ax)
    plt.savefig(tmfile)
    plt.close(fig)

    ReconObject = MCMC.MCMCReconstruction(BalrogObject, des, MCMC.ObjectLogL, truth=truth, nWalkers=nWalkers, reg=1.0e-10)
    ReconObject.BurnIn(burnin)
    ReconObject.Sample(steps)
    #print np.average(ReconObject.Sampler.acceptance_fraction)

    fig = plt.figure()
    ax = fig.add_subplot(1,1, 1)
    where = [0, None]
    BalrogObject.PlotTruthHistogram1D(where=where, ax=ax, plotkwargs={'label':'BT-G', 'color':'Blue'})
    BalrogObject.PlotMeasuredHistogram1D(where=where, ax=ax, plotkwargs={'label':'BO-G', 'color':'Red'})
    ReconObject.PlotMeasuredHistogram1D(where=where, ax=ax, plotkwargs={'label':'DO-G', 'color':'Gray'})
    ReconObject.PlotReconHistogram1D(where=where, ax=ax, plotkwargs={'label':'DR-G', 'color':'black', 'fmt':'.'})
    ax.legend(loc='best', ncol=2)

    where = [1, None]
    BalrogObject.PlotTruthHistogram1D(where=where, ax=ax, plotkwargs={'label':'BT-S', 'color':'Blue', 'ls':'dashed'})
    BalrogObject.PlotMeasuredHistogram1D(where=where, ax=ax, plotkwargs={'label':'BO-S', 'color':'Red', 'ls':'dashed'})
    ReconObject.PlotMeasuredHistogram1D(where=where, ax=ax, plotkwargs={'label':'DO-S', 'color':'Gray', 'ls':'dashed'})
    ReconObject.PlotReconHistogram1D(where=where, ax=ax, plotkwargs={'label':'DR-S', 'color':'black', 'fmt':'*'})
    centers, star, starerr = MCMC.ReturnHistogram(ReconObject, kind='RECONSTRUCTED', where=where)

    ax.legend(loc='best', ncol=2)
    ax.set_yscale('log')
    ax.set_ylim( [0.1, 10000] )
    plt.savefig(reconfile)
    plt.close(fig)

    #ReconObject.PlotAllChains(plotkwargs={'color':'black', 'linewidth':0.005})
    chains = [1, 10, -2]
    fig = plt.figure(figsize=(16,6))
    for i in range(len(chains)):
        ax = fig.add_subplot(1,len(chains), i+1)
        ReconObject.PlotChain(ax, chains[i], plotkwargs={'color':'black', 'linewidth':0.005})
    fig.tight_layout()
    plt.savefig(chainfile)
    plt.close(fig)

    #return centers, galaxy, galaxyerr, star, starerr, index
'''



def ReconImageLine(hpIndex, num, arr):
    i = np.insert(arr, 0, num)
    i = np.insert(i, 0, hpIndex)
    return i


def GatherImages(images):
    for i in range(len(images)):
        images[i] = np.array(images[i])
        images[i] = mpifunctions.Gather(images[i])

    return images


def SGRecon2Map(reconfile, version, magmin=22.5, magmax=24.5, nside=256, nest=False):
    hdus = pyfits.open(reconfile)
    #coords = hdus[-2].data[0,:]
    
    tab = hdus[-1].data
    hpIndex = tab['hpIndex']
    norm = 1.0/tab['numTruth'] * np.power(10.0,5.0)/np.power(np.power(10.0, 4.0), 2.0) * 1.0/np.power(0.27, 2.0) * np.power(3600.0, 2.0)

    coords = hdus[-2].data
    cut = np.where( (coords > magmin) & (coords < magmax) )
    start = np.amin(cut)
    end = np.amax(cut)

    galaxy = np.sum( hdus[1].data[:, start:end], axis=-1) * norm
    star = np.sum( hdus[2].data[:, start:end], axis=-1) * norm
    MakeMaps(galaxy, star, hpIndex, nside, version, nest=nest)


def FractionalOffsetMap(arr, hpIndex, nside):
    npix = hp.nside2npix(nside)
    map = np.zeros(npix) + hp.UNSEEN

    avg = np.average(arr)
    map[hpIndex] = (arr - avg)/avg
    mapmin = np.percentile(map[hpIndex], 5)
    mapmax = np.percentile(map[hpIndex], 95)
    larger = np.amax( np.fabs([mapmin, mapmax]) )
    mapmin = -larger
    mapmax = larger
    return map, mapmin, mapmax


def MakeMaps(galaxy, star, hpIndex, nside, version, nest=False):
    GalMap, galmin, galmax = FractionalOffsetMap(galaxy, hpIndex, nside)
    StarMap, starmin, starmax = FractionalOffsetMap(star, hpIndex, nside)
    healpyfunctions.VisualizeHealPixMap(GalMap, title=os.path.join(version, 'Galaxy-map-%s'%(version)), nest=nest, vmin=galmin, vmax=galmax, background='gray')
    healpyfunctions.VisualizeHealPixMap(StarMap, title=os.path.join(version, 'Star-map-%s'%(version)), nest=nest, vmin=starmin, vmax=starmax, background='gray')



def ByHealpixelAndJack(arrs, map, band, mcmc, balrogcoords=['ra','dec'], descoords=['alphawin_j2000','deltawin_j2000'], minnum=None, method='sort'):
    hpfield = map['hpfield']
    jfield = mcmc['jackfield']
    if minnum is None:
        minnum = mcmc['jacknife']

    for i in range(len(arrs)):
        if map['nside'] is None:
            hpInd = np.array( [-1]*len(arrs[i]) )
            #hpInd = np.zeros( len(arrs[i]), dtype=np.int64 )
        else:
            if i < (len(arrs)-1):
                ra = '%s_%s' %(balrogcoords[0], band)
                dec = '%s_%s' %(balrogcoords[1], band)
            else:
                ra = '%s_%s' %(descoords[0], band)
                dec = '%s_%s' %(descoords[1], band)
            
            hpInd = healpyfunctions.RaDec2Healpix(arrs[i][ra], arrs[i][dec], map['nside'], nest=map['nest'])
            
        if i==0:
            uInds = np.unique(hpInd)
        else:
            uInds = np.intersect1d(uInds, hpInd)

        arrs[i] = recfunctions.append_fields(arrs[i], hpfield, hpInd, usemask=False)

    for i in range(len(arrs)):
        cut = np.in1d(arrs[i][hpfield], uInds)
        arrs[i] = arrs[i][cut]

    rInds = np.zeros(0, dtype=np.int64)

    if method=='loop':
        for i in range(len(arrs)):
            if i in [0, len(arrs)-1]:
                
                count = np.zeros(len(arrs[i]))
                jack = np.zeros(len(arrs[i]))
                cc = np.unique(arrs[i][hpfield])

                for j in range(len(cc)):
                    cut = (arrs[i][hpfield]==cc[j])
                    cnt = np.sum(cut)
                    count[cut] = np.sum(cnt)

                    if (i==0) or (mcmc['jackdes']):
                        ipart = cnt / mcmc['jacknife']
                        rpart = cnt % mcmc['jacknife']
                        ii = np.append( np.repeat( np.arange(0, mcmc['jacknife'], 1), ipart ), np.arange(0, rpart, 1) )
                        np.random.shuffle(ii)
                        jack[cut] = ii
                        
                    
                cut = (count > minnum)
                uInds = np.intersect1d(uInds, arrs[i][cut][hpfield])
                pindex = arrs[i][hpfield]*mcmc['jacknife'] + jack
                arrs[i] = recfunctions.append_fields(arrs[i], 'pindex', pindex, usemask=False)


            elif i < len(arrs)-1:
                pindex = np.zeros(len(arrs[i]))
                count = np.zeros(len(arrs[i]))
                p = np.unique(arrs[0]['pindex'])
                #for pp in range(len(p)):
                for pp in p:
                    where = (arrs[0]['pindex']==pp)
                    b = arrs[0]['balrog_index_%s'%(band)][where]
                    cut = np.in1d(arrs[i]['balrog_index_%s'%(band)], b)
                    pindex[cut] = pp
                    count[cut] = np.sum(cut)

                arrs[i] = recfunctions.append_fields(arrs[i], 'pindex', pindex, usemask=False)
                cut = (count > minnum)
                uInds = np.intersect1d(uInds, arrs[i][cut][hpfield])

    elif method=='sort':
        for i in range(len(arrs)):
            if i == (len(arrs)-1):
                if mcmc['jackdes']:
                    jby = np.arange(len(arrs[i]))
                else:
                    jby = np.zeros(len(arrs[i]), dtype=np.int64)
            else:
                jby = arrs[i]['balrog_index_%s'%(band)]
            jack = np.mod(jby, mcmc['jacknife'])
            pindex = arrs[i][hpfield]*mcmc['jacknife'] + jack
            arrs[i] = recfunctions.append_fields(arrs[i], 'pindex', pindex, usemask=False)

            arrs[i] = np.sort(arrs[i], order='pindex')
            if np.sum(uInds==-1)==0:
                possible = np.repeat(uInds, mcmc['jacknife']) + np.repeat( [np.arange(mcmc['jacknife'])], len(uInds), axis=0 ).flatten()
            else:
                possible = np.arange(-mcmc['jacknife'], 0, 1)

            where = np.searchsorted(arrs[i]['pindex'], possible)
            found = np.append(where, len(arrs[i]))
            number = np.diff(found)
            cut = (number < minnum)
            r = np.unique(arrs[i][cut][hpfield])
            rInds = np.union1d(rInds, r)

    
        for i in range(len(arrs)):
            cut = -np.in1d(arrs[i][hpfield], rInds)
            arrs[i] = arrs[i][cut]

    
    for i in range(len(arrs)):
        cut = np.in1d(arrs[i][hpfield], uInds)
        arrs[i] = arrs[i][cut]
        arrs[i] = recfunctions.append_fields(arrs[i], jfield, np.mod(arrs[i]['pindex'],mcmc['jacknife']), usemask=False)
        #print len(arrs[i])

    if mcmc['jacknife'] == 1:
        for i in range(len(arrs)):
            arrs[i][jfield] = -1

   

    a = []
    for i in range(len(arrs)):
        if method=='sort':
            #arrs[i] = np.sort(arrs[i], order='pindex')
            splitby = np.unique(arrs[i]['pindex'])
            if len(splitby) > 1:
                splitat = np.searchsorted(arrs[i]['pindex'], splitby[1:])
                a.append( np.split(arrs[i], splitat) )
            else:
                a.append( [arrs[i]] )

        elif method=='loop':
            a.append( [] )
            pind = np.unique(arrs[i]['pindex'])
            for j in range(len(pind)):
                cut = (arrs[i]['pindex']==pind[j])
                a[i].append( arrs[i][cut] )


        if (i==(len(arrs)-1)) and (not mcmc['jackdes']):
            a[i] = list( np.repeat(a[i], mcmc['jacknife'], axis=0) )

        if (mcmc['jacknife'] > 1):
            hps = np.unique(arrs[i][hpfield])
            for h in hps:
                cut = (arrs[i][hpfield]==h)
                d = arrs[i][cut]
                d[jfield] = -1
                a[i].append(d)

        if map['nside'] is not None:
            js = np.unique(arrs[i][jfield])
            for j in js:
                cut = (arrs[i][jfield]==j)
                d = arrs[i][cut]
                d[hpfield] = -1
                a[i].append(d)

        if map['nside'] is not None:
            d = np.copy(arrs[i])
            d[hpfield] = -1
            d[jfield] = -1
            a[i].append(d)

    print len(a[0])
    return a





def DivideWork(truth, matched, nosim, des, band, map, mcmc, balrogcoords=['ra','dec'], descoords=['alphawin_j2000','deltawin_j2000'], minnum=None):
    if MPI.COMM_WORLD.Get_rank()==0:
        truth, matched, nosim, des = ByHealpixelAndJack([truth, matched, nosim, des], map, band, mcmc, balrogcoords=balrogcoords, descoords=descoords, minnum=minnum, method='sort')
       
    '''
    if MPI.COMM_WORLD.Get_rank()==0:
        for i in range(len(truth)):
            print len(truth[i]), len(matched[i]), len(nosim[i]), len(des[i])
    '''

    truth, matched, nosim, des = mpifunctions.Scatter(truth, matched, nosim, des)
    '''
    truth = mpifunctions.Scatter(truth)
    matched = mpifunctions.Scatter(matched)
    nosim = mpifunctions.Scatter(nosim)
    des = mpifunctions.Scatter(des)
    '''

    return truth, matched, nosim, des


def SGClassify(truth, matched, nosim, des):
    if MPI.COMM_WORLD.Get_rank()==0:
        print len(truth), len(nosim), len(matched), len(des)
        matched = Modestify(matched, byband=band)
        nosim = Modestify(nosim, byband='det')
        des = Modestify(des, byband=band)
    return truth, matched, nosim, des


def DoSGMcmc(truth, matched, nosim, des, mcmc, band, map):
    size = len(truth)
    images = [ [], [], [], [] ]
    hpIndex = np.empty(size)
    jIndex = np.empty(size)
    sizes = np.empty(size)
    acceptance = np.empty(size)

    hpfield = map['hpfield']
    jfield = mcmc['jackfield']
    PCAtype = mcmc['PCAon']

    if PCAtype is not None:
        evectors, evalues, master = PCALikelihood(map, mcmc, truth, matched, PCAtype, jfield, hpfield, band, doplot=True)
    else:
        evectors = None
        evalues = None
        master = None

    for i in range(size):
        if len(truth) > 0:
            #o, t, d, r, h, j, s, a = SGR2(truth[i], matched[i], des[i], band, mcmc['truthcolumns'], mcmc['truthbins'], mcmc['measuredcolumns'], mcmc['measuredbins'], nWalkers=mcmc['nWalkers'], burnin=mcmc['burnin'], steps=mcmc['steps'], out=mcmc['out'], hpfield=hpfield, jfield=jfield, threshold=mcmc['threshold'])
            o, t, d, r, h, j, s, a = SGR2(truth[i], matched[i], des[i], band, hpfield=hpfield, evectors=evectors, evalues=evalues, master=master, **mcmc)
            images[0].append(o)
            images[1].append(t)
            images[2].append(d)
            images[3].append(r)
            hpIndex[i] = h
            jIndex[i] = j
            sizes[i] = s
            acceptance[i] = a
    return images, hpIndex, jIndex, sizes, acceptance


def PCALikelihood(map, mcmc, truth, matched, PCAtype, jfield, hfield, band, doplot=False):
    size = len(truth)
    likes = []
    master = []
    for i in range(size):
        jin = truth[i][jfield][0]
        hin = truth[i][hfield][0]

        if (PCAtype=='jacknife' and jin!=-1) or (PCAtype=='healpix' and jin==-1):
            like = InitialLikelihood(truth[i], matched[i], mcmc['truthcolumns'], mcmc['truthbins'], mcmc['measuredcolumns'], mcmc['measuredbins'], threshold=mcmc['threshold'])
            if (jin==-1) & (hin==-1):
                master = np.copy(like)
            else:
                likes.append(like)

    likes = mpifunctions.Gather(likes)
    master = mpifunctions.Gather(master)
    if MPI.COMM_WORLD.Get_rank()==0:
        evectors, evalues = PCA.likelihoodPCA(likelihood=likes, likilhood_master=master, doplot=doplot, band=band, extent=None, residual=mcmc['residual'])
    else:
        evectors = None
        evalues = None
    evectors, evalues = mpifunctions.Broadcast(evectors, evalues)
    return evectors, evalues, master


def InitialLikelihood(truth, matched, truthcolumns, truthbins, measuredcolumns, measuredbins, threshold=0):
    BalrogObject = MCMC.BalrogLikelihood(truth, matched, truthcolumns=truthcolumns, truthbins=truthbins, measuredcolumns=measuredcolumns, measuredbins=measuredbins, threshold=threshold, n_component=4)
    return BalrogObject.TransferMatrix * np.reshape(BalrogObject.Window, (1,BalrogObject.Window.shape[0]))


def SGR2(truth, matched, des, band, truthcolumns, truthbins, measuredcolumns, measuredbins, nWalkers=1000, burnin=5000, steps=1000, out='SGPlots', hpfield='hpIndex', jfield='jacknife', threshold=0, evectors=None, evalues=None, master=None, Lcut=0, residual=False, n_component=4, sg=True, **other):
    #print '\n\n', len(truth[hpfield])

    index = truth[hpfield][0]
    jindex = truth[jfield][0]

    tmfile = os.path.join(out, 'TransferMatrix-%s-%i-%i.png' %(band,index,jindex))
    reconfile = os.path.join(out, 'ReconstructedHistogram-%s-%i-%i.png' %(band,index,jindex))
    chainfile = os.path.join(out, 'Chains-%s-%i-%i.png' %(band,index,jindex))
    burnfile = os.path.join(out, 'Burnin-%s-%i-%i.png' %(band,index,jindex))


    #BalrogObject = MCMC.BalrogLikelihood(truth, matched, truthcolumns=truthcolumns, truthbins=truthbins, measuredcolumns=measuredcolumns, measuredbins=measuredbins)
    BalrogObject = MCMC.BalrogLikelihood(truth, matched, truthcolumns=truthcolumns, truthbins=truthbins, measuredcolumns=measuredcolumns, measuredbins=measuredbins, threshold=threshold)
    if evectors is not None:
        w = np.reshape(BalrogObject.Window, (1,BalrogObject.Window.shape[0]))
        like = BalrogObject.TransferMatrix * w
        LikePCA, MasterPCA = PCA.doLikelihoodPCAfit(pcaComp=evectors, master=master, Lcut=Lcut, eigenval=evalues, likelihood=like, n_component=n_component, residual=residual)
        cut = (w > 0)
        BalrogObject.TransferMatrix[cut] = LikePCA[cut] / w[cut]
        BalrogObject.TransferMatrix[-cut] = 0

    fig = plt.figure()
    ax = fig.add_subplot(1,1, 1)
    BalrogObject.PlotTransferMatrix(fig, ax)
    plt.savefig(tmfile)

    #ReconObject = MCMC.MCMCReconstruction(BalrogObject, des, MCMC.ObjectLogL, truth=truth, nWalkers=nWalkers, reg=1.0e-10)
    #ReconObject = MCMC.MCMCReconstruction(BalrogObject, des, MCMC.ObjectLogL, truth=truth, nWalkers=nWalkers, reg=1.0e-10, samplelog=True)
    ReconObject = MCMC.MCMCReconstruction(BalrogObject, des, MCMC.ObjectLogThing, truth=truth, nWalkers=nWalkers, reg=1.0e-10)
    #ReconObject = MCMC.MCMCReconstruction(BalrogObject, des, MCMC.ObjectLogThing, truth=truth, nWalkers=nWalkers, reg=1.0e-10, samplelog=True)
    ReconObject.BurnIn(burnin, clear=False)

    c = len(truthbins[-1]) - 2
    chains = [c-4, c-3, c-2, c-1, c]
    
    #chains = [1, 10, -2]
    fig = plt.figure(figsize=(16,6))
    for i in range(len(chains)):
        ax = fig.add_subplot(1,len(chains), i+1)
        ReconObject.PlotChain(ax, chains[i], plotkwargs={'color':'black', 'linewidth':0.03}, twocolor=True)
    fig.tight_layout()
    plt.savefig(burnfile)
    plt.close(fig)

    ReconObject.ClearChain()


    ReconObject.Sample(steps)
    print index, jindex, np.average(ReconObject.Sampler.acceptance_fraction)
    acceptance = np.average(ReconObject.Sampler.acceptance_fraction)



    fig = plt.figure(figsize=(16,6))
    for i in range(len(chains)):
        ax = fig.add_subplot(1,len(chains), i+1)
        ReconObject.PlotChain(ax, chains[i], plotkwargs={'color':'black', 'linewidth':0.03}, twocolor=True)
    fig.tight_layout()
    plt.savefig(chainfile)
    plt.close(fig)

    wmid = len(BalrogObject.TruthBins[-1])-1
    wend = wmid * 2

    if sg:
        where = [0, None]
        galaxy_balrog_truth_center, galaxy_balrog_truth = BalrogObject.ReturnHistogram(kind='Truth', where=where)
        galaxy_recon_truth_center, galaxy_recon_truth, galaxy_recon_trutherr = ReconObject.ReturnHistogram(kind='RECONSTRUCTED', where=where)

        #where = None
        galaxy_balrog_obs_center, galaxy_balrog_obs = BalrogObject.ReturnHistogram(kind='Measured', where=where)
        galaxy_recon_obs_center, galaxy_recon_obs = ReconObject.ReturnHistogram(kind='Measured', where=where)
        galaxy_window = BalrogObject.Window[0:wmid]

        where = [1, None]
        star_balrog_truth_center, star_balrog_truth = BalrogObject.ReturnHistogram(kind='Truth', where=where)
        star_recon_truth_center, star_recon_truth, star_recon_trutherr = ReconObject.ReturnHistogram(kind='RECONSTRUCTED', where=where)

        #where = None
        star_balrog_obs_center, star_balrog_obs = BalrogObject.ReturnHistogram(kind='Measured', where=where)
        star_recon_obs_center, star_recon_obs = ReconObject.ReturnHistogram(kind='Measured', where=where)
        star_window = BalrogObject.Window[wmid:wend]

        where = [2, None]
        #where = None
        neither_balrog_obs_center, neither_balrog_obs = BalrogObject.ReturnHistogram(kind='Measured', where=where)

        #print len(galaxy_window), len(star_window), len(galaxy_balrog_truth)
        #balrog_obs = [galaxy_balrog_obs, star_balrog_obs, neither_balrog_obs, galaxy_balrog_obs_center]
        balrog_truth = [galaxy_balrog_truth, star_balrog_truth, galaxy_window, star_window, galaxy_balrog_truth_center]
        balrog_truth = [galaxy_balrog_truth, star_balrog_truth, galaxy_balrog_truth_center]
        recon_obs = [galaxy_recon_obs, star_recon_obs, galaxy_recon_obs_center]
        recon_truth = [galaxy_recon_truth, star_recon_truth, galaxy_recon_trutherr, star_recon_trutherr, galaxy_recon_truth_center]

    else:
        where = None
        galaxy_balrog_obs_center, galaxy_balrog_obs = BalrogObject.ReturnHistogram(kind='Measured', where=where)
        galaxy_balrog_truth_center, galaxy_balrog_truth = BalrogObject.ReturnHistogram(kind='Truth', where=where)
        galaxy_recon_obs_center, galaxy_recon_obs = ReconObject.ReturnHistogram(kind='Measured', where=where)
        galaxy_recon_truth_center, galaxy_recon_truth, galaxy_recon_trutherr = ReconObject.ReturnHistogram(kind='RECONSTRUCTED', where=where)

        balrog_obs = [galaxy_balrog_obs, galaxy_balrog_obs_center]
        balrog_truth = [galaxy_balrog_truth, BalrogObject.Window, galaxy_balrog_truth_center]
        recon_obs = [galaxy_recon_obs, galaxy_recon_obs_center]
        recon_truth = [galaxy_recon_truth, galaxy_recon_trutherr, galaxy_recon_truth_center]

    return balrog_obs, balrog_truth, recon_obs, recon_truth, index, jindex, len(truth), acceptance


def GatherWork(images, hpIndex, jIndex, sizes, acceptance):
    images = GatherImages(images)
    hpIndex, jIndex, sizes, acceptance = mpifunctions.Gather(hpIndex, jIndex, sizes, acceptance)
    return images, hpIndex, jIndex, sizes, acceptance


def SaveWork(images, hpIndex, jIndex, sizes, acceptance, map, mcmc):
    names = [os.path.join(map['version'], 'SG-Balrog-Observed-%s.fits' %(map['version'])),
             os.path.join(map['version'], 'SG-Balrog-Truth-%s.fits' %(map['version'])),
             os.path.join(map['version'], 'SG-Data-Observed-%s.fits' %(map['version'])),
             os.path.join(map['version'], 'SG-Data-Reconstructed-%s.fits' %(map['version']))]
    if MPI.COMM_WORLD.Get_rank()==0:
        for i in range(len(images)):
            hdus = [pyfits.PrimaryHDU()]
            for j in range(images[i].shape[1]-1):
                hdus.append( pyfits.ImageHDU(images[i][:,j,:]) )
            hdus.append( pyfits.ImageHDU(images[i][0, -1, :]) )

            tab = np.zeros( images[i].shape[0], dtype=[(map['hpfield'],np.int64), (mcmc['jackfield'],np.int64), ('numTruth',np.int64), ('acceptance',np.float64)] )
            tab[map['hpfield']] = np.int64(hpIndex)
            tab[mcmc['jackfield']] = np.int64(jIndex)
            tab['numTruth'] = np.int64(sizes)
            tab['acceptance'] = acceptance
            hdus.append( pyfits.BinTableHDU(tab) )
            hdus = pyfits.HDUList(hdus)
            hdus.writeto(names[i], clobber=True)

        '''
        if map['nside'] is not None:
            SGRecon2Map(names[-1], map['version'], magmin=map['summin'], magmax=map['summax'], nside=map['nside'], nest=map['nest'])
        '''

def RemoveNotUsed(truth, matched, nosim, des, mcmc, band):
    if MPI.COMM_WORLD.Get_rank()==0:
        truth = RemoveFields(truth, mcmc['truthcolumns'])
        des = RemoveFields(des, mcmc['measuredcolumns'])

        mcols = np.append(mcmc['truthcolumns'], mcmc['measuredcolumns'])
        matched = RemoveFields(matched, mcols)

        ncols = []
        for name in mcmc['measuredcolumns']:
            ncols.append(name.replace('_%s'%(band), '_det'))
        nosim = RemoveFields(nosim, ncols)

    return truth, matched, nosim, des


def RemoveFields(arr, columns):
    remove = []
    for name in arr.dtype.names:
        if (name not in columns) and (name.find('balrog_index')==-1) and (name.find('ra')==-1) and (name.find('dec')==-1) and (name.find('alphawin_j2000')==-1) and (name.find('deltawin_j2000')==-1):
            remove.append(name)
    arr = recfunctions.drop_fields(arr, remove, usemask=False)
    return arr


def DoSGRecon(select, mcmc, map, dbwrite=False, read=False, query=True, dbdir='/gpfs/mnt/gpfs01/astro/astronfs03/workarea/esuchyta/DBFits'):
    band = select['bands'][0]
    bversion = select['table']
    dir = os.path.join(dbdir, bversion)
    if MPI.COMM_WORLD.Get_rank()==0:
        if not os.path.exists(mcmc['out']):
            os.makedirs(mcmc['out'])

    if query:
        if MPI.COMM_WORLD.Get_rank()==0:
            print 'Querying DB'
        truth, matched, nosim, des = DBfunctions.GetAllViaTileQuery(select)
        #truth, matched, nosim, des = DBfunctions.GetAllViaTileQuery(select, limit=24)

    if dbwrite:
        if MPI.COMM_WORLD.Get_rank()==0:
            if not os.path.exists(dir):
                os.makedirs(dir)
            esutil.io.write(os.path.join(dir,'truth-%s.fits'%(band)), truth, clobber=True)
            esutil.io.write(os.path.join(dir,'matched-%s.fits'%(band)), matched, clobber=True)
            esutil.io.write(os.path.join(dir,'nosim-%s.fits'%(band)), nosim, clobber=True)
            esutil.io.write(os.path.join(dir,'des-%s.fits'%(band)), des, clobber=True)
    
    #sys.exit()

    if read:
        if MPI.COMM_WORLD.Get_rank()==0:
            truth = esutil.io.read(os.path.join(dir,'truth-%s.fits'%(band)))
            matched = esutil.io.read(os.path.join(dir,'matched-%s.fits'%(band)))
            nosim = esutil.io.read(os.path.join(dir,'nosim-%s.fits'%(band)))
            des = esutil.io.read(os.path.join(dir,'des-%s.fits'%(band)))
        else:
            truth = None
            matched = None
            nosim = None
            des = None

    truth, matched, nosim, des = SGClassify(truth, matched, nosim, des)
    truth, matched, nosim, des = RemoveNotUsed(truth, matched, nosim, des, mcmc, band)

    if MPI.COMM_WORLD.Get_rank()==0:
        print 'Scattering data'
    truth, matched, nosim, des = DivideWork(truth, matched, nosim, des, band, map, mcmc)

    if MPI.COMM_WORLD.Get_rank()==0:
        print 'MCMCing'
    images, hpIndex, jIndex, sizes, acceptance = DoSGMcmc(truth, matched, nosim, des, mcmc, band, map)


    if MPI.COMM_WORLD.Get_rank()==0:
        print 'Regathering data'
    images, hpIndex, jIndex, sizes, acceptance = GatherWork(images, hpIndex, jIndex, sizes, acceptance)

    if MPI.COMM_WORLD.Get_rank()==0:
        print 'Saving results'
    SaveWork(images, hpIndex, jIndex, sizes, acceptance, map, mcmc)


def DoStarGalaxy(select, mcmc, map):
    band = select['bands'][0]
    bi = 'balrog_index_%s' %(band)
    #truth, matched, nosim, des = DBfunctions.GetAllViaTileQuery(select, limit=40)
    truth, matched, nosim, des = DBfunctions.GetAllViaTileQuery(select)

    if MPI.COMM_WORLD.Get_rank()==0:
        print len(truth), len(nosim), len(matched), len(des)
        matched = Modestify(matched, byband=band)
        des = Modestify(des, byband=band)
        if not os.path.exists(mcmc['out']):
            os.makedirs(mcmc['out'])

    if map['nside'] is None:
        if MPI.COMM_WORLD.Get_rank()==0:
            truth = [truth]
            matched = [matched]
            nosim = [nosim]
            des = [des]
        else:
            truth = []
            matched = []
            nosim = []
            des = []
    else:
        truth, matched, nosim, des = healpyfunctions.ScatterByHealPixel(truth, matched, nosim, des, band, nside=map['nside'], nest=map['nest'])

  
    size = len(truth)
    #size = 1
    images = [ [], [], [], [] ]
    hpIndex = np.empty(size)
    sizes = np.empty(size)
    acceptance = np.empty(size)

    for i in range(size):
        if len(truth) > 0:
            if map['nside'] is None:
                o, t, d, r, h, s, a = StarGalaxyRecon(truth[i], matched[i], des[i], band, mcmc['truthcolumns'], mcmc['truthbins'], mcmc['measuredcolumns'], mcmc['measuredbins'], nWalkers=mcmc['nWalkers'], burnin=mcmc['burnin'], steps=mcmc['steps'], out=mcmc['out'], hpfield=None)
            else:
                o, t, d, r, h, s, a = StarGalaxyRecon(truth[i], matched[i], des[i], band, mcmc['truthcolumns'], mcmc['truthbins'], mcmc['measuredcolumns'], mcmc['measuredbins'], nWalkers=mcmc['nWalkers'], burnin=mcmc['burnin'], steps=mcmc['steps'], out=mcmc['out'], hpfield=map['hpfield'])
            images[0].append(o)
            images[1].append(t)
            images[2].append(d)
            images[3].append(r)
            hpIndex[i] = h
            sizes[i] = s
            acceptance[i] = a


    names = [os.path.join(map['version'], 'SG-Balrog-Observed-%s.fits' %(map['version'])),
             os.path.join(map['version'], 'SG-Balrog-Truth-%s.fits' %(map['version'])),
             os.path.join(map['version'], 'SG-Data-Observed-%s.fits' %(map['version'])),
             os.path.join(map['version'], 'SG-Data-Reconstructed-%s.fits' %(map['version']))]
    images = GatherImages(images)
    hpIndex, sizes, acceptance = mpifunctions.Gather(hpIndex, sizes, acceptance)
    if MPI.COMM_WORLD.Get_rank()==0:
        for i in range(len(images)):
            hdus = [pyfits.PrimaryHDU()]
            for j in range(images[i].shape[1]-1):
                hdus.append( pyfits.ImageHDU(images[i][:,j,:]) )
            hdus.append( pyfits.ImageHDU(images[i][0, -1, :]) )

            tab = np.zeros( images[i].shape[0], dtype=[(map['hpfield'],np.int64), ('numTruth',np.int64), ('acceptance',np.float64)] )
            tab[map['hpfield']] = np.int64(hpIndex)

            tab['numTruth'] = np.int64(sizes)
            tab['acceptance'] = acceptance
            hdus.append( pyfits.BinTableHDU(tab) )
            hdus = pyfits.HDUList(hdus)
            hdus.writeto(names[i], clobber=True)

        if map['nside'] is not None:
            SGRecon2Map(names[-1], map['version'], magmin=map['summin'], magmax=map['summax'], nside=map['nside'], nest=map['nest'])





if __name__=='__main__': 
    
    version = 'sva1v2'
    size = 0.5
    band = sys.argv[1]
    sg = True

    if band =='i':
        min = 17.5
        max = 25.0
        #max = 25.5
        tbins = np.arange(min, max+size, size)

        min = 17.5
        #max = 27.5
        #max = 28.5
        max = 34.5
        obins = np.arange(min, max+size, size)
        obins = np.insert(obins, 0, -100)
        obins = np.insert(obins, len(obins), 100)

    elif band=='r':
        min = 17.5
        max = 26.2
        tbins = np.arange(min, max+size, size)

        min = 17.5
        #max = 28.5
        max = 34.5
        obins = np.arange(min, max+size, size)
        obins = np.insert(obins, 0, -100)
        obins = np.insert(obins, len(obins), 100)

    elif band=='z':
        min = 17.5
        #max = 25.45
        max = 25.0
        tbins = np.arange(min, max+size, size)

        min = 17.5
        max = 28.5
        obins = np.arange(min, max+size, size)
        obins = np.insert(obins, 0, -100)
        obins = np.insert(obins, len(obins), 100)

    
    tc = ['objtype_%s'%(band), 'mag_%s'%(band)]
    tb = [np.arange(0.5, 5, 2.0), tbins]
    mc = ['modtype_%s'%(band), 'mag_auto_%s'%(band)]
    mb = [np.arange(0.5, 7, 2.0), obins]
    if not sg:
        tc = tc[-1:]
        tb = tb[-1:]
        mc = mc[-1:]
        mb = mb[-1:]


    DBselect = {'table': version,
                'des': 'sva1_coadd_objects',
                'bands': [band],
                'truth': ['balrog_index', 'mag', 'ra', 'dec', 'objtype'],
                'sim': ['mag_auto', 'flux_auto', 'fluxerr_auto', 'flags', 'spread_model', 'spreaderr_model', 'class_star', 'mag_psf', 'alphawin_j2000', 'deltawin_j2000']
               }

    MCMCconfig = {'sg': sg,
                  'truthcolumns': tc,
                  'truthbins': tb,
                  'measuredcolumns': mc,
                  'measuredbins': mb,

                  'threshold': 0,
                  'PCAon': None,
                  'Lcut': 0,
                  'n_component': 4,
                  'residual': False,

                  'burnin': 3000,
                  'steps': 1000,
                  'nWalkers': 1000,

                  'jackfield': 'jacknife',
                  'jacknife': 9,
                  'jackdes': True
                 }

    MapConfig = {#'nside': 64,
                 'nside': None,
                 'hpfield': 'hpIndex',

                 'version': '%s-%s-mbins=%.1f-sg=%s-Lcut=%s' %(version,band,size,str(sg), MCMCconfig['Lcut']),
                 'nest': False,
                 'summin': 22.5,
                 'summax': 24.5}

    MCMCconfig['out'] = os.path.join(MapConfig['version'], 'SGPlots-%s'%(MapConfig['version']))

    DoSGRecon(DBselect, MCMCconfig, MapConfig, query=False, dbwrite=False, read=True)
    #DoSGRecon(DBselect, MCMCconfig, MapConfig, query=True, dbwrite=True, read=False)

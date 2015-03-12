#!/usr/bin/env python

import numpy as np
import numpy.lib.recfunctions as recfunctions
from mpi4py import MPI
import os
import pyfits
import healpy as hp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mpifunctions
import DBfunctions
import MCMC
import healpyfunctions



def Modestify(data, byband='i'):
    modest = np.zeros(len(data), dtype=np.int32)

    galcut = (data['flags_%s'%(byband)] <=3) & -( ((data['class_star_%s'%(byband)] > 0.3) & (data['mag_auto_%s'%(byband)] < 18.0)) | ((data['spread_model_%s'%(byband)] + 3*data['spreaderr_model_%s'%(byband)]) < 0.003) | ((data['mag_psf_%s'%(byband)] > 30.0) & (data['mag_auto_%s'%(byband)] < 21.0)))
    modest[galcut] = 1

    starcut = (data['flags_%s'%(byband)] <=3) & ((data['class_star_%s'%(byband)] > 0.3) & (data['mag_auto_%s'%(byband)] < 18.0) & (data['mag_psf_%s'%(byband)] < 30.0) | (((data['spread_model_%s'%(byband)] + 3*data['spreaderr_model_%s'%(byband)]) < 0.003) & ((data['spread_model_%s'%(byband)] +3*data['spreaderr_model_%s'%(byband)]) > -0.003)))
    modest[starcut] = 3

    neither = -(galcut | starcut)
    modest[neither] = 5

    data = recfunctions.append_fields(data, 'modtype_%s'%(byband), modest)
    return data



def StarGalaxyRecon(truth, matched, des, band, truthcolumns, truthbins, measuredcolumns, measuredbins, nWalkers=1000, burnin=5000, steps=1000, out='SGPlots', hpfield='hpIndex'):
    index = truth[hpfield][0]
    tmfile = os.path.join(out, 'TransferMatrix-%s-%i.png' %(band,index))
    reconfile = os.path.join(out, 'ReconstructedHistogram-%s-%i.png' %(band,index))
    chainfile = os.path.join(out, 'Chains-%s-%i.png' %(band,index))
    burnfile = os.path.join(out, 'Burnin-%s-%i.png' %(band,index))


    BalrogObject = MCMC.BalrogLikelihood(truth, matched, truthcolumns=truthcolumns, truthbins=truthbins, measuredcolumns=measuredcolumns, measuredbins=measuredbins)
    ReconObject = MCMC.MCMCReconstruction(BalrogObject, des, MCMC.ObjectLogL, truth=truth, nWalkers=nWalkers, reg=1.0e-10)
    #ReconObject = MCMC.MCMCReconstruction(BalrogObject, des, MCMC.ObjectLogL, truth=truth, nWalkers=nWalkers, reg=1.0e-10, samplelog=True)
    ReconObject.BurnIn(burnin)

    c = len(measuredbins[-1]) - 1
    chains = [c-4, c-3, c-2, c-1, c]
    
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


    fig = plt.figure()
    ax = fig.add_subplot(1,1, 1)
    BalrogObject.PlotTransferMatrix(fig, ax)
    plt.savefig(tmfile)

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
    healpyfunctions.VisualizeHealPixMap(GalMap, title='Galaxy-map-%s'%(version), nest=nest, vmin=galmin, vmax=galmax, background='gray')
    healpyfunctions.VisualizeHealPixMap(StarMap, title='Star-map-%s'%(version), nest=nest, vmin=starmin, vmax=starmax, background='gray')


def DoStarGalaxy(select, mcmc, map):
    band = select['bands'][0]
    bi = 'balrog_index_%s' %(band)
    #truth, matched, nosim, des = DBfunctions.GetAllViaTileQuery(select, limit=8)
    truth, matched, nosim, des = DBfunctions.GetAllViaTileQuery(select)

    if MPI.COMM_WORLD.Get_rank()==0:
        print len(truth), len(nosim), len(matched), len(des)
        matched = Modestify(matched, byband=band)
        des = Modestify(des, byband=band)
        if not os.path.exists(mcmc['out']):
            os.makedirs(mcmc['out'])
    truth, matched, nosim, des = healpyfunctions.ScatterByHealPixel(truth, matched, nosim, des, band, nside=map['nside'], nest=map['nest'])

  
    size = len(truth)
    #size = 1
    images = [ [], [], [], [] ]
    hpIndex = np.empty(size)
    sizes = np.empty(size)
    acceptance = np.empty(size)

    for i in range(size):
        if len(truth) > 0:
            o, t, d, r, h, s, a = StarGalaxyRecon(truth[i], matched[i], des[i], band, mcmc['truthcolumns'], mcmc['truthbins'], mcmc['measuredcolumns'], mcmc['measuredbins'], nWalkers=mcmc['nWalkers'], burnin=mcmc['burnin'], steps=mcmc['steps'], out=mcmc['out'])
            images[0].append(o)
            images[1].append(t)
            images[2].append(d)
            images[3].append(r)
            hpIndex[i] = h
            sizes[i] = s
            acceptance[i] = a


    names = ['SG-Balrog-Observed-%s.fits' %(map['version']),
             'SG-Balrog-Truth-%s.fits' %(map['version']),
             'SG-Data-Observed-%s.fits' %(map['version']),
             'SG-Data-Reconstructed-%s.fits' %(map['version'])]
    images = GatherImages(images)
    hpIndex, sizes, acceptance = mpifunctions.Gather(hpIndex, sizes, acceptance)
    if MPI.COMM_WORLD.Get_rank()==0:
        for i in range(len(images)):
            hdus = [pyfits.PrimaryHDU()]
            for j in range(images[i].shape[1]-1):
                hdus.append( pyfits.ImageHDU(images[i][:,j,:]) )
            hdus.append( pyfits.ImageHDU(images[i][0, -1, :]) )

            tab = np.zeros( images[i].shape[0], dtype=[('hpIndex',np.int64), ('numTruth',np.int64), ('acceptance',np.float64)] )
            tab['hpIndex'] = np.int64(hpIndex)
            tab['numTruth'] = np.int64(sizes)
            tab['acceptance'] = acceptance
            hdus.append( pyfits.BinTableHDU(tab) )
            hdus = pyfits.HDUList(hdus)
            hdus.writeto(names[i], clobber=True)

        SGRecon2Map(names[-1], map['version'], magmin=map['summin'], magmax=map['summax'], nside=map['nside'], nest=map['nest'])


def PlotRecon(ax, file, ext, hp, errext=None, kind='plot', coordsext=-2, hpext=-1, plotkwargs={}):
    hdus = pyfits.open(file)
    hps = hdus[hpext].data['hpIndex']
    norm = 1.0/hdus[hpext].data['numTruth'] * np.power(10.0,5.0)/np.power(np.power(10.0, 4.0), 2.0) * 1.0/np.power(0.27, 2.0) * np.power(3600.0, 2.0)

    line = np.arange(len(hps))[hps==hp][0]
    coords = hdus[coordsext].data
    vals = hdus[ext].data[line, :] * norm[line]


    if errext!=None:
        err = hdus[errext].data[line, :] * norm[line]
    
    if kind=='plot':
        ax.plot(coords, vals, **plotkwargs)
    elif kind=='scatter':
        ax.scatter(coords, vals, **plotkwargs)
    elif kind=='errorbar':
        ax.errorbar(coords, vals, err, **plotkwargs)

    return ax




if __name__=='__main__': 

    band = 'i'

    MapConfig = {'nside': 64,
                 #'nside': 256,
                 'nest': False,
                 'version': 'v6',
                 'summin': 22.5,
                 'summax': 24.5}

    DBselect = {'table': 'sva1v2',
              'des': 'sva1_coadd_objects',
              'bands': [band],
              'truth': ['balrog_index', 'mag', 'ra', 'dec', 'objtype'],
              'sim': ['mag_auto', 'flux_auto', 'fluxerr_auto', 'flags', 'spread_model', 'spreaderr_model', 'class_star', 'mag_psf', 'alphawin_j2000', 'deltawin_j2000']
             }

    MCMCconfig = {'truthcolumns': ['objtype_%s'%(band), 'mag_%s'%(band)],
                  'truthbins': [np.arange(0.5, 5, 2.0), np.arange(17.5,25,0.25)],
                  'measuredcolumns': ['modtype_%s'%(band), 'mag_auto_%s'%(band)],
                  'measuredbins': [np.arange(0.5, 7, 2.0), np.arange(17.5,25,0.25)],
                  'burnin': 6000,
                  'steps': 1000,
                  'nWalkers': 1000,
                  'out': 'SGPlots-%s'%(MapConfig['version']),
                 }

    DoStarGalaxy(DBselect, MCMCconfig, MapConfig)

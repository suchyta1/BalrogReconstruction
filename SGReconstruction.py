#!/usr/bin/env python

import numpy as np
import numpy.lib.recfunctions as recfunctions
from mpi4py import MPI
import os
import pyfits
import esutil
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
    ReconObject = MCMC.MCMCReconstruction(BalrogObject, des, MCMC.ObjectLogThing, truth=truth, nWalkers=nWalkers, reg=1.0e-10)
    ReconObject.BurnIn(burnin)

    c = len(truthbins[-1]) - 2
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
    healpyfunctions.VisualizeHealPixMap(GalMap, title=os.path.join(version, 'Galaxy-map-%s'%(version)), nest=nest, vmin=galmin, vmax=galmax, background='gray')
    healpyfunctions.VisualizeHealPixMap(StarMap, title=os.path.join(version, 'Star-map-%s'%(version)), nest=nest, vmin=starmin, vmax=starmax, background='gray')



def ByHealpixelAndJack(arrs, map, band, mcmc, balrogcoords=['ra','dec'], descoords=['alphawin_j2000','deltawin_j2000'], minnum=None):
    hpfield = map['hpfield']
    jfield = mcmc['jackfield']
    if minnum is None:
        minnum = mcmc['jacknife']

    for i in range(len(arrs)):
        if map['nside'] is None:
            #hpInd = np.array( [-1]*len(arrs[i]) )
            hpInd = np.zeros( len(arrs[i]), dtype=np.int64 )
        else:
            if i < (len(arrs[i])-1):
                ra = '%s_%s' %(balrogcoords[0], band)
                dec = '%s_%s' %(balrogcoords[1], band)
            else:
                desra = '%s_%s' %(descoords[0], band)
                desdec = '%s_%s' %(descoords[1], band)

            hpInd = healpyfunctions.RaDec2Healpix(arrs[i][ra], arrs[i][dec], map['nside'], nest=map['nest'])
            
        if i==0:
            uInds = np.unique(hpInd)
        else:
            uInds = np.intersect1d(uInds, hpInd)

        arrs[i] = recfunctions.append_fields(arrs[i], hpfield, hpInd)

    for i in range(len(arrs)):
        cut = np.in1d(arrs[i][hpfield], uInds)
        arrs[i] = arrs[i][cut]

    p = np.arange( len(uInds)*mcmc['jacknife'] )
    for i in range(len(arrs)):

        if i in [0, len(arrs)-1]:
            
            count = np.zeros(len(arrs[i]))
            jack = np.zeros(len(arrs[i]))
            cc = np.unique(arrs[i][hpfield])

            for j in range(len(cc)):
                cut = (arrs[i][hpfield]==cc[j])
                cnt = np.sum(cut)
                count[cut] = np.sum(cnt)
                if i==0:
                    ipart = cnt / mcmc['jacknife']
                    rpart = cnt % mcmc['jacknife']
                    ii = np.append( np.repeat( np.arange(0, mcmc['jacknife'], 1), ipart ), np.arange(0, rpart, 1) )
                    np.random.shuffle(ii)
                    jack[cut] = ii
                    
                
            cut = (count > minnum)
            uInds = np.intersect1d(uInds, arrs[i][cut][hpfield])
            pindex = arrs[i][hpfield]*mcmc['jacknife'] + jack
            arrs[i] = recfunctions.append_fields(arrs[i], 'pindex', pindex)


        elif i < len(arrs)-1:
            pindex = np.zeros(len(arrs[i]))
            count = np.zeros(len(arrs[i]))
            for pp in range(len(p)):
                where = (arrs[0]['pindex']==pp)
                b = arrs[0]['balrog_index_%s'%(band)][where]
                cut = np.in1d(arrs[i]['balrog_index_%s'%(band)], b)
                pindex[cut] = pp
                count[cut] = np.sum(cut)

            arrs[i] = recfunctions.append_fields(arrs[i], 'pindex', pindex)
            cut = (count > minnum)
            uInds = np.intersect1d(uInds, arrs[i][cut][hpfield])
    
    for i in range(len(arrs)):
        cut = np.in1d(arrs[i][hpfield], uInds)
        arrs[i] = arrs[i][cut]
        arrs[i] = recfunctions.append_fields(arrs[i], jfield, np.mod(arrs[i]['pindex'],mcmc['jacknife']))

    a = []
    for i in range(len(arrs)):
        a.append( [] )
        pind = np.unique(arrs[i]['pindex'])
        for j in range(len(pind)):
            cut = (arrs[i]['pindex']==pind[j])
            a[i].append( arrs[i][cut] )
        if i==(len(arrs)-1):
            a[i] = list( np.repeat(a[i], mcmc['jacknife'], axis=0) )
    return a

    '''
    for i in range(len(arrs)):
        arrs[i] = recfunctions.append_fields(arrs[i], jfield, np.mod(arrs[i]['pindex'],mcmc['jacknife']))
        arrs[i] = np.sort(arrs[i], order='pindex')
        splitby = np.unique(arrs[i]['pindex'])
        if len(splitby) > 1:
            splitat = np.searchsorted(splitby[1:])
            arrs[i] = np.split(arrs[i], splitat)
        else:
            arrs[i] = [arrs[i]]

        if i==(len(arrs)-1):
            arrs[i] = list( np.repeat(arrs[i], mcmc['jacknife'], axis=0) )

    return arrs
    '''



def DivideWork(truth, matched, nosim, des, band, map, mcmc, balrogcoords=['ra','dec'], descoords=['alphawin_j2000','deltawin_j2000'], minnum=None):
    if MPI.COMM_WORLD.Get_rank()==0:
        truth, matched, nosim, des = ByHealpixelAndJack([truth, matched, nosim, des], map, band, mcmc, balrogcoords=balrogcoords, descoords=descoords, minnum=minnum)
        
    if MPI.COMM_WORLD.Get_rank()==0:
        for i in range(len(truth)):
            print len(truth[i]), len(matched[i]), len(nosim[i]), len(des[i])
    truth, matched, nosim, des = mpifunctions.Scatter(truth, matched, nosim, des)
    if MPI.COMM_WORLD.Get_rank()==0:
        print 'f'
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

    for i in range(size):
        if len(truth) > 0:
            o, t, d, r, h, j, s, a = SGR2(truth[i], matched[i], des[i], band, mcmc['truthcolumns'], mcmc['truthbins'], mcmc['measuredcolumns'], mcmc['measuredbins'], nWalkers=mcmc['nWalkers'], burnin=mcmc['burnin'], steps=mcmc['steps'], out=mcmc['out'], hpfield=map['hpfield'])
            images[0].append(o)
            images[1].append(t)
            images[2].append(d)
            images[3].append(r)
            hpIndex[i] = h
            jIndex[i] = j
            sizes[i] = s
            acceptance[i] = a
    return images, hpIndex, jIndex, sizes, acceptance



def SGR2(truth, matched, des, band, truthcolumns, truthbins, measuredcolumns, measuredbins, nWalkers=1000, burnin=5000, steps=1000, out='SGPlots', hpfield='hpIndex', jfield='jacknife'):
    index = truth[hpfield][0]
    jindex = truth[jfield][0]

    tmfile = os.path.join(out, 'TransferMatrix-%s-%i.png' %(band,index))
    reconfile = os.path.join(out, 'ReconstructedHistogram-%s-%i.png' %(band,index))
    chainfile = os.path.join(out, 'Chains-%s-%i.png' %(band,index))
    burnfile = os.path.join(out, 'Burnin-%s-%i.png' %(band,index))


    BalrogObject = MCMC.BalrogLikelihood(truth, matched, truthcolumns=truthcolumns, truthbins=truthbins, measuredcolumns=measuredcolumns, measuredbins=measuredbins)
    #ReconObject = MCMC.MCMCReconstruction(BalrogObject, des, MCMC.ObjectLogL, truth=truth, nWalkers=nWalkers, reg=1.0e-10)
    #ReconObject = MCMC.MCMCReconstruction(BalrogObject, des, MCMC.ObjectLogL, truth=truth, nWalkers=nWalkers, reg=1.0e-10, samplelog=True)
    ReconObject = MCMC.MCMCReconstruction(BalrogObject, des, MCMC.ObjectLogThing, truth=truth, nWalkers=nWalkers, reg=1.0e-10)
    ReconObject.BurnIn(burnin)

    c = len(truthbins[-1]) - 2
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
    print index, jindex, np.average(ReconObject.Sampler.acceptance_fraction)
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


def DoSGRecon(select, mcmc, map):
    band = select['bands'][0]
    
    truth, matched, nosim, des = DBfunctions.GetAllViaTileQuery(select, limit=40)
    '''
    if MPI.COMM_WORLD.Get_rank()==0:
        #esutil.io.write('tmp/truth.fits', truth)
        #esutil.io.write('tmp/matched.fits', matched)
        #esutil.io.write('tmp/nosim.fits', nosim)
        #esutil.io.write('tmp/des.fits', des)

        truth = esutil.io.read('tmp/truth.fits')
        matched = esutil.io.read('tmp/matched.fits')
        nosim = esutil.io.read('tmp/nosim.fits')
        des = esutil.io.read('tmp/des.fits')

        if not os.path.exists(mcmc['out']):
            os.makedirs(mcmc['out'])
    else:
        truth = None
        matched = None
        nosim = None
        des = None
    '''


    if MPI.COMM_WORLD.Get_rank()==0:
        print 'classify'
    truth, matched, nosim, des = SGClassify(truth, matched, nosim, des)

    if MPI.COMM_WORLD.Get_rank()==0:
        print 'divide'
    truth, matched, nosim, des = DivideWork(truth, matched, nosim, des, band, map, mcmc)

    if MPI.COMM_WORLD.Get_rank()==0:
        print 'mcmc'
    images, hpIndex, jIndex, sizes, acceptance = DoSGMcmc(truth, matched, nosim, des, mcmc, band, map)


    if MPI.COMM_WORLD.Get_rank()==0:
        print 'gather'
    images, hpIndex, jIndex, sizes, acceptance = GatherWork(images, hpIndex, jIndex, sizes, acceptance)

    if MPI.COMM_WORLD.Get_rank()==0:
        print 'save'
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

    size = 0.75
    min = 17.5
    max = 25.0
    tbins = np.arange(min, max+size, size)

    min = 17.5
    max = 27.5
    obins = np.arange(min, max+size, size)
    obins = np.insert(obins, 0, -100)
    obins = np.insert(obins, len(obins), 100)
    
    band = 'i'

    MapConfig = {#'nside': 64,
                 'nside': None,
                 'hpfield': 'hpIndex',

                 'version': 'sva1v2-fullarea',
                 'nest': False,
                 'summin': 22.5,
                 'summax': 24.5}

    DBselect = {'table': 'sva1v2',
              'des': 'sva1_coadd_objects',
              'bands': [band],
              'truth': ['balrog_index', 'mag', 'ra', 'dec', 'objtype'],
              'sim': ['mag_auto', 'flux_auto', 'fluxerr_auto', 'flags', 'spread_model', 'spreaderr_model', 'class_star', 'mag_psf', 'alphawin_j2000', 'deltawin_j2000']
             }

    MCMCconfig = {'truthcolumns': ['objtype_%s'%(band), 'mag_%s'%(band)],
                  'truthbins': [np.arange(0.5, 5, 2.0), tbins],
                  'measuredcolumns': ['modtype_%s'%(band), 'mag_auto_%s'%(band)],
                  'measuredbins': [np.arange(0.5, 7, 2.0), obins],
                  #'burnin': 10000,
                  'burnin': 5000,
                  'steps': 1000,
                  'nWalkers': 1000,
                  'out': os.path.join(MapConfig['version'], 'SGPlots-%s'%(MapConfig['version'])),

                  'jackfield': 'jacknife',
                  'jacknife': 10,
                 }

    #DoStarGalaxy(DBselect, MCMCconfig, MapConfig)
    DoSGRecon(DBselect, MCMCconfig, MapConfig)

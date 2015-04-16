import desdb
import numpy as np
import esutil
import pyfits
import sys
import argparse
import healpy as hp
import os

#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt


def likelihoodPCA(likelihood,  likelihood_master=None, doplot=False, outdir='./', band=None, extent=None, residual=False, plotscale=1.0e-3, nticks=3):
  # This does a simple PCA on the array of likelihood matrices to find a compact basis with which to represent the likelihood.
    origShape = np.shape(likelihood)
    likelihood_1d = np.reshape(likelihood, (origShape[0]*origShape[1], origShape[2]))

    if residual:
        L1d_master = np.reshape(likelihood_master, origShape[0]*origShape[1])
    
        # Subtract L1d_master from each row of L1d:
        likelihood_1d = likelihood_1d - np.reshape(L1d_master, (1, L1d_master.shape[0]))

    L1d = likelihood_1d.T
    U,s,Vt = np.linalg.svd(L1d,full_matrices=False)
    V = Vt.T
    ind = np.argsort(s)[::-1]
    ind = np.argsort(s)[::-1]
    U = U[:, ind]
    s = s[ind]
    V = V[:, ind]

    if residual:
        likelihood_1d = likelihood_1d + np.reshape(L1d_master, (1, L1d_master.shape[0]))
  
    likelihood_pcomp = V.reshape(origShape)
    
    if doplot is True:
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.colors import LogNorm, Normalize
        if band is None:
            print "Must supply band (g,r,i,z,Y) in order to save PCA plots."
            stop
            
        pp = PdfPages(os.path.join(outdir, 'likelihood_pca_components.pdf'))

        for i,thing in zip(xrange(s.size),s):
            fig,ax = plt.subplots(nrows=1,ncols=1)
            LikelihoodArcsinh(likelihood_pcomp[:,:,i], fig, ax, plotscale=plotscale, nticks=nticks, extent=extent)
            
            '''
            #aarg = -likelihood_pcomp[:,:,i]/plotscale
            aarg = likelihood_pcomp[:,:,i]/plotscale
            aimage = np.arcsinh(aarg)
            amin = np.amin(aimage)
            amax = np.amax(aimage)

            aticks = np.linspace(amin, amax, num=nticks)
            alabels = np.sinh(aticks) * plotscale
            tlabels = []
            for i in range(len(alabels)):
                tlabels.append('%.2e'%(alabels[i]))

            im = ax.imshow(aimage, origin='lower', cmap=plt.cm.Greys, extent=extent)
            cbar = fig.colorbar(im, ax=ax, ticks=aticks)
            cbar.ax.set_yticklabels(tlabels)
            
            #im = ax.imshow( -likelihood_pcomp[:,:,i],origin='lower',cmap=plt.cm.Greys, extent = extent,vmin=-1,vmax=1)
            #ax.set_xlabel(band+' mag (true)')
            #ax.set_ylabel(band+' mag (meas)')
            #fig.colorbar(im,ax=ax)
            
            plt.tight_layout()
            '''
            pp.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        fig,ax = plt.subplots(1,1,figsize = (6.,6.) )
        ax.plot(np.abs(s))
        ax.set_yscale('log')
        ax.set_xlabel('rank')
        ax.set_ylabel('eigenvalue')
        pp.savefig(fig)
        pp.close()

    return likelihood_pcomp, s


def LikelihoodArcsinh(like, fig, ax, plotscale=1.0e-3, nticks=3, extent=None, start=True, cax=None):
    aarg = like/plotscale
    aimage = np.arcsinh(aarg)
    amin = np.amin(aimage)
    amax = np.amax(aimage)

    aticks = np.linspace(amin, amax, num=nticks)
    alabels = np.sinh(aticks) * plotscale
    tlabels = []
    for i in range(len(alabels)):
        #tlabels.append('%.2e'%(alabels[i]))
        tlabels.append('%.3f'%(alabels[i]))
    

    ''' 
    aimage = np.copy(like)
    c = (like > 0)
    aimage[c] = np.log10(like[c])
    #amin = np.amin(aimage)
    amin = -4
    amax = np.amax(aimage)
    aimage[-c] = amin
    cc = (aimage <= amin)
    aimage[cc] = amin
    alabels =  np.logspace(amin, amax, num=nticks) 
    aticks = np.log10(alabels)
    tlabels = []
    for i in range(len(alabels)):
        tlabels.append('%.1e'%(alabels[i]))
    '''
   

    im = ax.imshow(aimage, origin='lower', cmap=plt.cm.Greys, extent=extent, interpolation='nearest')
    if cax is None:
        cbar = fig.colorbar(im, ticks=aticks)
    else:
        cbar = plt.colorbar(im, cax=cax, ticks=aticks)
    cbar.ax.set_yticklabels(tlabels)

    '''
    ff, aa = plt.subplots(nrows=1, ncols=1)
    im = aa.imshow(aimage, origin='lower', cmap=plt.cm.Greys, extent=extent, interpolation='nearest')
    if cax is None:
        cbar = ff.colorbar(im, ticks=aticks)
    else:
        cbar = plt.colorbar(im, cax=cax, ticks=aticks)
    cbar.ax.set_yticklabels(tlabels)
    #aa.axhline(y=22.5, color='red')
    aa.set_xlabel('Truth Magnitude', fontsize=20)
    aa.set_ylabel('Measured Magnitude', fontsize=20)
    aa.set_xticks(np.arange(18, 25, 2))
    plt.setp(plt.gca().get_xticklabels(), fontsize=14)
    plt.setp(plt.gca().get_yticklabels(), fontsize=14)
    plt.savefig('like.png')
    '''
    
    plt.tight_layout()
    return ax


def doLikelihoodPCAfit(pcaComp=None, master=None, eigenval=None, likelihood=None, n_component=5, Lcut=0., residual=False):

    # Perform least-squares: Find the best combination of master + pcaComps[:,:,0:n_component] that fits likelihood
    origShape = likelihood.shape

    if residual:
        L1d = likelihood - master
    else:
        L1d = np.copy(likelihood)
    L1d = likelihood.reshape(likelihood.size)

    pca1d = pcaComp.reshape( ( likelihood.size, pcaComp.shape[-1]) )
    pcafit = pca1d[:,0:(n_component)]
    m1d = np.reshape(master,master.size)
    #allfit = np.hstack((m1d[:,None], pcafit) )
    allfit = pcafit
    coeff, resid, _, _  = np.linalg.lstsq(allfit, L1d)
    bestFit = np.dot(allfit,coeff)
    bestFit2d = bestFit.reshape(likelihood.shape)
    if residual:
        bestFit2d = bestFit2d + master
    bestFit2d[bestFit2d < Lcut] = 0.

    m_coeff, m_resid, _, _ = np.linalg.lstsq(allfit, m1d)
    m1d_fit = np.dot(allfit, m_coeff)
    m2d = np.reshape(m1d_fit,master.shape)
    
    return bestFit2d, m2d

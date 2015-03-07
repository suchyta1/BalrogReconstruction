#!/usr/bin/env python

import desdb
import numpy as np
import esutil
import pyfits
import sys
import healpy as hp

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#import seaborn as sns


def CatMatch(c1, c2, band1, band2):
    radius = 1/3600.0
    m1, m2, d12 = esutil.htm.HTM().match(c1['ra_%s'%(band1)],c1['dec_%s'%(band1)], c2['ra_%s'%(band2)],c2['dec_%s'%(band2)], radius)
    rm2, rm1, rd12 = esutil.htm.HTM().match(c2['ra_%s'%(band2)],c2['dec_%s'%(band2)], c1['ra_%s'%(band1)],c1['dec_%s'%(band1)], radius)

    dtype = [('1',np.float32), ('2', np.float32)]
    m = np.empty( len(m1), dtype=dtype)
    m['1'] = m1
    m['2'] = m2
    r = np.empty( len(rm1), dtype=dtype)
    r['1'] = rm1
    r['2'] = rm2
    cut = np.in1d(m, r)
    return c1[m1[cut]], c2[m2[cut]]


def GetDepthMap(depth_file):
    map = hp.read_map(depth_file, nest=True)
    nside = hp.npix2nside(map.size)
    return map, nside


def GetPhi(ra):
    return ra * np.pi / 180.0

def GetRa(phi):
    return phi*180.0/np.pi

def GetTheta(dec):
    return (90.0 - dec) * np.pi / 180.0

def GetDec(theta):
    return 90.0 - theta*180.0/np.pi

def GetRaDec(theta, phi):
    return [GetRa(phi), GetDec(theta)]

def GetPix(nside, ra, dec, nest=True):
    phi = GetPhi(ra)
    theta = GetTheta(dec)
    pix = hp.ang2pix(nside, theta, phi, nest=nest)
    return pix

def GetDepthCut(map, nside, ra, dec, depth = 50.0):
    pix = GetPix(nside, ra, dec)
    depths = map[pix]
    ok_depths = (depths > 0 ) & (depths < depth)
    #ok_depths = (depths > 0 )
    return ok_depths

def ValidDepth(map, nside, arr, rakey='ra', deckey='dec', depth = 50.0):
    ok_depths = GetDepthCut(map, nside, arr[rakey], arr[deckey], depth = depth)
    arr = arr[ok_depths]
    return arr

def InSurvey(map, nside, ra, dec):
    ok_depths = GetDepthCut(map, nside, ra, dec)
    return ok_depths


def InTile(data, ura, udec, rakey='ra', deckey='dec'):
    inside = (data[rakey] > ura[0]) & (data[rakey] < ura[1]) & (data[deckey] > udec[0]) & (data[deckey] < udec[1])
    return inside

def RemoveTileOverlap(tilestuff, data, col='tilename', rakey='ra', deckey='dec'):
    datatile = data[col]
    tiles = np.unique(datatile)
    keep = np.zeros( len(data), dtype=np.bool_)
    for tile in tiles:
        cut = (datatile==tile)
        entry = tilestuff[tile]
        ura = (entry['urall'], entry['uraur'])
        udec = (entry['udecll'], entry['udecur'])
        u = InTile(data[cut], ura, udec, rakey=rakey, deckey=deckey)
        keep[cut] =  u
    return data[keep]


def in_tile(ra, dec, ura, udec):
    inside = (ra > ura[0]) & (ra < ura[1]) & (dec > udec[0]) & (dec < udec[1])
    return inside

def hpInTiles(tiles, tileinfo, data, depthmap, depth_nside, max, out_nside):
    lims = []
    num = np.empty(len(data))
    for i in range(len(data)):
        ra = data[i][0]
        dec = data[i][1]
        found = np.zeros(len(ra), dtype=np.bool_)
        for tile in tiles:
            entry = tileinfo[tile]
            ura = (entry['urall'], entry['uraur'])
            udec = (entry['udecll'], entry['udecur'])
            inc = (in_tile(ra,dec, ura,udec) & InSurvey(depthmap, depth_nside, ra, dec))
            found = (found | inc )

            if i==0:
                lims.append( [ura, udec] )
        
        num[i] = np.sum(found) / max * hp.nside2pixarea(out_nside, degrees=True)
    return num, lims



def EqualNumBinning(arr, num=5000, join='max'):
    a = np.sort(arr)
    size = len(a)

    r = size % num
    n = size / num
    if r!=0:
        n += 1

    if join=='min' and r!=0:
        a = a[::-1]
    a = np.array_split(a, n)
    if r!=0:
        a[-2] = np.append(a[-2], a[-1])
        a = a[:-1]

    if join=='min' and r!=0:
        a = a[::-1]
        first = -1
        last = 0
    else:
        first = 0
        last = -1
    
    bins = [a[0][first]]
    nums = [len(a[0])]
    for i in range(len(a)-1):
        btwn = (a[i][last] + a[i+1][first]) / 2.0
        bins.append(btwn)
        nums.append(len(a[i+1]))
    bins.append(a[-1][last])

    bins = np.array(bins)
    nums = np.array(nums)
    db = np.diff(bins)
    p = nums / (len(arr) * db)
    return p, bins


def CorrectColorDistribution(balrog_truth, balrog_sim, des):
    '''
    sim_pdf, sim_bins = EqualNumBinning(sim['gr'])
    sim_c = (sim_bins[1:] + sim_bins[:-1]) / 2.0
    plt.figure(2)
    #plt.plot(sim_c, sim_pdf)
    d = np.diff(sim_bins)
    plt.bar(sim_bins[:-1], sim_pdf, width=d)
    '''

    gr_des_bins = np.linspace(-5.5, 0.5, num=50, endpoint=True)
    iz_des_bins = np.linspace(0.1, 5.5, num=50, endpoint=True)

    fig = plt.figure(2, figsize=(12,4))
    ax = fig.add_subplot(1,3, 1)
    des_hist, xbins, ybins = np.histogram2d(des['iz'], des['gr'], bins=[iz_des_bins, gr_des_bins])
    ax.imshow(np.log10(des_hist.transpose()), extent=[0.1,5.5, -5.5,0.5], origin='lower', interpolation='nearest')
    ax.set_title(r'DES Observed')
    ax.set_xlabel(r'i-z')
    ax.set_ylabel(r'g-r')

    ax.plot( [0.0, 2.8], [-2.05, 0.5], color='black')
    ax.set_xlim([0.1,2.9])
    ax.set_ylim([-2.1,0.5])

    ax = fig.add_subplot(1,3, 2)
    sim_hist, xbins, ybins = np.histogram2d(balrog_sim['iz'], balrog_sim['gr'], bins=[iz_des_bins, gr_des_bins])
    ax.imshow(np.log10(sim_hist.transpose()), extent=[0.1,5.5, -5.5,0.5], origin='lower', interpolation='nearest')
    ax.set_title(r'Balrog Observed')
    ax.set_xlabel(r'i-z')
    ax.set_ylabel(r'g-r')

    ax.plot( [0.0, 2.8], [-2.05, 0.5], color='black')
    ax.set_xlim([0.1,2.9])
    ax.set_ylim([-2.1,0.5])

    for i in range(len(sim_hist)):
        for j in range(len(sim_hist[i])):
            if sim_hist[i][j]==0 and des_hist[i][j]!=0:
                print sim_hist[i][j], des_hist[i][j]


    gr_truth_bins = np.linspace(-1, 4, num=80)
    iz_truth_bins = np.linspace(-1, 2, num=80)

    ax = fig.add_subplot(1,3, 3)
    truth_hist, xbins, ybins = np.histogram2d(balrog_sim['truth_iz'], balrog_sim['truth_gr'], bins=[iz_truth_bins, gr_truth_bins])
    ax.imshow(np.log10(truth_hist.transpose()), extent=[-1,2, -1,4], origin='lower', interpolation='nearest')
    ax.set_title(r'Balrog Truth')
    ax.set_xlabel(r'i-z')
    ax.set_ylabel(r'g-r')
    #ax.set_xticks(np.arange(-1,2,1))
    majorLocator = MultipleLocator(1)
    minorLocator = MultipleLocator(0.2)
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.set_ylim([-1, 2.5])

    plt.tight_layout()


def CorrectMags(des_info, balrog_info, truth_info, inds, band='i'):

    mag_bins = np.arange(16,27, 0.1)
    tmag_bins = np.arange(18,24, 0.1)
    des_corr = []
    err_des_corr = []

    for k in range(len(des_info)):
        des = des_info[k]
        balrog = balrog_info[k]
        truth = truth_info[k]

        #m = 'mag_%s' %(band)
        #tm = 'truth_mag_%s' %(band)
        m = 'mag'
        tm = 'truth_mag'

        des_hist_corr = np.zeros( len(tmag_bins)-1 )
        err_des_hist_corr = np.zeros( len(tmag_bins)-1 )
        des_hist, mag_bins = np.histogram(des[m], bins=mag_bins)
        balrog_hist, mag_bins = np.histogram(balrog[m],bins=mag_bins)
        tbalrog_hist, tmag_bins = np.histogram(balrog[tm], bins=tmag_bins)

        for j in range(len(mag_bins)-1):
            des_num = des_hist[j]
            cut = np.zeros(len(balrog), dtype=np.bool_)
            f = 0
            while np.sum(cut)==0:
                dm = f * (mag_bins[j+1] - mag_bins[j])
                cut = (balrog[m] > (mag_bins[j] - dm)) & (balrog[m] < (mag_bins[j+1] + dm))
                f += 1
            balrog_inbin = balrog[cut]
            truth_hist, tmag_bins = np.histogram(balrog_inbin[tm], bins=tmag_bins)

            size =  float(len(balrog_inbin))
            des_hist_corr = des_hist_corr + (truth_hist / size) * des_num
            if des_num > 0:
                frac = np.sqrt( 1.0/des_num + 1.0/size )
            else:
                frac = 0
            err_des_hist_corr = err_des_hist_corr + truth_hist * (frac / size * des_num)
       
        completeness = np.zeros( des_hist_corr.shape )
        err_completeness = np.zeros( err_des_hist_corr.shape )
        p = False
        for j in range(len(tmag_bins)-1):
            f = 0
            cut = np.zeros(len(truth), dtype=np.bool_)
            while np.sum(cut)==0:
                dm = f * (tmag_bins[j+1] - tmag_bins[j])
                cut = (truth[tm] > (tmag_bins[j]-dm)) & (truth[tm] < (tmag_bins[j+1]+dm)) 
                f += 1
                if tmag_bins[j] > 21:
                    break

            den = float( np.sum(cut) )
            n = np.in1d(balrog['balrog_index'], truth[cut]['balrog_index'])
            num = np.sum(n)
            if den > 0:
                comp = num / den
                completeness[j] = comp
                err_completeness[j] = np.sqrt(num)/den
            
            if (completeness[j]==0) and (not p) and tmag_bins[j]>21:
                print k, inds[k], tmag_bins[j], len(des)
                p = True


        corr = np.zeros( des_hist_corr.shape )
        err_corr = np.zeros( des_hist_corr.shape )

        cut = (completeness > 0)
        corr[cut] = des_hist_corr[cut]/completeness[cut]

        cut = (completeness > 0) & (des_hist_corr > 0)
        t1 = err_des_hist_corr[cut]/des_hist_corr[cut]
        t2 = err_completeness[cut]/completeness[cut]
        frac = np.sqrt( t1*t1 + t2*t2 )
        err_corr[cut] = frac * corr[cut]

        des_corr.append(np.sum(corr))
        err_des_corr.append( np.sqrt(sum(err_corr*err_corr)) )

        if k==40:
            plt.figure(10)
            plt.plot(tmag_bins[:-1], des_hist_corr, color='blue')
            plt.plot(mag_bins[:-1], des_hist, color='red')
            plt.plot(mag_bins[:-1], balrog_hist, color='green')
            plt.plot(tmag_bins[:-1], corr, color='cyan')

            plt.figure(11)
            plt.plot(tmag_bins[:-1], completeness, color='blue')

    return des_corr, err_des_corr


def CorrectColors(des_info, balrog_info, truth_info):
    gr_o_bins = np.linspace(-5.5, 0.5, num=50, endpoint=True)
    iz_o_bins = np.linspace(0.1, 5.5, num=50, endpoint=True)
    gr_t_bins = np.linspace(-1, 4, num=80)
    iz_t_bins = np.linspace(-1, 2, num=80)

    des_corr = []
    #plus = 51
    #for k in range(len(des_info[plus:(plus+1)])):
    for k in range(len(des_info)):
        des = des_info[k]
        balrog = balrog_info[k]
        truth = truth_info[k]

        des_hist_corr = np.zeros( (len(iz_t_bins)-1, len(gr_t_bins)-1) )
        des_hist, iz_o_bins, gr_o_bins = np.histogram2d(des['iz'], des['gr'], bins=[iz_o_bins, gr_o_bins])
        balrog_hist, iz_o_bins, gr_o_bins = np.histogram2d(balrog['iz'], balrog['gr'], bins=[iz_o_bins, gr_o_bins])
        tbalrog_hist, iz_t_bins, gr_t_bins = np.histogram2d(balrog['truth_iz'], balrog['truth_gr'], bins=[iz_t_bins, gr_t_bins])

        for j in range(len(iz_o_bins)-1):
            for i in range(len(gr_o_bins)-1):
                des_num = des_hist[j][i]
                cut = np.zeros(len(balrog), dtype=np.bool_)
                f = 0
                while np.sum(cut)==0:
                    d_gr = f * (gr_o_bins[i+1] - gr_o_bins[i])
                    d_iz = f * (iz_o_bins[j+1] - iz_o_bins[j])
                    cut = ( (balrog['gr'] > (gr_o_bins[i] - d_gr)) & (balrog['gr'] < (gr_o_bins[i+1] + d_gr)) & (balrog['iz'] > (iz_o_bins[j] - d_iz)) & (balrog['iz'] < (iz_o_bins[j+1] + d_iz)) )
                    f += 1
                balrog_inbin = balrog[cut]
                truth_hist, iz_t_bins, gr_t_bins = np.histogram2d(balrog_inbin['truth_iz'], balrog_inbin['truth_gr'], bins=[iz_t_bins,gr_t_bins])
                dt_iz = np.diff(iz_t_bins)
                dt_gr = np.diff(gr_t_bins)
                des_hist_corr = des_hist_corr + (truth_hist / float(len(balrog_inbin))) * des_num
       
        completeness = np.zeros( des_hist_corr.shape )
        for j in range(len(iz_t_bins)-1):
            for i in range(len(gr_t_bins)-1):
                cut = (truth['truth_gr'] > gr_t_bins[i]) & (truth['truth_gr'] < gr_t_bins[i+1]) & (truth['truth_iz'] > iz_t_bins[j]) & (truth['truth_iz'] < iz_t_bins[j+1])
                den = float( np.sum(cut) )
                n = np.in1d(balrog['balrog_index'], truth[cut]['balrog_index'])
                num = np.sum(n)
                if den > 0:
                    comp = num / den
                    completeness[j][i] = comp
        
     
        corr = np.zeros( des_hist_corr.shape )
        cut = (completeness > 0)
        corr[cut] = des_hist_corr[cut]/completeness[cut]
        des_corr.append(np.sum(corr))
        #des_hist_corr

        """
        fig = plt.figure(3, figsize=(16,8))
        ax = fig.add_subplot(2,3, 1)
        cax = ax.imshow(np.log10(des_hist.transpose()), extent=[0.1,5.5, -5.5,0.5], origin='lower', interpolation='nearest', cmap=mpl.cm.binary)
        cbar = fig.colorbar(cax)
        
        ax.set_title(r'DES Observed')
        ax.set_xlabel(r'i-z')
        ax.set_ylabel(r'g-r')
        ax.set_xlim([0.1,2.9])
        ax.set_ylim([-2.1,0.5])

        ax = fig.add_subplot(2,3, 2)
        cax = ax.imshow(np.log10(des_hist_corr.transpose()), extent=[-1,2, -1,4], origin='lower', interpolation='nearest', cmap=mpl.cm.binary)
        ax.set_title(r'DES, color corrected')
        ax.set_xlabel(r'i-z')
        ax.set_ylabel(r'g-r')
        majorLocator = MultipleLocator(1)
        minorLocator = MultipleLocator(0.2)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.set_ylim([-1, 2.5])
        cbar = fig.colorbar(cax)

        ax = fig.add_subplot(2,3, 3)
        cax = ax.imshow(np.log10(corr.transpose()), extent=[-1,2, -1,4], origin='lower', interpolation='nearest', cmap=mpl.cm.binary)
        ax.set_title(r'DES, color/comp. corrected')
        ax.set_xlabel(r'i-z')
        ax.set_ylabel(r'g-r')
        majorLocator = MultipleLocator(1)
        minorLocator = MultipleLocator(0.2)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.set_ylim([-1, 2.5])
        cbar = fig.colorbar(cax)

        ax = fig.add_subplot(2,3, 4)
        cax = ax.imshow(np.log10(balrog_hist.transpose()), extent=[0.1,5.5, -5.5,0.5], origin='lower', interpolation='nearest', cmap=mpl.cm.binary)
        ax.set_title(r'Balrog Observed')
        ax.set_xlabel(r'i-z')
        ax.set_ylabel(r'g-r')
        ax.set_xlim([0.1,2.9])
        ax.set_ylim([-2.1,0.5])
        cbar = fig.colorbar(cax)

        ax = fig.add_subplot(2,3, 5)
        cax = ax.imshow(np.log10(tbalrog_hist.transpose()), extent=[-1,2, -1,4], origin='lower', interpolation='nearest', cmap=mpl.cm.binary)
        ax.set_title(r'Balrog Truth')
        ax.set_xlabel(r'i-z')
        ax.set_ylabel(r'g-r')
        majorLocator = MultipleLocator(1)
        minorLocator = MultipleLocator(0.2)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.set_ylim([-1, 2.5])
        cbar = fig.colorbar(cax)

        ax = fig.add_subplot(2,3, 6)
        cax = ax.imshow(np.log10(completeness.transpose()), extent=[-1,2, -1,4], origin='lower', interpolation='nearest', cmap=mpl.cm.binary)
        ax.set_title(r'Completeness')
        ax.set_xlabel(r'i-z')
        ax.set_ylabel(r'g-r')
        majorLocator = MultipleLocator(1)
        minorLocator = MultipleLocator(0.2)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.set_ylim([-1, 2.5])
        cbar = fig.colorbar(cax)

    plt.tight_layout()
        """

    return des_corr



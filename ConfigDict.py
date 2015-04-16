import sys
import numpy as np
import shelve
import time
import os

def BuildDict(band='i', savedir=None):
    if savedir is None:
        savedir = os.path.join(os.environ['GLOBALDIR'], 'saved-runs')

    #version = 'sva1v3_3'
    #version = 'sva1v2'
    version = 'combined'
    size = 0.5
    #size = 0.1
    sg = True

    if band =='i':
        min = 17.5
        max = 25.0
        #max = 27.0
        tbins = np.arange(min, max+size, size)

        min = 17.5
        max = 27.0
        #max = 29.0

        obins = np.arange(min, max+size, size)
        #obins = np.insert(obins, 0, -100)
        #obins = np.insert(obins, len(obins), 100)
        #obins = np.copy(tbins)

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
    tb = [list(np.arange(0.5, 5, 2.0)), tbins]
    mc = ['modtype_%s'%(band), 'mag_auto_%s'%(band)]
    mb = [list(np.arange(0.5, 7, 2.0)), obins]
    if not sg:
        tc = tc[-1:]
        tb = tb[-1:]
        mc = mc[-1:]
        mb = mb[-1:]

    for i in range(len(tb)):
        tb[i] = list(tb[i])
    for i in range(len(mb)):
        mb[i] = list(mb[i])


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
                  'Lcut': 0,

                  'bcut': None,
                  's2n': None,

                  'PCAon': 'healpix',
                  #'PCAon': None,
                  'residual': False,
                  'n_component': 10,
                  'burnin': 3000,
                  'steps': 1000,

                  'nWalkers': 1000,
                  'jackfield': 'jacknife',
                  'jacknife': 1,
                  'jackdes': True
                 }

    MapConfig = {'nside': 64,
                 #'nside': None,
                 'hpfield': 'hpIndex',
                 'nest': False,
                 'summin': 22.5,
                 'summax': 24.5}

    config = [DBselect, MCMCconfig, MapConfig]
    run = Save2Shelf(config)
    config[2]['version'] = run
    config[1]['out'] = os.path.join(savedir, config[2]['version'], 'SGPlots-%s'%(config[2]['version']))
    config[1]['hpjdir'] = os.path.join(savedir, config[2]['version'])

    return config


def Save2Shelf(config, shelfname='runs-shelf'):
    db = shelve.open(shelfname)

    runname = FindEntry(config)
    if runname is None:
        runname = str(int(time.time()))
    print runname
    db[runname] = config
    db.close()
    return runname


def FindEntry(config, shelfname='runs-shelf'):
    db = shelve.open(shelfname)
    k = None
    for key in db.keys():
        if db[key]==config:
            k = key
            break
    db.close()
    return k

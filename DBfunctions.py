import desdb
import numpy as np
import numpy.lib.recfunctions as recfunctions
import healpy as hp
from mpi4py import MPI
import mpifunctions


def AngularDistance(r1, r2, d1, d2):
    ra1, ra2, dec1, dec2 = deg2rad(r1, r2, d1, d2)
    ddec = np.radians(dec2-dec1)
    dra = np.radians(ra2-ra1)
    a = np.sin(ddec/2) * np.sin(ddec/2) + np.cos(dec1) * np.cos(dec2) * np.sin(dra/2) * np.sin(dra/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return np.degrees(c)


def QuerySFW(s, f, w, select, where):
    cur = desdb.connect()
    if where != None:
        for i in range(len(select['bands'])):
            ww = 'truth_%s.%s' %(select['bands'][i],where)
            w.insert(0, ww)

    selects = 'SELECT %s' %(', '.join(s))
    froms = 'FROM %s' %(', '.join(f))
    wheres = 'WHERE %s' %(' and '.join(w))
    q = '%s %s %s' %(selects, froms, wheres)
    return cur.quick(q, array=True)


def TruthSelect(select, where=None):
    f = []
    w = []
    s = []
    band0 = select['bands'][0]
    for i in range(len(select['bands'])):
        band = select['bands'][i]
        f.append('balrog_%s_truth_%s truth_%s' %(select['table'], band, band) )
        for sel in select['truth']:
            s.append('truth_%s.%s as %s_%s' %(band,sel,sel,band))
        if i > 0:
            w.append('truth_%s.balrog_index=truth_%s.balrog_index' %(band0,band) )

    return QuerySFW(s, f, w, select, where)


def SimSelect(select, where=None):
    f = []
    w = []
    s = []
    band0 = select['bands'][0]

    '''
    qq = GetQuery(select, where, kind='truth')
    cur = desdb.connect()
    a = cur.quick(qq, array=True)
    print len(a), a.dtype.names
    sys.exit()
    '''

    for i in range(len(select['bands'])):
        band = select['bands'][i]

        f.append('balrog_%s_sim_%s sim_%s' %(select['table'], band, band) )
        f.append('balrog_%s_truth_%s truth_%s' %(select['table'], band, band) )
        for sel in select['sim']:
            s.append('sim_%s.%s as %s_%s' %(band,sel,sel,band))
        for sel in select['truth']:
            s.append('truth_%s.%s as %s_%s' %(band,sel,sel,band))
        w.append('truth_%s.balrog_index=sim_%s.balrog_index' %(band,band) )
        if i > 0:
            w.append('truth_%s.balrog_index=truth_%s.balrog_index' %(band0,band) )

    return QuerySFW(s, f, w, select, where)


def NosimSelect(select, where=None):
    f = []
    w = []
    s = []

    band0 = select['bands'][0]
    f.append('balrog_%s_nosim_det nosim_det' %(select['table']) )
    for sel in select['sim']:
        s.append('nosim_det.%s as %s_det' %(sel,sel))
    for i in range(len(select['bands'])):
        band = select['bands'][i]
        f.append('balrog_%s_truth_%s truth_%s' %(select['table'], band, band) )
        for sel in select['truth']:
            s.append('truth_%s.%s as %s_%s' %(band,sel,sel,band))
        w.append('truth_%s.balrog_index=nosim_det.balrog_index' %(band) )

    return QuerySFW(s, f, w, select, where)


def GetHealPixRectangles(nside, dbrange, nest):
    hpindex = np.arange(hp.nside2npix(nside))

    vec_corners = hp.boundaries(nside, hpindex, nest=nest)
    vec_corners = np.transpose(vec_corners, (0,2,1))
    vec_corners = np.reshape(vec_corners, (vec_corners.shape[0]*vec_corners.shape[1], vec_corners.shape[2]))
   
    theta_corners, phi_corners = hp.vec2ang(vec_corners)
    theta_corners = np.reshape(theta_corners, (theta_corners.shape[0]/4, 4))
    phi_corners = np.reshape(phi_corners, (phi_corners.shape[0]/4, 4))

    ra_corners = np.degrees(phi_corners)
    dec_corners = 90.0 - np.degrees(theta_corners)

    rainside = ( (ra_corners > dbrange[0]) & (ra_corners < dbrange[1]) )
    rakeep = np.sum(rainside, axis=-1)
    decinside = ( (dec_corners > dbrange[2]) & (dec_corners < dbrange[3]) )
    deckeep = np.sum(decinside, axis=-1)
    keep = ( (rakeep > 0) & (deckeep > 0) )
    ra_corners, dec_corners, hpindex = Cut(ra_corners, dec_corners, hpindex, cut=keep)

    ramin = np.amin(ra_corners, axis=-1)
    ramax = np.amax(ra_corners, axis=-1)
    decmin = np.amin(dec_corners, axis=-1)
    decmax = np.amax(dec_corners, axis=-1)

    return ramin, ramax, decmin, decmax, hpindex


def GetCourseRange(select):
    cur = desdb.connect()
    arr = cur.quick("SELECT min(ra) as ramin, max(ra) as ramax, min(dec) as decmin, max(dec) as decmax FROM balrog_%s_truth_%s" %(select['table'],select['bands'][0]), array=True)
    return arr[0]

def DoCount(ramin, ramax, decmin, decmax, hpindex, select):
    cur = desdb.connect()
    nonzero = np.zeros( len(hpindex), dtype=np.bool_)
    for i in range(len(hpindex)):
        q = "SELECT count(*) as count from balrog_%s_truth_%s where dec between %f and %f and ra between %f and %f" %(select['table'], select['bands'][0], decmin[i],decmax[i],ramin[i],ramax[i])
        arr = cur.quick(q, array=True)
        if arr['count'][0] > 0:
            nonzero[i] = True
    return nonzero


def Cut(*args, **kwargs):
    a = [None]*len(args)
    for i in range(len(args)):
        a[i] = args[i][kwargs['cut']]
    return a



def AppendSelects(ss, select, truthband=None, simband=None, truthlabel='truth', simlabel='sim', simbandappend='', truthbandappend=''):
    for sel in select['truth']:
        ss.append('%s_%s.%s as %s%s' %(truthlabel,truthband,sel,sel,truthbandappend))

    if simlabel != 'truth':
        for sel in select['sim']:
            ss.append('%s_%s.%s as %s%s' %(simlabel,simband,sel,sel,simbandappend))
    return ss


def GetQuery(select, truthwhere, simwhere, kind='sim'):
    js = []
    ss = []
    bands = select['bands']
    for i in range(len(bands)):
        truthband = bands[i]
        simband = bands[i]
        truthbandappend = '_%s' %(bands[i])
        simbandappend = '_%s' %(bands[i])
        simlabel = 'joined'
        if kind=='nosim':
            simband = 'det'
            if i==0:
                simbandappend = '_det'
            else:
                simlabel = 'truth'
        elif kind=='truth':
            simlabel = 'truth'

        bq = Join2Truth(select, bands[i], simband, truthwhere, simwhere, kind=kind)
        ss = AppendSelects(ss, select, truthband=bands[i], simband=bands[i], truthlabel='joined', simlabel=simlabel, truthbandappend=truthbandappend, simbandappend=simbandappend)
        if i==0:
            f = """FROM (%s) joined_%s"""%(bq, bands[i])
        else:
            js.append("""JOIN (%s) joined_%s ON joined_%s.balrog_index=joined_%s.balrog_index"""%(bq, bands[i], bands[i], bands[0]))

    q = 'SELECT %s %s %s' %(', '.join(ss), f, ' '.join(js))
    return q


def Join2Truth(select, truthband, simband, truthwhere, simwhere, kind='sim'):
    ss = []
    ss = AppendSelects(ss, select, truthband=truthband, simband=simband, truthlabel='truth', simlabel=kind)
    ss = ', '.join(ss)
    f = """(SELECT %s FROM balrog_%s_truth_%s %s) truth_%s"""%(', '.join(select['truth']), select['table'],truthband, truthwhere, truthband)
    q = """SELECT %s FROM %s""" %(ss, f)
    if kind in ['sim','nosim']:
        j = """(SELECT %s, balrog_index FROM balrog_%s_%s_%s %s) %s_%s"""%(', '.join(select['sim']), select['table'], kind, simband, simwhere, kind, simband)
        o = """truth_%s.balrog_index=%s_%s.balrog_index""" %(truthband,kind,simband)
        q = """%s JOIN %s ON %s"""%(q,j,o)
    return q


def GetBalrog(select, truthwhere='', simwhere=''):
    cur = desdb.connect()

    q = GetQuery(select, truthwhere, simwhere, kind='truth')
    truth = cur.quick(q, array=True)

    q = GetQuery(select, truthwhere, simwhere, kind='sim')
    sim = cur.quick(q, array=True)

    q = GetQuery(select, truthwhere, simwhere, kind='nosim')
    nosim = cur.quick(q, array=True)
    
    return [truth, sim, nosim]


def GetDES(select, where=''):
    cur = desdb.connect()
    ss = []
    for band in select['bands']:
        for sel in select['sim']:
            ss.append('%s_%s' %(sel,band))
    ss = ', '.join(ss)
    q = """SELECT %s FROM %s %s"""%(ss, select['des'], where)
    des = cur.quick(q, array=True)
    return des


def GetAllViaTileQuery(select, limit=None):
    cur = desdb.connect()
    if MPI.COMM_WORLD.Get_rank()==0:
        arr = cur.quick('SELECT unique(tilename) from balrog_%s_truth_%s' %(select['table'],select['bands'][0]), array=True)
        tiles = arr['tilename']
        if limit!=None:
            tiles = tiles[0:limit]
        print len(tiles), 'tiles'
    else:
        tiles = None

    tiles = mpifunctions.Scatter(tiles)
    arr = []
    for tile in tiles:
        where = "where tilename='%s'"%(tile)
        truth, sim, nosim = GetBalrog(select, truthwhere=where, simwhere=where)
        des = GetDES(select, where=where)
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



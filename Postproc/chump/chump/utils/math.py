import numpy as np
import pandas as pd
import scipy.stats as st

#%% iwfm related

def krige_interpolation_factors(NVertex, XP, YP, X, Y):
    """calculate the simple kriging interpolation factors for a given vertex (XP, YP) with the observations at (X, Y)
    """

    # distance matrix
    X = np.asarray(X)
    Y = np.asarray(Y)

    # RHS
    b = np.sqrt((X - XP) ** 2 + (Y - YP) ** 2)
    if any(np.isclose(b, 0)):
        weights = np.zeros(NVertex)
        weights[np.isclose(b, 0)] = 1 / sum(np.isclose(b, 0))
        return weights

    # distance matrix
    d = np.zeros((NVertex, NVertex))
    for i in range(NVertex):
        for j in range(i, NVertex):
            d[i, j] = np.sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
        for j in range(i):
            d[i, j] = d[j, i]

    # correlation matrix (spherical model)
    a = np.nanmax(d)
    A = np.where(d <= a, 1.5 * d/a - (0.5*d/a) ** 3,  a)


    # unbias condition
    A = np.insert(A, NVertex, 1, axis=0)
    A = np.insert(A, NVertex, 1, axis=1)
    A[-1, -1] = 0

    b = np.insert(b , NVertex, 1)
    l = np.linalg.solve(A, b)[:-1]

    # sum of Lagrange multipliers
    l[l<0] = 0

    # kriging interpolation factors
    f = l / np.sum(l)
    return f



def InterpolationCoeffs(NVertex, XP, YP, X, Y):

    # Initialize
    Coeff = np.zeros(NVertex)

    # Triangular element
    if (NVertex == 3):
        XIJ = X[0]-X[1]
        XJK = X[1]-X[2]
        XKI = X[2]-X[0]
        YIJ = Y[0]-Y[1]
        YJK = Y[1]-Y[2]
        YKI = Y[2]-Y[0]
        XT = (-X[0]*Y[2]+X[2]*Y[0]+YKI*XP-XKI*YP)/(-XKI*YJK+XJK*YKI)
        XT = max(0.0, XT)
        YT = (X[0]*Y[1]-X[1]*Y[0]+YIJ*XP-XIJ*YP)/(-XKI*YJK+XJK*YKI)
        YT = max(0.0, YT)
        Coeff[0] = 1.0-min(1.0, XT+YT)
        Coeff[1] = XT
        Coeff[2] = YT

    # Quadrilateral element
    elif NVertex==4:
        A = (Y[0]-Y[1])*(X[2]-X[3])-(Y[2]-Y[3])*(X[0]-X[1])
        BO = Y[0]*X[3]-Y[1]*X[2]+Y[2]*X[1]-Y[3]*X[0]
        BX = -Y[0]+Y[1]-Y[2]+Y[3]
        BY = X[0]-X[1]+X[2]-X[3]
        B = BO+BX*XP+BY*YP
        CO = -(Y[0]+Y[1])*(X[2]+X[3])+(Y[2]+Y[3])*(X[0]+X[1])
        CX = Y[0]+Y[1]-Y[2]-Y[3]
        CY = -X[0]-X[1]+X[2]+X[3]
        C = CO+2.0*(CX*XP+CY*YP)
        if (A == 0.0):
            if (B == 0.0):
                return Coeff
            XT = -C/(2.0*B)
        else:
            XT = B*B-A*C
            if (XT < 0.0):
                return Coeff
            XT = (-B+(XT)**0.5)/A

        XT = max(-1.0, min(XT, 1.0))
        A = (Y[1]-Y[2])*(X[0]-X[3])-(Y[0]-Y[3])*(X[1]-X[2])
        BO = Y[0]*X[1]-Y[1]*X[0]+Y[2]*X[3]-Y[3]*X[2]
        BX = -Y[0]+Y[1]-Y[2]+Y[3]
        BY = X[0]-X[1]+X[2]-X[3]
        B = BO+BX*XP+BY*YP
        CO = -(Y[0]+Y[3])*(X[1]+X[2])+(Y[1]+Y[2])*(X[0]+X[3])
        CX = Y[0]-Y[1]-Y[2]+Y[3]
        CY = -X[0]+X[1]+X[2]-X[3]
        C = CO+2.0*(CX*XP+CY*YP)
        if (A == 0.0):
            if (B == 0.0):
                return Coeff
            YT = -C/(2.0*B)
        else:
            YT = B*B-A*C
            if (YT < 0.0):
                return Coeff
            YT = (-B-(YT)**0.5)/A

        YT = max(-1.0, min(YT, 1.0))
        Coeff[0] = 0.25*(1.0-XT)*(1.0-YT)
        Coeff[1] = 0.25*(1.0+XT)*(1.0-YT)
        Coeff[2] = 0.25*(1.0+XT)*(1.0+YT)
        Coeff[3] = 0.25*(1.0-XT)*(1.0+YT)

    else:
        Coeff = krige_interpolation_factors(NVertex, XP, YP, X, Y)
    return Coeff

def triarea(xyt):
    return np.abs( xyt[0][0] * (xyt[1][1]-xyt[2][1]) + xyt[1][0] * (xyt[2][1] - xyt[0][1]) + xyt[2][0] * (xyt[0][1] - xyt[1][1])) * 0.5

def QUAD_SHAPE(XI,ETA,NSHP=2):
    if (NSHP==1): return 0.25*(XI-1.0)*(ETA-1.0)
    if (NSHP==2): return 0.25*(XI+1.0)*(1.0-ETA)
    if (NSHP==3): return 0.25*(XI+1.0)*(ETA+1.0)
    if (NSHP==4): return 0.25*(1.0-XI)*(ETA+1.0)

def DXI_QUAD_SHAPE(ETA,NSHP=2):
    if (NSHP==1): return 0.25*(ETA-1.0)
    if (NSHP==2): return 0.25*(1.0-ETA)
    if (NSHP==3): return 0.25*(ETA+1.0)
    if (NSHP==4): return -0.25*(ETA+1.0)

def DETA_QUAD_SHAPE(XI,NSHP=2):
    if (NSHP==1): return 0.25*(XI-1.0)
    if (NSHP==2): return -0.25*(XI+1.0)
    if (NSHP==3): return 0.25*(XI+1.0)
    if (NSHP==4): return 0.25*(1.0-XI)


def D_DXI(ETA,P):
    return sum([P[INDX]*DXI_QUAD_SHAPE(ETA,INDX+1) for INDX in range(4)])

def D_DETA(XI,P):
    return sum([P[INDX]*DETA_QUAD_SHAPE(XI,INDX+1) for INDX in range(4)])

def DET_JACOB(XI,ETA,XP,YP):
    return D_DXI(ETA,XP)*D_DETA(XI,YP) - D_DXI(ETA,YP)*D_DETA(XI,XP)

def QUAD_FUNC_AREA(I,J,XI,ETA,XP,YP):
    return QUAD_SHAPE(XI,ETA,I) * DET_JACOB(XI,ETA,XP,YP)


def QUAD_INTGRL(I,J,XP,YP,NPOINT=2,FUNC=QUAD_FUNC_AREA):


    #2-POINT GAUSSIAN QUADRATURE
    if (NPOINT==2):
      VPOINT=1./3.**0.5
      return  FUNC(I,J,VPOINT,VPOINT,XP,YP)   \
            + FUNC(I,J,VPOINT,-VPOINT,XP,YP)  \
            + FUNC(I,J,-VPOINT,VPOINT,XP,YP)  \
            + FUNC(I,J,-VPOINT,-VPOINT,XP,YP)

    #3-POINT GAUSSIAN QUADRATURE
    if (NPOINT==3):
      VPOINT1=0
      VPOINT2=0.774596669241483
      W1=0.8888888888888889
      W2=0.5555555555555556
      return  W1*W1*QUAD_FUNC_AREA(I,J,VPOINT1,VPOINT1,XP,YP)   \
            + W1*W2*FUNC(I,J,VPOINT1,VPOINT2,XP,YP)   \
            + W1*W2*FUNC(I,J,VPOINT1,-VPOINT2,XP,YP)  \
            + W2*W1*FUNC(I,J,VPOINT2,VPOINT1,XP,YP)   \
            + W2*W2*FUNC(I,J,VPOINT2,VPOINT2,XP,YP)   \
            + W2*W2*FUNC(I,J,VPOINT2,-VPOINT2,XP,YP)  \
            + W2*W1*FUNC(I,J,-VPOINT2,VPOINT1,XP,YP)  \
            + W2*W2*FUNC(I,J,-VPOINT2,VPOINT2,XP,YP)  \
            + W2*W2*FUNC(I,J,-VPOINT2,-VPOINT2,XP,YP)

    #4-POINT GAUSSIAN QUADRATURE
    if (NPOINT==4):
      VPOINT1=0.339981043584856
      VPOINT2=0.861136311594053
      W1=0.652145154862546
      W2=0.347854845137454
      return  W1*W1*QUAD_FUNC_AREA(I,J,VPOINT1,VPOINT1,XP,YP)   \
            + W1*W1*FUNC(I,J,VPOINT1,-VPOINT1,XP,YP)  \
            + W1*W2*FUNC(I,J,VPOINT1,VPOINT2,XP,YP)   \
            + W1*W2*FUNC(I,J,VPOINT1,-VPOINT2,XP,YP)  \
            + W1*W1*FUNC(I,J,-VPOINT1,VPOINT1,XP,YP)  \
            + W1*W1*FUNC(I,J,-VPOINT1,-VPOINT1,XP,YP) \
            + W1*W2*FUNC(I,J,-VPOINT1,VPOINT2,XP,YP)  \
            + W1*W2*FUNC(I,J,-VPOINT1,-VPOINT2,XP,YP) \
            + W2*W1*FUNC(I,J,VPOINT2,VPOINT1,XP,YP)   \
            + W2*W1*FUNC(I,J,VPOINT2,-VPOINT1,XP,YP)  \
            + W2*W2*FUNC(I,J,VPOINT2,VPOINT2,XP,YP)   \
            + W2*W2*FUNC(I,J,VPOINT2,-VPOINT2,XP,YP)  \
            + W2*W1*FUNC(I,J,-VPOINT2,VPOINT1,XP,YP)  \
            + W2*W1*FUNC(I,J,-VPOINT2,-VPOINT1,XP,YP) \
            + W2*W2*FUNC(I,J,-VPOINT2,VPOINT2,XP,YP)  \
            + W2*W2*FUNC(I,J,-VPOINT2,-VPOINT2,XP,YP)


#%%



def bilinear_interpolation_factors(xx, yy, dx, dy, ilay=None, ibound=None):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y).
    The four points can be in any order.  They should form a rectangle.

        Example:
        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4),
        ...                         (20, 4),
        ...                         (10, 6),
        ...                         (20, 6)])

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    xx, yy, dx, dy = np.array(xx), np.array(yy), np.array(dx), np.array(dy)
    xmax = sum(dx)
    ymax = sum(dy)
    inside_domain = (xx>0) & (xx<xmax) & (yy>0) & (yy<ymax)
    if ilay is None:
        # single layer
        ilay = 1
        nl = 1
    else:
        ilay = np.array(ilay)
        nl = max(ilay)

    nc = len(dx)
    nr = len(dy)

    if ibound is None:
        ibound = np.ones([nl, nr, nc])
    # else:
    ibound = np.where(ibound==0, 0, 1)

    xvertices = np.cumsum([0] + list(dx))
    yvertices = np.cumsum([0] + list(dy[::-1]))[::-1]
    xcenter = (xvertices[:-1] + xvertices[1:])/2
    ycenter = (yvertices[:-1] + yvertices[1:])/2

    wellprop = pd.DataFrame()

    ir = np.searchsorted(-yvertices, -yy)
    ic = np.searchsorted( xvertices,  xx)

    wellprop.loc[:, 'Row']    = np.where(ir < nr, ir, nr)
    wellprop.loc[:, 'Column'] = np.where(ic < nc, ic, nc)
    wellprop.loc[inside_domain, 'nInterp'] = 4
    # wellprop.loc[inside_domain, 'avgMethod'] = 'layer'


    ir = np.searchsorted(-ycenter, -yy)
    ic = np.searchsorted( xcenter,  xx)


    ic0 = np.array([i-1 if i>0 else 0 for i in ic ])
    ic1 = np.array([i if i<nc else nc-1 for i in ic ])
    ir0 = np.array([i-1 if i>0 else 0 for i in ir ])
    ir1 = np.array([i if i<nr else nr-1 for i in ir ])

    x1 = xcenter[ic0]
    x2 = xcenter[ic1]

    y1 = ycenter[ir0]
    y2 = ycenter[ir1]

    coeff11 = (x2 - xx) * (y2 - yy) * ibound[ilay-1, ir0, ic0]
    coeff21 = (xx - x1) * (y2 - yy) * ibound[ilay-1, ir0, ic1]
    coeff12 = (x2 - xx) * (yy - y1) * ibound[ilay-1, ir1, ic0]
    coeff22 = (xx - x1) * (yy - y1) * ibound[ilay-1, ir1, ic1]


    wellprop.loc[inside_domain, 'ibound1'] = ibound[ilay-1, ir0, ic0][inside_domain]
    wellprop.loc[inside_domain, 'ibound2'] = ibound[ilay-1, ir0, ic1][inside_domain]
    wellprop.loc[inside_domain, 'ibound3'] = ibound[ilay-1, ir1, ic0][inside_domain]
    wellprop.loc[inside_domain, 'ibound4'] = ibound[ilay-1, ir1, ic1][inside_domain]

    inside_domain = inside_domain & (wellprop[['ibound1', 'ibound2', 'ibound3', 'ibound4']]>0).any(axis=1)

    wellprop.loc[~inside_domain, 'ibound1'] = -999
    wellprop.loc[~inside_domain, 'ibound2'] = -999
    wellprop.loc[~inside_domain, 'ibound3'] = -999
    wellprop.loc[~inside_domain, 'ibound4'] = -999

    wellprop.loc[inside_domain, 'Column1'] = (ic0 + 1)[inside_domain]
    wellprop.loc[inside_domain, 'Column2'] = (ic1 + 1)[inside_domain]
    wellprop.loc[inside_domain, 'Column3'] = (ic0 + 1)[inside_domain]
    wellprop.loc[inside_domain, 'Column4'] = (ic1 + 1)[inside_domain]

    wellprop.loc[inside_domain, 'Row1'] = (ir0 + 1)[inside_domain]
    wellprop.loc[inside_domain, 'Row2'] = (ir0 + 1)[inside_domain]
    wellprop.loc[inside_domain, 'Row3'] = (ir1 + 1)[inside_domain]
    wellprop.loc[inside_domain, 'Row4'] = (ir1 + 1)[inside_domain]

    wellprop.loc[inside_domain, 'Weight1'] = coeff11[inside_domain]
    wellprop.loc[inside_domain, 'Weight2'] = coeff21[inside_domain]
    wellprop.loc[inside_domain, 'Weight3'] = coeff12[inside_domain]
    wellprop.loc[inside_domain, 'Weight4'] = coeff22[inside_domain]
    weightsum = wellprop.loc[:, 'Weight1'] + wellprop.loc[:, 'Weight2'] + wellprop.loc[:, 'Weight3'] + wellprop.loc[:, 'Weight4']
    wellprop.loc[:, 'Weight1'] /= weightsum
    wellprop.loc[:, 'Weight2'] /= weightsum
    wellprop.loc[:, 'Weight3'] /= weightsum
    wellprop.loc[:, 'Weight4'] /= weightsum
    return wellprop

def merr(e):
    return e.mean()

def mae(e):
    return e.abs().mean()

def std(e):
    return e.std()

def maxae(e):
    return e.abs().max()

def rmse(e):
    return np.sqrt((e**2).mean())


# def merr(obs, sim):
#     return (sim - obs).mean()

# def mae(obs, sim):
#     return (sim - obs).abs().mean()

# def rmse(obs, sim):
#     return np.sqrt(((sim - obs)**2).mean())

# def r2(obs, sim):
#     n = obs.shape[0]
#     obsum = obs.sum()
#     simsum = sim.sum()
#     return (n * (sim * obs).sum() - obsum * simsum) ** 2 / ((n * (obs ** 2).sum() - obsum ** 2) * (n * (sim ** 2).sum() - simsum ** 2))

def r2(obs, sim):
    assert len(obs) == len(sim), 'r2: Lengths are different between obs and sim'
    if len(obs)<3:
        return None
    o = obs - np.nanmean(obs)
    s = sim - np.nanmean(sim)
    return np.nansum(o * s) ** 2 / np.nansum(o ** 2) / np.nansum(s ** 2)

def agreement_index(obs, sim):
    obsmean = np.nanmean(obs)
    return 1.0 - np.nansum((obs - sim) ** 2) / np.nansum((np.abs(sim-obsmean)+np.abs(obs-obsmean))**2)

def cal_sim_stat(obs, sim):
    obs.name = 'obs'
    sim.name = 'sim'

    outer = pd.concat([obs, sim], axis=1).sort_index()
    outer.sim = outer.sim.interpolate()
    outer['err'] = outer.sim - outer.obs
    inner = outer.dropna(subset=['obs'])

    # mean error
    merr = inner.loc[:, 'err'].mean()

    # abs mean error
    mae = inner.loc[:, 'err'].abs().mean()

    # root mean square
    rmse = np.sqrt((inner.loc[:, 'err'] ** 2).mean())

    # r2
    if inner.shape[0] > 2:
        r2 = ((inner.obs - inner.obs.mean()) * (inner.sim - inner.sim.mean())).sum() ** 2 / ((inner.obs - inner.obs.mean()) ** 2).sum() / ((inner.sim - inner.sim.mean()) ** 2).sum()
        # trend in error
        days = inner.index
        z = np.polyfit(days, inner.err, 1)
        p = np.poly1d(z)
        trend = z[0] * 365.25
        days = outer.index
        outer['errtrend'] = p(days)
    else:
        r2 = None
        trend= None
        outer['errtrend'] = pd.NA
    return merr, mae, rmse, r2, trend, outer


def predict_interval_orig(x, y, predict=False, y_as_error=True, confidence=0.95):
    # predict == True will give prediction interval otherwise it is confidence intervel
    x = np.array(x)
    err = np.array(y) if y_as_error else np.array(y)-x
    n = len(err)
    sse = np.sum(err ** 2)
    mse = sse / (n - 2)
    tval = st.t.ppf((1 + confidence) / 2., n - 2)
    fit = 1 if predict else 0
    std = (x - np.mean(x))**2
    fit = (fit + (1 / n) + std / np.sum(std))
    return tval * np.sqrt(mse * fit)

def predict_interval(x, y, y_as_error=True, confidence=0.95):
    x = np.array(x)
    err = np.array(y) if y_as_error else np.array(y)-x
    n = len(err)
    sse = np.sum(err ** 2)
    mse = sse / (n - 2)
    tval = st.t.ppf((1 + confidence) / 2., n - 2)
    std = (x - np.mean(x))**2
    fit_ci =     (1 / n) + std / np.sum(std)
    fit_pi = 1 + fit_ci
    return tval * np.sqrt(mse * fit_ci), tval * np.sqrt(mse * fit_pi)


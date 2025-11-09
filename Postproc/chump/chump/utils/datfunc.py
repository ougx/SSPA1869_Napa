import importlib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from .misc import is_true, iterate, get_index_names, iterate_dicts, merge_id_df

max_numeric_cols = 100

def numeric_cols(obj, ):
    """@private"""
    # check if there is non-numeric
    return (slice(None) if obj._dat.shape[1]>max_numeric_cols else
        [dtype.kind in 'biufc' for dtype in obj._dat.dtypes]
    )


def aggfunc(obj, numericfunc, otherfunc='first'):
    """@private"""
    # check if there is non-numeric
    return {c:(numericfunc if obj._dat[c].dtype.kind in 'biufc' else otherfunc)
            for c in obj._dat.columns} if obj._dat.shape[1] < max_numeric_cols else numericfunc


def exttable(obj, key, on=None):
    on = on or obj.idcol
    exttables = obj.dict[key]
    exttable = []
    if is_true(exttables):
        for kkv in iterate_dicts(exttables):
            for k, kv in kkv.items():
                for v in iterate(kv):
                    tab = getattr(obj.parent, k)[v]
                    if tab.idcol is not None:
                        obj.dict['idcol'] = tab.idcol
                    exttable.append(tab.dat.drop('geometry', axis=1, errors='ignore').reset_index())
        obj._dat = merge_id_df(obj._dat, pd.concat(exttable), on=on)
        obj.exttable = None


def transpose(obj, key):
    if obj.dict[key]:
        obj._dat = obj._dat.T


def countlimit(obj, key):
    if obj.dict[key] > 1:
        count = obj._dat.groupby(level=0)[obj._dat.columns[0]].count()
        count = count[count>=obj.countlimit]
        obj._dat = obj._dat.loc[count.index]


def drop(obj, key):
    drops = {k:v for k,v in obj.dict[key].items()}
    drops['errors']='ignore'
    obj._dat = obj._dat.drop(**drops,  )


def filter(obj, key):
    obj._dat = obj._dat.query(obj.dict[key])


def exclude(obj, key):
    obj.exclude = obj._dat.query(obj.dict[key])
    obj._dat = obj._dat[~obj._dat.index.isin(obj.exclude.index)]
    if obj.exclude.shape[0] == 0:
        obj.exclude = None


def minlayer(obj, key):
    if 'layer' in get_index_names(obj._dat) and obj.dict[key] > 0:
        layers = obj._dat.index.get_level_values('layer')
        obj._dat  = obj._dat[layers>=obj.dict[key]]


def maxlayer(obj, key):
    if 'layer' in get_index_names(obj._dat) and obj.dict[key] < 9999:
        layers = obj._dat.index.get_level_values('layer')
        obj._dat  = obj._dat[layers<=obj.dict[key]]


def lfunc(obj, key):
    fc = {c:(obj.dict[key] if obj._dat[c].dtype.kind in 'biufc' else 'first')
            for c in obj._dat.columns}
    index_names = get_index_names(obj._dat)
    nas = obj._dat.isna().all(axis=0)
    if len(index_names)>1:
        otherlevels = np.setdiff1d(index_names, ['layer'])
        obj._dat  = obj._dat.groupby(level=otherlevels).agg(fc)
    else:
        obj._dat  = obj._dat.agg(obj.dict[key])
    obj._dat.loc[:,nas.values] = np.nan


def mintime(obj, key):
    dat = obj.dat.reset_index()
    if obj.timecol in dat.columns:
        times = dat[obj.timecol].values
        mintime = obj.dict[key]
        if obj.start_date:
            if is_numeric_dtype(mintime):
                mintime = obj.start_date + pd.Timedelta(mintime, obj.time_unit)
        if not is_numeric_dtype(times):
            mintime = pd.Timestamp(mintime)
        obj._dat  = obj._dat.loc[times>=mintime].copy()


def maxtime(obj, key):
    dat = obj.dat.reset_index()
    if obj.timecol in dat.columns:
        times = dat[obj.timecol].values
        maxtime = obj.dict[key]
        if obj.start_date:
            if is_numeric_dtype(maxtime):
                maxtime = obj.start_date + pd.Timedelta(maxtime, obj.time_unit)
        if not is_numeric_dtype(times):
            maxtime = pd.Timestamp(maxtime)
        obj._dat  = obj._dat.loc[times<=maxtime].copy()


def tfunc(obj, key):
    index_names = get_index_names(obj._dat)
    nas = obj._dat.isna().all(axis=0)
    if len(index_names)>1:
        otherlevels = np.setdiff1d(index_names, ['time'])
        obj._dat  = obj._dat.groupby(level=otherlevels).agg(aggfunc(obj, obj.dict[key]))
    else:
        obj._dat  = obj._dat.agg(obj.dict[key])
    obj._dat.loc[:,nas.values] = np.nan


def a_scale(obj, key):
    obj._dat.loc[:,numeric_cols(obj)] *= float(obj.dict[key])


def a_offset(obj, key):
    obj._dat.loc[:,numeric_cols(obj)] += float(obj.dict[key])


def resample(obj, key):
    index_names = get_index_names(obj._dat)
    nas = obj._dat.isna().all(axis=0)
    for rr in iterate_dicts(obj.dict[key]):
        r = rr.copy()
        resamplefunc = r.pop('func')
        if len(index_names)>1:
            if 'level' not in r:
                r['level'] = obj.timecol
            otherlevels = np.setdiff1d(index_names, r['level'])

            obj._dat = obj._dat.groupby(level=otherlevels).resample(
                **r).agg(aggfunc(obj, resamplefunc))
        else:
            obj._dat = obj._dat.resample(**r).agg(aggfunc(obj, resamplefunc))
    obj._dat.loc[:,nas.values] = np.nan



def rolling(obj, key):
    for rr in iterate_dicts(obj.dict[key]):
        r = rr.copy()
        resamplefunc = r.pop('func')
        obj._dat = obj._dat.rolling(**r).agg(aggfunc(obj, resamplefunc))


def subtract(obj, key):
    for vv in iterate_dicts(obj.dict[key]):
        v = vv.copy()
        other = obj._getobj(v.pop('other'))
        obj._dat = obj._dat.subtract(other.dat, **v)


def add(obj, key):
    for vv in iterate_dicts(obj.dict[key]):
        v = vv.copy()
        other = obj._getobj(v.pop('other'))
        obj._dat = obj._dat.add(other.dat, **v)


def sub(obj, key):
    for vv in iterate_dicts(obj.dict[key]):
        v = vv.copy()
        other = obj._getobj(v.pop('other'))
        obj._dat = obj._dat.sub(other.dat, **v)


def mul(obj, key):
    for vv in iterate_dicts(obj.dict[key]):
        v = vv.copy()
        other = obj._getobj(v.pop('other'))
        obj._dat = obj._dat.mul(other.dat, **v)


def div(obj, key):
    for vv in iterate_dicts(obj.dict[key]):
        v = vv.copy()
        other = obj._getobj(v.pop('other'))
        obj._dat = obj._dat.div(other.dat, **v)


def rename(obj, key):
    rr = {k:v for k,v in obj.dict[key].items()}
    rr['errors'] = 'ignore'
    obj._dat = obj._dat.rename(**obj.rename, )
    for k, v in obj.rename.items():
        if k == 'columns':
            for kk, vv in v.items():
                if kk in obj.labels:
                    obj._labels[obj._labels.index(kk)] = vv


def limits(obj, key):
    nc = numeric_cols(obj)
    dat = obj._dat.loc[:,nc]
    limits = obj.dict[key]
    obj._dat.loc[:,nc] = np.where(
        (dat>=limits[0]) & (dat<=limits[1]),
        dat,
        np.nan
    )


def abslimits(obj, key):
    nc = numeric_cols(obj)
    dat = obj._dat.loc[:,nc]
    limits = obj.dict[key]
    absdat = dat.abs()
    obj._dat.loc[:,nc] = np.where((absdat>=limits[0]) & (absdat<=limits[1]), dat, np.nan)


def fillna(obj, key):
    if isinstance(obj.dict[key], dict):
        obj._dat = obj._dat.fillna(**obj.dict[key])
    else:
        obj._dat = obj._dat.fillna(obj.dict[key])


def replace(obj, key):
    re = {}
    for k, v in obj.dict[key].items():
        if k.lower() == 'nan':
            k = np.nan
        if isinstance(v, str):
            if v.lower() == 'nan':
                v = np.nan

        if isinstance(v, dict):
            re[k] = {}
            for kk, vv in v.items():
                if kk.lower() == 'nan':
                    kk = np.nan
                if isinstance(v, str):
                    if vv.lower() == 'nan':
                        vv = np.nan
                re[k][kk] = vv
        else:
            re[k] = v
    index_names = get_index_names(obj._dat)
    obj._dat = obj._dat.reset_index().replace(re, ).set_index(index_names)


def rate_to_vol(obj, key):
    if not obj.dict[key]:
        return
    times = obj._dat.reset_index()[obj.timecol]
    if not is_numeric_dtype(times[0]):
        if obj.start_date is None:
            raise ValueError(f'`start_date` must be defined for {obj.fullname} to calculate the time step length for the first time step')
        times = [(t - obj.start_date)/pd.Timedelta(1, obj.time_unit) for t in iterate(times)]
    obj._dat['time999__@'] = times
    vals = []
    for w, df in obj._dat.groupby(level=0):
        times  = df['time999__@'].values
        dt     = np.zeros(len(times))
        dt[1:] = np.diff(times)
        dt[0]  = times[0]
        vals.append(df[obj.labels].values * dt.reshape([-1, 1]))
    obj._dat[obj.labels] = np.vstack(vals)
    obj._dat.drop('time999__@', axis=1, inplace=True)


def rate_to_cumvol(obj, key):
    if not obj.dict[key]:
        return
    rate_to_vol(obj, key)
    obj._dat = obj._dat.groupby(level=0)[obj.labels].cumsum()


def dfopts(obj, key):
    for d in iterate_dicts(obj.dict[key]):
        for k, v in d.items():
            obj._dat = getattr(obj._dat, k)(**v)


def to_csv(obj, key):
    obj.dat.to_csv(obj.dict[key])


def join(obj, key):
    obj.dat.to_csv(obj.dict[key])


def extfunc(obj, key):
    extfunc = importlib.import_module(obj.dict[key].strip().rstrip('.py'))
    obj._dat = extfunc.extfunc(obj)


def writefunc(obj, key):
    writefunc = importlib.import_module(obj.dict[key].strip().rstrip('.py'))
    writefunc.writefunc(obj)

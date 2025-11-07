import numpy as np
import pandas as pd
import traceback
from pandas.api.types import is_datetime64_any_dtype as is_datetime
#%%
def warn_deprecation(obj, msg):
    print(f'\n * Deprecation warning regarding `{obj.name}`')
    print(msg)
    print('\n')

def print_callstack():
    for line in traceback.format_stack():
        print('   ' + line.strip())

def get_unique_unsorted(a):
    return np.array(a)[np.sort(np.unique(a, return_index=True)[1])]
#%%
def is_true(v):
    if v is None: return False
    if isinstance(v, bool): return v
    if isinstance(v, str):
        if v.strip().lower() in ('', 'false', 'f', 'no', 'n', 'null', 'not'):
            return False

    if isinstance(v, int) or isinstance(v, float):
        return v != 0
    try:
        return len(v) > 0
    except:
        return bool(v)

def iterate(v):
    if type(v) is str or type(v) is int or type(v) is float:
        return [v]
    try:
        return list(v)
    except:
        return [v]

def iterate_dicts(v):
    if isinstance(v, dict):
        return [v]
    try:
        return list(v)
    except:
        return [v]

def concat_dicts(dict2, dict1, ):
    """combine dicts. if identical key existing in two dicts, the below these keys will be combined.

    Args:
        dict1 (_type_): main dict
        dict2 (_type_): secondary dict

    Returns:
        _type_: new dict combine two dicts
    """
    dict_ = {k:v for k, v in dict1.items()}
    for k,v2 in dict2.items():
        if k in dict1:
            v1 = dict1[k]
            if isinstance(v1, dict) and isinstance(v2, dict):
                dict_[k].update(v2)
            else:
                dict_[k] = v1   # if not dicts, keep the first one
        else:
            dict_[k] = v2
    return dict_
#%% index related
def get_index_names(df):
    if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        if df.index.nlevels >1:
            return list(df.index.names)
        else:
            return [df.index.name]
    else:
        raise TypeError('Not DataFrame:\n'+str(type(df)))


def merge_id_df(df1, df2, on=None, newindex='left', how='inner', *args, **kwargs):
    """merge two data frames with intersected index

    Args:
        df1 (_type_): _description_
        df2 (_type_): _description_
        newindex (str, optional): _description_. Defaults to 'left'.
        how (str, optional): _description_. Defaults to 'inner'.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    index1 = iterate(df1.index.name or df1.index.names)
    index2 = iterate(df2.index.name or df2.index.names)
    if newindex == 'left':
        index = index1
    elif newindex == 'right':
        index = index2
    elif newindex == 'both':
        index = pd.unique(index1 + index2)
    else:
        raise ValueError(f'Unknown newindex {newindex} in merge_id_df')

    # find the common index names
    if on is None:
        on = []
        for c in set(index1 + index2):
            if c in index1 and c in index2:
                on.append(c)

    index = [i for i in index if i is not None]


    df = pd.merge(
        df1.reset_index(drop=False),
        df2.reset_index(drop=False),
        on=on, how=how, *args, **kwargs)
    if is_true(index):
        df = df.set_index(index, drop=True)
    return df

def interpolate_missing_data(dat, valcols=None, indexcol='time'):
    if valcols is None:
        valcols = dat.columns
    mask = dat[valcols].isna().any(axis=1).values
    if any(mask):
        indexnames = get_index_names(dat)
        assert indexcol in indexnames, f'cannot find index level {indexcol} in {indexnames} for interpolation'
        indexnames.remove(indexcol)
        dats = []
        if len(indexnames) == 1:
            indexnames = indexnames[0]
        for w, df in dat.groupby(level=indexnames):
            for cn in valcols:
                df.loc[:, cn] = pd.to_numeric(df.loc[:, cn], errors='coerce').droplevel(0).interpolate(method='index', limit_area='inside', ).values
            dats.append(df)
        dat = pd.concat(dats, )
    return dat

def impute_time(dat, times, valcols=None, interp=True):
    indexname0 = get_index_names(dat)
    indexnames = [ii for ii in indexname0 if ii != 'time']
    valcols = valcols or dat.columns
    dfindex = dat.reset_index()[indexname0]

    idx_time = indexname0.index('time')
    empty = []
    for ii, other in dfindex.groupby(indexnames):
        empty += [ii[:idx_time] + (t, ) + ii[idx_time:] for t in np.setdiff1d(times, other['time'].values)]

    empty = pd.DataFrame(np.nan, index=pd.MultiIndex.from_tuples(empty, names=indexname0), columns=valcols)
    dat = pd.DataFrame(np.vstack([dat, empty]),
                       index=pd.MultiIndex.from_tuples(list(dat.index) + list(empty.index), names=indexname0),
                       columns=dat.columns).sort_index()

    if interp:
        return dat.groupby(level=indexnames).transform(
            lambda x: x.droplevel(indexnames).interpolate(method='index')).reset_index().set_index(indexname0)
    else:
        return dat.groupby(level=indexnames).transform(
            lambda x: x.droplevel(indexnames).ffill()).reset_index().set_index(indexname0)


def get_prettylabel(names, values, timefmt=None):
    """get the label of the current index. for multi-index, it will squeeze the indexes"""

    label = ''
    for c, v in zip(iterate(names), iterate(values)):
        if c == 'time':
            if timefmt:
                label += pd.Timestamp(v).strftime(timefmt) + ' '
            else:
                label += f'time:{v} '
        else:
            if isinstance(v, str):
                label += v + ' '
            else:
                label += str(c) + ':' +  str(v) + ' '
    return label.strip()


#%% time related
def concise_timefmt(times):

    times = np.array(times[times != False])
    if not is_datetime(times):
        return None

    if len(times) <= 1:
        return '%Y-%m-%d'

    if all(times[1:] == times[0]):
        return '%Y-%m-%d'

    ts = np.array([[t.year, t.month, t.day, t.hour, t.minute, t.second] for t in pd.to_datetime(times)])

    days = ''
    if ts[0, 0] != ts[-1, 0] or any(ts[0, 1] != ts[1:, 1]):
        days = '%Y-%m'

    for i in range(len(ts)-1):
        if (ts[i, 1] == ts[i+1, 1]) and (ts[i, 2] != ts[i+1, 2]):
            days += '-%d'
            break

    days = days.strip('-').strip()
    hours = ''
    t0 = ts[0]
    for t in ts[1:]:
        if all(t[:3] == t0[:3]) and any(t[3:] != t0[3:]):
            hours = 'T%H:%M'
            if t[5] != t0[5]:
                hours = 'T%H:%M:%S'
                break
            break


    return (days + hours).strip()


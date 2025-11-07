import numpy as np
import pandas as pd
import geopandas as gpd
from io import StringIO
from pandas.api.types import is_string_dtype
import json

#%% print dict
def prettify_dict(d, indent, dent):
    res = ""
    for k, v in d.items():
        res += dent*indent + str(k) + "\n"
        if isinstance(v, dict):
            res += prettify_dict(v, indent+1, dent)
        elif isinstance(v, list):
            for vv in v:
                res += dent*(indent+1) + str(vv) + "\n"
        else:
            res += dent*(indent+1) + str(v) + "\n"
    return res

def pretty_print_dict(d, indent=0, dent='  '):
    print(prettify_dict(d, indent=indent, dent=dent))
#%% read spreadsheet-like data
def read_table(filename, sheet=None, crs=None, sep=',', **kwargs):
    not_shp=True
    if filename.lower().endswith('.csv'):
        if sep==" ":
            kwargs["skipinitialspace"] = True
        df = pd.read_csv(filename, sep=sep, **kwargs)
    elif filename.lower().endswith('.shp') or filename.lower().endswith('.zip'):
        df = gpd.read_file(filename, **kwargs)
        not_shp = False
    elif filename.lower().endswith('.xlsx') or filename.lower().endswith('.xls') or filename.lower().endswith('.xlsm'):
        df = pd.read_excel(filename, sheet, **kwargs)
    elif type(filename) is list and (filename[0].lower().endswith('.xlsx') or filename[0].lower().endswith('.xls') or filename[0].lower().endswith('.xlsm')):
        df = pd.read_excel(*filename, **kwargs)
    else:
        # assume text file with delimiter
        try:
            df = pd.read_csv(filename, sep=sep, skipinitialspace=True, **kwargs)
        except:
            raise OSError(f'unrecognized file format for {filename}')

    if 'x' in df.columns and 'y' in df.columns and not_shp:
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['x'], df['y']))
    elif 'wkt_geom' in df.columns and not_shp:
        df = gpd.GeoDataFrame(df.drop('wkt_geom', axis=1), geometry=gpd.GeoSeries.from_wkt(df['wkt_geom']))
    if getattr(df, 'crs', None) is None and crs is not None:
        df.crs = crs

    if 'time' in df.columns:
        try:
            if is_string_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'])
        except:
            pass
    return df

def readarray(source, valwidth=None, dtype=float, skiprows=None, nrows=None, *args, **kwargs):
    vals = []
    with open(source) as f:
        if skiprows:
            for i in range(int(skiprows)):
                f.readline()

        nrows = int(nrows) if nrows else 99999999
        for irow, l in enumerate(f):
            if valwidth:
                l = l.rstrip()
                vals.extend([l[k*valwidth:(k+1)*valwidth] for k in range(int(len(l)/valwidth))])
            else:
                vals.extend(l.split())
            if irow == nrows - 1: break

    return np.array(vals).astype(dtype)

#%%

def read_swat_output(file, IYR=None, NYSKIP=None, IPRINT=None):
    """
    IYR : Beginning year of simulation
    NYSKIP: number of years to skip output printing/summarization
    IPRINT: print code (month, day, year)

    Returns
    -------
    dataframe of the SWAT output

    """

    # check for AREAkm2
    output_type = file.rstrip()[-3:].lower()
    with open(file, 'r') as f:
        while True:
            line = f.readline()
            if 'AREAkm2' in line:
                loc = line.index('AREAkm2')
                column1 = line[:loc+7].split()
                column2 = line[loc+7:].rstrip()
                if output_type == 'rch': #  or output_type == 'rsv'
                    width = 12
                else:
                    width = 10

                if output_type == 'hru':
                    column1 = column1[1:]

                ncol = int(len(column2) / width)
                column2 = column2[-ncol*width:]
                column2 = [column2[i*width:(i+1)*width].strip() for i in range(ncol)]
                break

        dat = pd.read_csv(f, sep=' ', header=None, skipinitialspace=True)
        columns = column1 + column2
        dat = dat.iloc[:, 1:]
        dat.columns = columns


    unit = dat.columns[0]
    nunit = dat[unit].max()
    if 'DA' in column1:
        dat.rename(columns={'YR':'year', 'MO':'month', 'DA':'day'}, inplace=True)
        dat['time'] = pd.to_datetime(dat[['year', 'month', 'day']])
    else:
        if IPRINT is None:
            # check IPRINT
            if all(dat.MON<366):
                IPRINT = 1
            elif all(dat.MON.iloc[:-nunit]>366):
                IPRINT = 2
            else:
                IPRINT = 0
        if IPRINT == 0:
            # monthly output
            dat_yr = dat.MON.copy()
            dat_yr[dat_yr<=366] = pd.NA
            dat['year']=dat_yr.fillna(method='bfill')
            dat = dat[dat.MON<=366].dropna(subset='year')
            dat['day'] = 1
            dat.rename(columns={'MON':'month'}, inplace=True)
            dat['day'] = [t.days_in_month for t in pd.to_datetime(dat[['year', 'month', 'day']])]
            dat['time'] = pd.to_datetime(dat[['year', 'month', 'day']])
        elif IPRINT == 1:
            # daily output
            dat['year'] = 0
            for i in range(nunit, dat.shape[0], nunit):
                if dat.loc[i, 'MON'] <  dat.loc[i-1, 'MON']:
                    dat.loc[i, 'year'] += 1
            dat['year'] = dat['year'].cumsum() + IYR + NYSKIP
            date0 = pd.Timestamp(f'{dat.iloc[0].year:.0f}-01-01')  + pd.Timedelta(dat.iloc[0].MON-1, 'D')
            date1 = pd.Timestamp(f'{dat.iloc[-1].year:.0f}-01-01') + pd.Timedelta(dat.iloc[-1].MON-1, 'D')
            dat['time'] = np.repeat(pd.date_range(date0, date1, freq='D'), nunit)
        elif IPRINT == 2:
            # annual output
            dat = dat[dat.MON>366]
            dat['time'] = pd.to_datetime(dat.MON.astype(int).astype(str)+'-12-31')


    return dat.set_index([unit, 'time'], drop=True).sort_index()[column2]

#%% io for IWFM
def read_iwmffile(file):
    return [l for l in open(file) if (l[0]!='c' and l[0]!='C' and l[0]!='*') ]

def read_iwfmhead(headfile, nodes=None):

    nskiplines = 6
    with open(headfile) as f:
        for i in range(nskiplines):
            f.readline()
        lines = f.readlines()

    heads = np.loadtxt(StringIO(''.join([l[16:] for l in lines])), dtype='f4')
    times = [pd.Timestamp(t) for t in (' '.join([l[:16] for l in lines])).replace('_24:00', 'T23:59:59').split()]

    return  pd.DataFrame(heads, columns=range(1, heads.shape[1]+1),
        index=pd.MultiIndex.from_product([times, range(1, int(len(heads)/len(times)+1)), ], names=["time","layer",])
    )

#%% read modfloe sfr output

def readsfr2df(sfrfile):
    _header = 'period step layer row column segment reach qin qaq qout overland precip et stage depth width condutance gradient'.split()
    with open(sfrfile) as f:
        lines = f.readlines()

    ii = []
    for i, l in enumerate(lines):
        if 'PERIOD' in l:
            ii.append(i)

    dfs = []
    for i1,i2 in zip(ii, ii[1:]+[len(lines)]):
        sp = lines[i1][lines[i1].find('PERIOD')+6:].replace('STEP', '').strip() + ' '
        dfs.append(pd.read_csv(StringIO(''.join([sp + l for l in lines[i1+5:i2] if l != '\n'])), header=None,
                   sep=' ', skipinitialspace=True, index_col=False, names=_header))
    return pd.concat(dfs).reset_index(drop=True)


#%% io from flopy
def binaryread(file, vartype, shape=(1,), charlen=16):
    """
    Uses numpy to read from binary file.  This was found to be faster than the
        struct approach and is used as the default.
    """

    # read a string variable of length charlen
    if vartype == str:
        result = file.read(charlen * 1)
    else:
        # find the number of values
        nval = np.prod(shape)
        result = np.fromfile(file, vartype, nval)
        if nval > 1:
            result = np.reshape(result, shape)
    return result

def get_headfile_precision(filename):
    """
    Determine precision of a MODFLOW head file.
    Parameters
    ----------
    filename : str
    Name of binary MODFLOW file to determine precision.
    Returns
    -------
    result : str
    Result will be unknown, single, or double
    """

    # Set default result if neither single or double works
    result = "unknown"

    # Create string containing set of ascii characters
    asciiset = " "
    for i in range(33, 127):
        asciiset += chr(i)

    # Open file, and check filesize to ensure this is not an empty file
    f = open(filename, "rb")
    f.seek(0, 2)
    totalbytes = f.tell()
    f.seek(0, 0)  # reset to beginning
    assert f.tell() == 0
    if totalbytes == 0:
        raise IOError("datafile error: file is empty: " + str(filename))

    # first try single
    vartype = [
        ("kstp", "<i4"),
        ("kper", "<i4"),
        ("pertim", "<f4"),
        ("totim", "<f4"),
        ("text", "S16"),
    ]
    hdr = binaryread(f, vartype)
    text = hdr[0][4]
    try:
        text = text.decode()
        for t in text:
            if t.upper() not in asciiset:
                raise Exception()
        result = "single"
        success = True
    except:
        success = False

    # next try double
    if not success:
        f.seek(0)
        vartype = [
            ("kstp", "<i4"),
            ("kper", "<i4"),
            ("pertim", "<f8"),
            ("totim", "<f8"),
            ("text", "S16"),
        ]
        hdr = binaryread(f, vartype)
        text = hdr[0][4]
        try:
            text = text.decode()
            for t in text:
                if t.upper() not in asciiset:
                    raise Exception()
            result = "double"
        except:
            f.close()
            e = (
                "Could not determine the precision of "
                + "the headfile {}".format(filename)
            )
            raise IOError(e)

    # close and return result
    f.close()
    return result

def read_ts_rowcol(filename, loc):

    precision = get_headfile_precision(filename)
    floattype = "f8" if precision == "double" else "f4"
    header_dtype = np.dtype(
        [
            ("kstp", "i4"),
            ("kper", "i4"),
            ("pertim", floattype),
            ("totim", floattype),
            ("text", "a16"),
            ("ncol", "i4"),
            ("nrow", "i4"),
            ("ilay", "i4"),
        ]
    )

    # get dimension
    nrow = 0
    ncol = 0
    nlay = 0
    with open(filename, 'rb') as file:
        header = binaryread(file, header_dtype, (1,))
        nrow = int(header["nrow"])
        ncol = int(header["ncol"])


    # get all the row and columns
    inodes = np.sort(np.unique([(l[-2]-1)*ncol+l[-1]-1 for l in loc]))
    npl = nrow * ncol
    heads = {}

    with open(filename, 'rb') as file:
        while True:
            header = np.fromfile(file, header_dtype, 1)
            if header.size == 0:
                break
            htmp = np.fromfile(file, floattype, npl)
            time = header[0][3]
            layer = header[0][7]
            nlay = max(nlay, layer)
            heads[(time, layer)] = htmp[inodes]

    heads = pd.DataFrame(heads, index=inodes+1).T.unstack().swaplevel(axis=1)
    heads.index.name = 'time'
    heads.columns.name = ('layer', 'node')
    return heads, (nlay, nrow, ncol)

def read_ts_nodes(filename, nodes):

    precision = get_headfile_precision(filename)
    floattype = "f8" if precision == "double" else "f4"
    header_dtype = np.dtype(
        [
            ("kstp", "i4"),
            ("kper", "i4"),
            ("pertim", floattype),
            ("totim", floattype),
            ("text", "a16"),
            ("ncol", "i4"),
            ("nrow", "i4"),
            ("ilay", "i4"),
        ]
    )

    # get dimension
    nrow = 0
    ncol = 0
    nlay = 0
    with open(filename, 'rb') as file:
        header = binaryread(file, header_dtype, (1,))
        nrow = int(header["nrow"])
        ncol = int(header["ncol"])

    npl = nrow * ncol

    # get all the row and columns
    heads = {}
    inodes = nodes - 1
    with open(filename, 'rb') as file:
        while True:
            header = np.fromfile(file, header_dtype, 1)
            if header.size == 0:
                break
            htmp = np.fromfile(file, floattype, npl)
            time = header[0][3]
            layer = header[0][7]
            nlay = max(nlay, layer)
            heads[(time, layer)] = htmp[inodes]

    heads = pd.DataFrame(heads, index=nodes).T.unstack().swaplevel(axis=1)
    heads.index.name = 'time'
    heads.columns.name = ('layer', 'node')
    return heads, (nlay, nrow, ncol)


def read_hds(filename):

    precision = get_headfile_precision(filename)
    floattype = "f8" if precision == "double" else "f4"
    header_dtype = np.dtype(
        [
            ("kstp", "i4"),
            ("kper", "i4"),
            ("pertim", floattype),
            ("totim", floattype),
            ("text", "a16"),
            ("ncol", "i4"),
            ("nrow", "i4"),
            ("ilay", "i4"),
        ]
    )

    with open(filename, 'rb') as file:
        header = binaryread(file, header_dtype, (1,))
        nrow = int(header["nrow"])
        ncol = int(header["ncol"])

    array_dtype = np.dtype(header_dtype.descr + [('head', floattype, (nrow, ncol))])

    heads = np.fromfile(filename, dtype=array_dtype)

    ntim = len(np.unique(heads['totim']))
    nlay = max(heads['ilay'])
    result = np.empty([ntim, nlay, nrow, ncol])
    result.fill(np.nan)

    itim = {t:i for i,t in enumerate(np.unique(heads['totim']))}
    itotim = [itim[t] for t in heads['totim']]
    result[itotim, heads['ilay']-1] = heads['head']
    return heads['totim'], result

def agg_hds(filename, minlayer=None, maxlayer=None, mintime=None, maxtime=None, tfunc=None, lfunc=None, limits=None, nodes=None, t_scale=1.0, t_offset=0.0):

    mintime = mintime or -np.inf
    maxtime = maxtime or np.inf
    minlayer = minlayer or 1
    maxlayer = maxlayer or np.inf
    limits = limits or [-np.inf, np.inf]
    precision = get_headfile_precision(filename)
    if nodes is not None:
        nodes = np.array(nodes, dtype=int) - 1
    floattype = "f8" if precision == "double" else "f4"
    header_dtype = np.dtype(
        [
            ("kstp", "i4"),
            ("kper", "i4"),
            ("pertim", floattype),
            ("totim", floattype),
            ("text", "a16"),
            ("ncol", "i4"),
            ("nrow", "i4"),
            ("ilay", "i4"),
        ]
    )

    # get data
    totim = []
    layer  = []
    heads  = []
    ncpl = []
    with open(filename, 'rb') as file:
        while True:
            header = np.fromfile(file, dtype=header_dtype, count=1)
            if header.size == 0:
                break
            nrow = int(header["nrow"][0])
            ncol = int(header["ncol"][0])
            if header["text"][0].strip().upper().endswith(b'U'):
                ncpl.append(nrow-ncol+1)
            else:
                ncpl.append(nrow*ncol)
            htmp = np.fromfile(file, dtype=floattype, count=ncpl[-1])

            t = header['totim'][0] * t_scale + t_offset
            l = header['ilay'][0]
            if t>=mintime and t<=maxtime and l>=minlayer and l<=maxlayer:
                totim.append(t)
                layer.append (l)
                if nodes is None:
                    heads.append(htmp)
                else:
                    heads.append(htmp[nodes])


    heads = pd.DataFrame(heads)
    heads.columns.name = 'node'
    if nodes is None:
        heads.columns = range(1, heads.shape[1]+1)
    else:
        heads.columns = nodes + 1

    heads['time'] = totim
    heads['layer'] = layer
    assert heads.shape[0] > 0, f'Could not find desired data in {filename}'
    return heads#.set_index(['time', 'layer', ])



if __name__ == "__main__":
    pass

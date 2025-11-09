import re

import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from datetime import datetime
from ..utils.misc import (get_index_names, is_true, iterate,
                          iterate_dicts, merge_id_df)
from ..utils import datfunc
from .mobject import mobject


class mobject2d(mobject):
    """`mobject2d` represents tabular data structures.
    """

    @property
    def idcol(self) -> str:
        """set the column name for site/well ID"""
        if 'idcol' in self.dict:
            return self.dict['idcol']
        elif is_true(self._dat):
            return get_index_names(self.dat)[0]

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        self.a_scale: int | float  = self.dict.get('a_scale', 1.0)
        """apply scaling value `a_scale` to all values. """

        self.a_offset: int | float  = self.dict.get('a_offset', 0.)
        """apply offset value `a_offset` to all values. """

        self.minlayer: int  = self.dict.get('minlayer', 0)
        """Filter data with layers larger than or euqal to `minlayer`. """

        self.maxlayer: int  = self.dict.get('maxlayer', 9999)
        """Filter data with layers smaller than or euqal to `maxlayer`. """

        self.lfunc: str  = self.dict.get('lfunc', None)
        """apply a function over the layers.

        Example:
        >>> lfunc = 'max'
        calculate the maximum values over the layers
        """

        self.mintime: int | float | str| datetime  = self.dict.get('mintime', None)
        """Filter data with time larger than or euqal to `mintime`. """

        self.maxtime: int | float | str| datetime  = self.dict.get('maxtime', None)
        """Filter data with time smaller than or euqal to `maxtime`. """

        self.tfunc: str  = self.dict.get('tfunc', None)
        """apply a function over time.

        Example:
        >>> tfunc = 'mean'
        calculate the mean values over time
        """

        self.transpose: str  = self.dict.get('transpose', False)
        """if transpose is true, apply transpose.
        """

        self.exttable: dict | list  = self.dict.get('exttable', {})
        """set external table object(s) to join this object.

        Example:
        >>> exttable = {csv='wells'}
        >>> exttable = [{csv='NEwells'},{csv='KSwells'},{csv='COwells'}]
        """

        self.limits: list  = self.dict.get('limits', None)
        """drop values not in `limits`.

        Example:
        >>> limits = [1000, 5000]
        """

        self.abslimits: list  = self.dict.get('abslimits', None)
        """drop the absolute values not in `abslimits`.

        Example:
        >>> abslimits = [1, 1e10]
        """

        self.countlimit: int  = self.dict.get('countlimit', 1)
        """drop the well/site whose value count is smaller than `countlimit`.
        """

        self.rename: int  = self.dict.get('rename', None)
        """rename index or columns.

        Example:
        >>> rename = {columns={qaq='Simulated'}}
        """

        self.replace: dict  = self.dict.get('replace', None)
        """replace value.

        Example:
        >>> replace = {1e30 = nan}
        """

        self.filter: str  = self.dict.get('filter', None)
        """filter the columns of a DataFrame with a boolean expression. see
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html.

        Example:
        >>> filter = 'obsval > 1000'
        """

        self.writedata: bool  = self.dict.get('writedata', False)
        """set `writedata` to true to export the data. The data will saved as a CSV or SHP file.
        """

        self.exclude: str  = self.dict.get('exclude', None)
        """filter a subset of data that will be plotted differernt.
        also see `filter`. The plotarg for the data subset is defined by `plotarg_exclude`.

        Examples:
        >>> exclude         = 'obsweight == 0'
        >>> plotarg_exclude = {color = 'grey', alpha = 0.5}
        """

        self.plotarg_exclude: dict  = self.dict.get('plotarg_exclude', {})
        """plot arguments for the excluded data.
        """

        self.drop: dict  = self.dict.get('drop', None)
        """drop specified labels from rows or columns, see
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html.

        Example:
        >>> drop = {columns=['colA','colB']}
        """

        self.fillna: dict  = self.dict.get('fillna', None)
        """Fill NA/NaN values using the specified method., see
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html.

        Example:
        >>> fillna = 0
        >>> fillna = {value=0}
        """

        self.resample: dict  = self.dict.get('resample', None)
        """resample data to other time frequency. see
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html.

        Example:
        >>> resample = {rule='7D', func='mean'}
        """

        self.rolling: dict  = self.dict.get('rolling', None)
        """provide rolling window calculations. see
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html.

        Example:
        >>> rolling = {window=12, func='sum'}
        """

        self.subtract: dict = self.dict.get('subtract', None)
        """subtract values of another object.

        Example:
        >>> subtract = {other={csv='obslevel'}, fill_value=0}
        """

        self.add: dict = self.dict.get('add', None)
        """add values of another object.

        Example:
        >>> add = {other={csv='obslevel'}, fill_value=0}
        """

        self.mul: dict = self.dict.get('mul', None)
        """multiply values of another object.

        Example:
        >>> mul = {other={csv='obslevel'}, fill_value=1}
        """

        self.div: dict = self.dict.get('div', None)
        """divide values of another object.

        Example:
        >>> div = {other={csv='obslevel'}, fill_value=1}
        """

        self.concat: list | str = self.dict.get('concat', None)
        """concatenate data from other objects with current object.

        Examples:
        >>> concat = 'head1'
        >>> concat = ['head1', 'head2']
        """

        self.calculate: str = self.dict.get('calculate', None)
        '''set the object as a result of a mathmatic expression.
        The expression will be evaluated by the Python's `eval` function.

        Examples:
        >>> [mfbin.simdiff]
        >>> calculate = '(simheadA - simHeadB) - (simheadC - simHeadD)'

        where `simheadA`, `simheadB`, `simheadC` and `simheadD` are names of other `mfbin` objects.
        '''

        self.dfopts: list | dict = self.dict.get('dfopts', None)
        '''perform dataframe operation.

        Example:
        >>> dfopts = [{diff={}},{sub={other={table='avglevel'}}}]
        '''

        self.to_csv: str = self.dict.get('str', None)
        '''write the data to a CSV file.

        Example:
        >>> to_csv = 'head.csv'
        '''

        self.rate_to_vol: bool = self.dict.get('rate_to_vol', False)
        """if this is true, the values will be multiplied with the duration (time difference). Default is false.
        """

        self.rate_to_cumvol: bool = self.dict.get('rate_to_cumvol', False)
        """if this is true, the values will be multiplied with the duration (time difference) and apply cumsum function. Default is false.
        """

        self.extfunc: str = self.dict.get('extfunc', None)
        '''call a function in an external Python script after reading the data.
        `extfunc` sets the file name of the script.
        This script needs to be placed in the working directory.
        The function name must be "extfunc" and the first argument is the object.
        The dataframe of the object can be accessed through `obj.dat` where obj is the argument name in your function.

        Example:
        >>> # code inside the external script
        >>> def extfunc(obj):
        >>>     # extract results for the final 12 month
        >>>     obj.dat = obj.dat.iloc[-12:]
        '''

        self.writefunc: str = self.dict.get('writefunc', None)
        '''call a function in an external Python script to write data.
        `writefunc` sets the file name of the script.
        This script needs to be placed in the working directory.
        The function name must be "writefunc" and the first argument is the object.
        The dataframe of the object can be accessed through `obj.dat` where obj is the argument name in your function.

        Example:
        >>> # code inside the external script
        >>> import numpy as np
        >>> def writefunc(obj):
        >>>     # write head results as arrray
        >>>     nrow = 50
        >>>     ncol = 100
        >>>     for (time,layer), r in obj.dat.iterrows():
        >>>         np.savetxt(f'time{time}_layer{layer}.dat', r.reshape([nrow, ncol]))
        '''

        self.second_y: bool = self.dict.get('second_y', False)
        '''if `second_y` is true, the data will be plotted on secondary (on the right) Y axis.

        Example:
        >>> second_y = true
        '''

        self._ignoreindex: bool = self.dict.get('ignoreindex', False)

        self.dimcols = []
        self.valcols = []


    def _concat(self):
        elems = []
        for e in iterate(self.concat):
            elems.append(self._getobj({type(self).__name__:e}))
        self._dat = pd.concat([e.dat for e in elems])
        self.copyother(elems)


    def _calculate(self):

        siblings = list(getattr(self.parent, type(self).__name__).keys())
        siblings.remove(self.name)
        eq_final = self.calculate.replace(' ', '')

        # find items in brackets
        elems1 = re.findall(r"\{.+?\}", eq_final)
        elems1 = list(set(elems1))

        # find other items for the same data type
        c1 = eq_final + ''
        for e in elems1:
            c1 = c1.replace(e, '')

        elems2 = []
        objtype = type(self).__name__
        for ee in re.split(r'\W+', c1):
            if ee in siblings:
                key = '{'+objtype+'.'+ee+'}'
                if key not in elems2:
                    elems2.append(key)
                    self.calculate = re.sub(r'\b'+ee+r'\b', key, self.calculate)

        print('      Evaluating equation: ' + re.sub(r'[\{\}]', '', self.calculate) )
        elems = []
        for i, e in enumerate(elems1 + elems2):
            elems.append(self._getobj(e[1:-1], True))
            self.calculate = self.calculate.replace(e, f"elems[{i}].dat")
        if self.verbose >1:
            # print(elems)
            print('Final eq:\n', self.calculate)
        self._dat = eval(self.calculate)
        # if self._ignoreindex:
        #     self._dat.index = index

        self._copyother(elems)

    def _copyother(self, elems):
        # copy the identical dict elements
        keys = list(elems[0].dict.keys())
        for d in elems[1:]:
            keys = np.intersect1d(keys, list(d.dict.keys()))

        for k in keys:
            if k in self.dict:
                continue
            vs = [d.dict[k] for d in elems]
            if all([v==vs[0] for v in vs[1:]]):
                self.dict[k] = vs[0]

        # copy the identical property
        keys = list(elems[0].__dict__.keys())
        for d in elems[1:]:
            keys = np.intersect1d(keys, list(d.__dict__.keys()))

        for k in keys:
            if k in ['_dat', 'dict', 'dat', '_stat', 'exclude']:
                continue
            if k in self.__dict__:
                if is_true(getattr(self, k)):
                    continue
            vs = [d.__dict__[k] for d in elems]
            vs0 = vs[0]
            if isinstance(vs0, pd.DataFrame) or isinstance(vs0, pd.Series):
                if all([vs0.equals(v) for v in vs[1:]]):
                    setattr(self, k, vs0)
            elif isinstance(vs0, np.ndarray):
                if all([np.array_equal(v, vs0, equal_nan=True) for v in vs[1:]]):
                    setattr(self, k, vs0)
            else:
                if all([v==vs0 for v in vs[1:]]):
                    setattr(self, k, vs0)


    def _loaddata(self, ignoreindex=False):
        if self.verbose > 0:
            print(f'    Loading {self.fullname}')

        self._nplot = 0
        self._ignoreindex = ignoreindex and self._ignoreindex

        # get data from either source or other objects
        if self.calculate:
            self._calculate()
        elif self.concat:
            self._concat()
        elif self._dat is None:
            self.readdata() # read data from source

        self._finalizedata()
        self._writedata()


    def _finalizedata(self):
        if not isinstance(self._dat, pd.DataFrame):
            return

        for k in self.dict.keys():
        # functions run in the order defined in the configuration file
            if k == 'exttable':
                continue

            if k.startswith('print'):
                print('\n'+'='*30,self.fullname,'='*30)
                print(self._dat)
                print('='*30,self.fullname,'='*30,'\n')
            else:
                for kk in dir(datfunc):
                    if k.startswith(kk):
                        if self.verbose >= 1:
                            print(f'      {self.fullname} applying {k} ...')
                        getattr(datfunc, kk)(self, k)
                        assert self._dat.shape[0] > 0 and self._dat.shape[1] > 0, (
                        f'{self.fullname} data size reduces to zero after {k}.')


    def _writedata(self):
        if self.writedata:
            if isinstance(self._dat, gpd.GeoDataFrame) and 'time' not in self._dat.reset_index():
                self._dat.to_file(self.fullname.replace(":",".") + '.shp')
            elif isinstance(self._dat, pd.DataFrame):
                d = self._dat
                if d.shape[1] > 10000 and d.shape[1] > d.shape[0]:
                    d = d.T
                d.to_csv(self.fullname.replace(":",".")+'.csv')


    def readdata(self):
        """@private"""
        return


    def _lowerheaders(self, inplace=False):
        df = self.dat.copy()
        self.orig_colnames = df.columns
        df.columns = [c.lower() for c in df.columns]
        if inplace:
            self._dat = df
        else:
            return df


    def update_plot(self, idx=0, hasData=True):
        """@private"""
        return

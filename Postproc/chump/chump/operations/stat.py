from collections import OrderedDict

import numpy as np
import pandas as pd

from ..objects.mobject2d import mobject2d
from ..utils.math import mae, merr, r2, rmse
from ..utils.misc import is_true, iterate, merge_id_df, iterate_dicts, get_index_names
# from ...utils.misc import (concise_timefmt, get_index_names, get_prettylabel,
#                            is_true, iterate, iterate_dicts, merge_id_df)

class stat(mobject2d):
    """
    `stat` is a command used to calculate the statistics of simulation error for each site/well for the `stat` objects defined in the configuration file.

    Example:
    >>> chump.exe stat example.toml
    """


    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        self.data: dict = self.dict['data']
        """ `data` or keys starting with 'data' (e.g. data1, data2 .. or data_new) set the data object.

        Example:
        >>> data  = {tseries='simhead'}
        >>> data  = {pstres='head', tseries='head1'}
        >>> data1 = {pstres='head'}
        >>> data2 = {tseries='head1'}
        """

        self.exttable: dict | list = self.dict.get('exttable', None)
        """ `exttable` sets table object to join with the attribute table of this object. Default is None.
        If `idcol` is defined, the statistics will be aggregated by the `idcol`.

        Example:
        >>> exttable = {csv = 'geology_unit'}
        """

        self.xcol: str = self.dict.get('xcol', 'Observed')
        """ `xcol` is name of the column that used to subtract from simulated values. Default is 'Observed'.

        Example:
        >>> xcol = 'obs'
        """


    def readdata(self):

        datas = []
        for k, v in self.dict.items():
            if k.startswith('data'):
                datas.extend(iterate_dicts(v))

        data = []
        for plotdat in datas:
            for k, kv in plotdat.items():
                for v in iterate(kv):
                    e = getattr(self.parent, k)[v]
                    data.append(e)

        dats = []
        self.labels = []
        for e in data:
            dat = e.dat.copy()

            for c in dat.columns:
                if dat[c].dtype.kind not in 'iufc':
                    continue
                if c in self.labels:
                    cc = c+'_'+e.name
                    dat.rename(columns={c:cc}, inplace=True)
                    c = cc
                if c != self.xcol:
                    self.labels.append(c)

            dats.append(dat)

        self._dat = pd.concat(dats, axis=1)
        assert self.xcol in self._dat.columns, f'xcol is not in the plot data for {self.fullname}.'

    def run(self):
        results = []
        indexnames = get_index_names(self.dat)
        for w, df in self.dat.groupby(level=0):
            x = df[self.xcol]
            for s in self.labels:
                y = df[s]
                err = x - y if self.parent.err_as_oms else y - x
                results.append([w, s, merr(err), mae(err), rmse(err), err.abs().max(), r2(x, y)])


        if is_true(self.exttable):
            exttable = []
            for k, kv in self.exttable.items():
                for v in iterate(kv):
                    tab = getattr(self.parent, k)[v]
                    exttable.append(tab.dat)
            self.exttable = pd.concat(exttable)

        # self.exttable.to_csv("t1.csv")
        # pd.DataFrame(results, columns=[indexnames[0], 'Simulation', 'merr', 'mae', 'rmse', 'maxae', 'r2']).set_index(indexnames[0]).to_csv("t2.csv")
        pd.concat([
            self.exttable,
            pd.DataFrame(results, columns=[indexnames[0], 'Simulation', 'merr', 'mae', 'rmse', 'maxae', 'r2']).set_index(indexnames[0])
        ], axis=1, join='inner').to_csv(f'{self.name}.csv', )
        print(f'{self.name} completed!')

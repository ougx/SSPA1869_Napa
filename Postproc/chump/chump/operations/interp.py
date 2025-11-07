"""@private"""

from collections import OrderedDict

import numpy as np
import pandas as pd

from ..objects.mobject3d import mobject3d
from ..utils.math import mae, merr, r2, rmse
from ..utils.misc import is_true, iterate, merge_id_df, iterate_dicts, get_index_names
# from ...utils.misc import (concise_timefmt, get_index_names, get_prettylabel,
#                            is_true, iterate, iterate_dicts, merge_id_df)

class interp(mobject3d):
    """ Under development

    `interp` is a command used to perform Kriging interpolation.

    Example:
    >>> chump.exe interp example.toml
    """

    def readdata(self):

        self.xcol = self.dict.get('xcol', 'Observed')

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
                err = y - x if self.parent.err_as_oms else x - y
                results.append([w, s, merr(err), mae(err), rmse(err), err.abs().max(), r2(x, y)])

        self.exttable = None
        if 'exttable' in self.dict:
            exttable = []
            for k, kv in self.dict['exttable'].items():
                for v in iterate(kv):
                    tab = getattr(self.parent, k)[v]
                    exttable.append(tab.dat)
            self.exttable = pd.concat(exttable)

        pd.concat([
            self.exttable,
            pd.DataFrame(results, columns=[indexnames[0], 'Simulation', 'merr', 'mae', 'rmse', 'maxae', 'r2']).set_index(indexnames[0])
        ], axis=1, join='inner').to_csv(f'{self.name}.csv', )
        print(f'{self.name} completed!')

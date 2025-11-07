import numpy as np
import pandas as pd

from ...utils.io import agg_hds
from ...utils.misc import is_true
from .modelarray import modelarray


class mfbin(modelarray):
    """read MODFLOW binary output file (head, drawdown or concentration)
    """

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        # the parameters that were defined in the parent classes (initialize here to show it in `pdoc`)
        # data
        self.source      = self.source
        self.model       = self.model
        self.a_scale     = self.a_scale
        self.a_offset    = self.a_offset
        self.limits      = self.limits
        self.abslimits   = self.abslimits
        self.minlayer    = self.minlayer
        self.maxlayer    = self.maxlayer
        self.mintime     = self.mintime
        self.maxtime     = self.maxtime
        self.lfunc       = self.lfunc
        self.tfunc       = self.tfunc
        self.resample    = self.resample
        self.add         = self.add
        self.subtract    = self.subtract
        self.mul         = self.mul
        self.div         = self.div
        self.calculate   = self.calculate
        self.extfunc     = self.extfunc
        self.t_scale     = self.t_scale
        self.t_offset    = self.t_offset

        # plotting
        self.plottype    = self.plottype
        self.plotarg     = self.plotarg
        self.norm        = self.norm
        self.legend      = self.legend
        self.colorbararg = self.colorbararg
        self.clabelarg   = self.clabelarg
        self.levels      = self.levels
        self.dlevel      = self.dlevel
        self.label       = self.label

        # output
        self.writedata   = self.writedata
        self.writeshp    = self.writeshp
        self.writefunc   = self.writefunc


    def readdata(self):

        # print(f'  Loading mfbin {self.name}')
        mf = self._getobj('model')
        self._dat = agg_hds(self.source, )

        if is_true(mf.start_date):
            self.start_date = mf.start_date
            self.time_unit = mf.time_unit
            self._dat[self.timecol] = self._parse_time_col(self._dat[self.timecol])
        self._dat.set_index(["time", "layer"], inplace=True)

        if mf.version =='mf6' and mf.nlay == 0:
            # MF 6 is not layered in disu and its outputs
            layers = mf.dat.index.get_level_values('layer')
            nlay = layers.max()
            if nlay > 1:
                masks = [layers == layer for layer in range(1, nlay + 1)]
                heads = []
                times = []
                layers = []
                for (t,l), hh in self._dat.iterrows():
                    heads += ([list(hh[m]) for m in masks])
                    times += ([t] * nlay)
                    layers += list(range(1, nlay + 1))
                dat = pd.DataFrame(heads, )
                dat['time'] = times
                dat['layer'] = layers
                dat = dat.set_index(['time', 'layer'])
                dat.columns = range(1, dat.shape[1]+1)
                self._dat = dat.sort_index()


    def extract_nodes(self, nodes):

        mf = self._getobj('model')
        _dat = agg_hds(self.source, nodes=nodes)

        if mf.start_date:
            self.start_date = mf.start_date
            self.time_unit = mf.time_unit
            _dat[self.timecol] = self._parse_time_col(_dat[self.timecol])
        _dat.set_index(["time", "layer"], inplace=True)
        orig_dat = self._dat
        self._dat = _dat
        self._finalizedata()
        _dat = self._dat
        self._dat = orig_dat
        return _dat

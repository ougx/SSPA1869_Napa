
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

from .table import table
from ...utils.io import read_table
from ...utils.misc import get_index_names, iterate, iterate_dicts, merge_id_df, is_true
from ...utils.datfunc import exttable, aggfunc

class tseries(table):
    """read time series data."""

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        self._labels = None

        self.source    = self.source
        self.sheet     = self.sheet
        self.sep       = self.sep

        self.valcol: str | list = self.dict.get('valcol')
        """a string or a list of string defining the value columns in the data source.

        Example:
        >>> valcol = 'Simulated'
        >>> valcol = ['Observed', 'Simulated']
        """

        # plotting
        self.plottype = self.dict.get('plottype', 'line')
        """set the plotting type of the time series, must be 'bar' or 'line'. Default is 'line'.

        Example:
        >>> plottype = 'bar'
        """

        self.plotarg = self.plotarg
        self.plotargs: dict = self.dict.get('plotargs', {})
        """a dictionary to set additonal plotting arguments for different data columns.

        Example:
        >>> plotargs = {Observed={color='b'}, Simulated={color='r'}}
        """
        self.location = self.dict.get('location', {})
        """a dictionary containing a geospatial data object `shp` or `CSV` and `table` to define the locations of the plotting site/well.

        Example:
        >>> location = {shp='huc12'}
        """

        self.nmin = self.dict.get('nmin', 0)
        """minimum number of value entry. For sites/wells with observation counts less than this number will be excluded. Default is 0.

        Example:
        >>> nmin = 5
        """

        self.stack = self.dict.get('stack', None)
        """a list of idcol and valcol names.
        if set, it stacks the columns in the table . it can be used for MF6 OBS outputs

        Example:
        >>> stack = ['WellName', 'SimulatedHead']
        """

        self.aggfunc = self.dict.get('aggfunc', 'mean')
        """apply aggregation function based on the site/well id. Default is None.

        Example:
        >>> aggfunc = 'sum'
        """

        self.a_scale   = self.a_scale
        self.a_offset  = self.a_offset
        self.t_scale   = self.t_scale
        self.t_offset  = self.t_offset
        self.limits    = self.limits
        self.abslimits = self.abslimits
        self.mintime   = self.mintime
        self.maxtime   = self.maxtime
        self.tfunc     = self.tfunc
        self.resample  = self.resample
        self.add       = self.add
        self.subtract  = self.subtract
        self.mul       = self.mul
        self.div       = self.div
        self.calculate = self.calculate
        self.extfunc   = self.extfunc

        # output
        self.writedata = self.writedata
        self.writefunc = self.writefunc

        self.axmap = None
        """@private"""

        self.legend = self.legend

        self._stat = None


    @property
    def labels(self, ):
        """@private
        store the names for value columns
        """
        if self._labels is None:
            if is_true(self.valcol):
                self._labels = [c for c in iterate(self.valcol)]
            else:
                self._labels = [c for c in self.dat.columns if not (
                    c.endswith('_log') or c.endswith('_diff') or c in iterate(self.idcol) or c == 'time')]
        return self._labels

    def readdata(self, ):
        if self._dat is None:
            self._dat = read_table(self.source, self.sheet, self.crs, sep=self.sep)

        self.dimcols.append(self.timecol)
        self._dat[self.timecol] = self._parse_time_col(self._dat[self.timecol])

        if is_true(self.stack):
            self.dict['idcol']  = self.stack[0]
            self.valcol         = self.stack[1]
            self._dat.columns.name     = self.stack[0]
            self._dat = self._dat.set_index(self.timecol).stack().rename(self.valcol).reset_index()

        # get the external naming table
        if self.exttable:
            exttable(self, 'exttable', )

        self._process_duplicatedtime()

    def _process_duplicatedtime(self):
        # calculate the mean if there are identical times
        indexnames = [self.idcol, self.timecol] if self.idcol else self.timecol
        af = aggfunc(self, self.aggfunc)
        if isinstance(af, dict):
            af = {k:v for k,v in aggfunc(self, self.aggfunc).items() if k not in indexnames}
        self._dat = self._dat.groupby(indexnames, sort=True).agg(af)


    def _finalizedata(self, ):
        super()._finalizedata()
        self._dat = self._dat.loc[:, self.labels]
        self._nplot = len(pd.unique(self._dat.index.get_level_values(0)))


    def plot(self, *args, **kwargs):
        """@private
        """
        self.lines = {}
        labels = kwargs.get('labels', self.labels)
        plotargs = {}
        for i, l in enumerate(labels):
            plotarg = {}
            plotarg.update(self.plotarg)
            if isinstance(self.plotargs, list):
                plotarg.update(self.plotargs[i])
            elif isinstance(self.plotargs, dict): # this is a dict
                if l in self.plotargs:
                    plotarg.update(self.plotargs[l])

            plotargs[l] = plotarg
            if self.plottype == 'line':
                self.lines[l] = self.ax.plot([], [], label=l, **plotarg)[0]
            elif self.plottype == 'bar':
                self.lines[l] = self.ax.bar([], [], label=l, **plotarg)
        self.plotargs = plotargs

        # check if feature exsits
        if 'location' in self.dict and (self.axmap is not None):
            self.location = self._getobj('location')
            if self.location:
                self.location.ax = self.axmap.ax
                self.location.fig = self.axmap
                self.axmap.legends.update(**self.location.plot())
            # print('============debugging', self.fullname, 'location', self.location.fullname)
        else:
            self.location = None
        return self.lines if self.legend else None

    def update_plot(self, idx=None, hasData=True, *args, **kwargs):
        labels = kwargs.get('labels', self.labels)
        if not hasData:
            for l in labels:
                self.lines[l].set_data([], [])
            return
        if idx == slice(None) or idx is None:
            idx = slice(None)
            tss = self.dat
        else:
            idx = pd.Series({k:v for k,v in idx.items() if k in get_index_names(self.dat)})
            tss = self.dat.xs(tuple(idx), level=tuple(idx.index), )

        for l in labels:
            ts = tss[l]
            ts = ts.dropna()
            if len(ts>0):
                if self.plottype == 'line':
                    self.lines[l].set_data(ts.index, ts.values)
                elif self.plottype == 'bar':
                    self.lines[l].remove()
                    self.lines[l] = self.ax.bar(ts.index, ts.values, label=l, **self.plotargs[l])
            else:
                self.lines[l].set_data([], [])

        if is_true (self.location):
            self.location.update_plot(idx)

    @property
    def stat(self):
        """@private
        """
        if self._stat is None:
            index_names = get_index_names(self.dat)
            index_names.remove('time')
            self._stat = self.dat.groupby(level=0).mean()


    def extract_nodes(self, nodes):
        """@private"""
        return self.dat.loc[:, nodes]

import os

import numpy as np
import pandas as pd

from ...utils.plotting import zorders
from ...utils.misc import is_true
from .figure import figure
from tabulate import tabulate

class edfplot(figure):
    """
    Plot empirical distribution functions.

    Error frequency or cumulative distribution plot with min, max, avg and abs avg

    Args:
        figure ([type]): [description]

    Returns:
        [type]: [description]
    """
    def _loaddata(self):
        super()._loaddata()

        dats = []
        excludes = []
        self.labels = []
        for e in self.plotdata:
            dat = e.dat.copy()
            ex = None
            if is_true(e.exclude):
                ex  = e.exclude.copy()
            for c in dat.columns:
                if dat[c].dtype.kind not in 'iufc':
                    continue
                if c in self.labels:
                    cc = c+'_'+e.name
                    dat.rename(columns={c:cc}, inplace=True)
                    if is_true(ex):
                        ex.rename(columns={c:cc}, inplace=True)
                    c = cc
                self.labels.append(c)
            dats.append(dat)
            if is_true(ex):
                excludes.append(ex)

        self._dat = pd.concat(dats, axis=1)[self.labels]
        self.exclude = None
        if len(excludes)>0:
            self.exclude = pd.concat(excludes, axis=1)[self.labels]

        # calculate the difference/error/residual
        if self.dict.get('diff', False):
            self.xcol = self.dict.get('xcol', 'Observed')
            if len((self._dat[self.xcol]).shape) > 1:
                x = self._dat[self.xcol].mean(axis=1)
                self._dat.drop(columns=[self.xcol], inplace=True)
                self._dat[self.xcol] = x
            errfact = -1 if is_true(self.parent.err_as_oms) else 1
            for c in self._dat.columns:
                if c == self.xcol: continue
                # check if it is numeric
                self._dat[c] = (self._dat[c] - self._dat[self.xcol]) * errfact
            self._dat.drop(self.xcol, axis=1, inplace=True)

        self.onepage = self.dict.get('onepage', True)


    def add_elements(self, ):

        ax = self.ax
        self.val_on_xaxis = self.dict.get('val_on_xaxis', False) # x or y axis for values
        self._nplot = self.dat.shape[1]

        self.vmin = np.nanmin(self.dat)
        self.vmax = np.nanmax(self.dat)

        if self.val_on_xaxis:
            if 'xlim' not in self.dict:
                self.ax.set_xlim([self.vmin - 0.01*(self.vmax - self.vmin), self.vmax + 0.01*(self.vmax - self.vmin)])
            if 'ylim' not in self.dict:
                self.ax.set_ylim(-1, 101)
        else:
            if 'ylim' not in self.dict:
                self.ax.set_ylim([self.vmin - 0.01*(self.vmax - self.vmin), self.vmax + 0.01*(self.vmax - self.vmin)])
            if 'xlim' not in self.dict:
                self.ax.set_xlim(-1, 101)

        if self.onepage:
            self._nplot = 0
            self._bookmark = None
            legends = {}
            for i, c in enumerate(self.dat.columns):
                d = pd.Series(self.dat[c].rank(method='average', pct=True).values * 100, index=self.dat[c].values, name=c).sort_index()
                if self.val_on_xaxis:
                    legends[c] = ax.plot( d.index, d.values, label=c, color=f'C{i}', **self.plotarg)[0]
                else:
                    legends[c] = ax.plot( d.values, d.index, label=c, color=f'C{i}', **self.plotarg)[0]

            self.legends = legends
        else:
            self._bookmark = list()
            self.edf = ax.plot([], [], **self.plotarg)


        if is_true(self.dict.get('stats')):
            plotarg = self.dict.get('plotarg_stat', {})
            if 'x' not in plotarg:
                plotarg['x'] = 0.98
            if 'y' not in plotarg:
                plotarg['y'] = 0.02
            plotarg['s'] = ' '
            if 'fontsize' not in plotarg:
                plotarg['fontsize'] = 'small'
            if 'va' not in plotarg:
                plotarg['va'] = 'bottom'
            if 'ha' not in plotarg:
                plotarg['ha'] = 'right'
            if 'zorder' not in plotarg:
                plotarg['zorder'] = zorders['statstable']
            if 'fontfamily' not in plotarg:
                plotarg['fontfamily'] = 'monospace'

            if self.onepage:
                st = self.dat.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).iloc[1:]
                plotarg['s'] = tabulate(st, headers='keys', showindex=True, tablefmt='simple', numalign='right', stralign='left')
            self.stattext = ax.text(**plotarg, transform=ax.transAxes)

        # if 'xlabel' not in self

    def update_plot(self, ):
        if self.onepage:
            self.saveplot(0)
        else:
            for iplot, c in enumerate(self.dat.columns):
                print(f'    Writing {self.fullname} {iplot+1:<4} {c}')
                d = pd.Series(self.dat[c].rank(method='average', pct=True).values * 100, index=self.dat[c].values, name=c).sort_index()
                if self.val_on_xaxis:
                    self.edf.set_data(d.index, d.values)
                else:
                    self.edf.set_data(d.values, d.index, )

                st = self.dat.loc[:,c:c].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).iloc[1:]
                self.stattext.set_text(tabulate(st, showindex=True, tablefmt='simple', numalign='right', stralign='left'))
                self.saveplot(iplot)

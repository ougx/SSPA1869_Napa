from collections import OrderedDict
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from ...utils.constants import navalue
from ...utils.math import agreement_index, predict_interval, r2
from ...utils.misc import is_true, merge_id_df, iterate, get_index_names, interpolate_missing_data
from ...utils.plotting import get_catgorical_cmap, zorders, left_adjust
from .figure import figure

# statistics: r2, mae, me, rmse,

class scatterplot(figure):
    '''
    source
    wellprop
    grouping
    exclude: conditions appiled to exclude points not used in the
    exclude_color: the color used for the excluded points
    '''

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        self.xcol = self.dict.get('xcol', 'Observed')
        """column names used for x-axis. """

        self.segmented = self.dict.get('segmented', False)
        """create multiple segmented regression lines for each group. """



    def _loaddata(self):
        super()._loaddata()


        idcol = None
        dats = []
        excludes = []
        self.labels = []
        self.droplevel = []
        for e in self.plotdata:
            dat = e.dat.copy()
            ex = None
            if is_true(e.exclude):
                ex  = e.exclude.copy()
            for c in e.labels or dat.columns:
                if dat[c].dtype.kind not in 'iufc':
                    continue
                if c in self.labels:
                    cc = c+'_'+e.name
                    dat.rename(columns={c:cc}, inplace=True)
                    if is_true(ex):
                        ex.rename(columns={c:cc}, inplace=True)
                    c = cc
                if c != self.xcol:
                    self.labels.append(c)
            for idx in get_index_names(dat):
                if idx not in self.droplevel:
                    self.droplevel.append(idx)
            dats.append(dat)
            if is_true(ex):
                excludes.append(ex)


        self._dat = pd.concat(dats, axis=1)
        self.exclude = None
        if any(map(lambda x: x is not None, excludes)):
            self.exclude = pd.concat(excludes, axis=1)

        assert self.xcol in self._dat.columns, f'xcol is not in the plot data for {self.fullname}.'

        # merge xcol if there are multiple
        if np.count_nonzero(self._dat.columns == self.xcol) > 1:
            x = self._dat[self.xcol].mean(axis=1)
            self._dat.drop(columns=[self.xcol], inplace=True)
            self._dat[self.xcol] = x

        self._nplot = len(self.labels) # exclude the xcol


        if is_true(self.exttable):
            self._dat = merge_id_df(self._dat, self.exttable, newindex='left').sort_index()

        indexcol = get_index_names(dat)[-1]
        dat = interpolate_missing_data(dat, self.labels, indexcol)

        self._dat = self._dat.dropna(subset=self.xcol).reset_index()
        self.vmin = np.nanmin(self.dat[[self.xcol]+self.labels].values)
        self.vmax = np.nanmax(self.dat[[self.xcol]+self.labels].values)

        self.grouping = None
        if 'grouping' in self.dict:
            self.grouping = self.dict.get('grouping')
            assert self.grouping in self._dat.columns, f'Groupung name {self.grouping} not in {self.fullname}'
            self.groups = self._dat[self.grouping]
            self.plotarg['cmap'], self.plotarg['c'], self.factors = get_catgorical_cmap(self._dat[self.grouping].values, self.plotarg.get('cmap'))
            # self.plotarg['vmin'] = 0
            # self.plotarg['vmax'] = 1
            if self.grouping in self.labels:
                self.labels.remove(self.grouping)
        else:
            self.groups = None
            self.segmented = False


        self._x = self._dat[self.xcol]
        self._dat = (self._dat.set_index(self.xcol)[self.labels]).stack(future_stack=True).swaplevel()
        self._indices = pd.DataFrame(True, index=self._dat.index, columns=[self.name]).groupby(level=0, sort=False).any() # TODO: better handle this in future for bookmarks

        self.err = self._dat.copy().rename('error')
        self.err[:] = self.err.values - self.err.index.get_level_values(self.xcol).values
        self.errfact = -1 if is_true(self.parent.err_as_oms) else 1

        self.emin = np.nanmin(self.err.values * self.errfact)
        self.emax = np.nanmax(self.err.values * self.errfact)

        if is_true(self.exclude) and is_true(self.indextable):
            self.exclude = merge_id_df(self.exclude, self.indextable, newindex='left').set_index(self.xcol)[self.labels].stack().squeeze().swaplevel()

        # both log scale if any of it is log scale
        if self.dict.get('xscale') == 'log' or self.dict.get('yscale') == 'log':
            self.dict['xscale'] = 'log'
            self.dict['yscale'] = 'log'

    def add_elements(self, ):
        ax = self.ax
        # add stat text
        self.stattext = False
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

            self.stattext = ax.text(**plotarg, transform=ax.transAxes)


        # add scatter
        if 'zorder' not in self.plotarg:
            self.plotarg['zorder'] = zorders['scatter']
        if 'alpha' not in self.plotarg:
            self.plotarg['alpha'] = 0.8
        if 's' not in self.plotarg:
            self.plotarg['s'] = 10

        self.sc   = ax.scatter(self.x, self.x, **self.plotarg) #
        self.scexclude = None
        if is_true(self.exclude):
            plotarg = self.plotarg.copy()
            plotarg['c'] = self.dict.get('exclude_color', 'grey')
            plotarg['zorder'] = plotarg['zorder'] - 1
            self.scexclude = ax.scatter([], [], **plotarg) #


        self.xmin = self.vmin - 0.02 * (self.vmax - self.vmin)
        self.xmax = self.vmax + 0.02 * (self.vmax - self.vmin)
        self.percentile = self.dict.get('percentile', [0.05, 0.95])

        self.ci = None
        if is_true(self.percentile): # and not (self.dict.get('xscale') == 'log' or self.dict.get('yscale') == 'log')
            self.ci = ax.fill_between([],[],[], color="lightgrey", alpha=0.5, zorder=zorders['scatter_ci'])
            self.median = ax.plot([], [], color='darkgrey', linewidth=1.0, zorder=zorders['scatter_med'], linestyle='--')[0]

        self.line11 = ax.axline([1.,1.], [2.,2.], color='k', linewidth=1.0, zorder=zorders['scatter_11'], )

        self.regline = None
        if self.dict.get('regression_line', True):
            plotarg = self.dict.get('plotarg_regression', {})
            if 'linewidth' not in plotarg: plotarg['linewidth'] = 1.0
            if 'zorder'    not in plotarg: plotarg['zorder'] = zorders['scatter_reg']
            if self.segmented:
                self.regline = {}
                for i in range(len(self.factors)):
                    plotarg['color'] = self.plotarg['cmap'](i)
                    self.regline[self.factors[i]] = ax.plot(
                        [self.xmin,self.xmin], [self.xmax,self.xmax],
                    **plotarg)[0]
            else:
                if 'color'     not in plotarg: plotarg['color'] = 'm'
                self.regline = ax.plot([self.xmin,self.xmin], [self.xmax,self.xmax], **plotarg)[0]
        self.yerror = self.dict.get('yerror', False)

        handles =[]
        labels = []
        title = None
        if is_true(self.grouping):
            # handles += [Line2D([], [], marker='o', linestyle='none', color=self.plotarg['cmap'](i/len(self.grouping))) for i in range(len(self.grouping)) ]
            handles += [Line2D([], [], marker='o', linestyle='none', color=self.plotarg['cmap'](i)) for i in range(len(self.factors)) ]
            labels += iterate(self.factors)
            if not isinstance(labels[0], str):
                title = str(self.grouping).capitalize()

        if is_true(self.exclude):
            plotarg = {'color':self.dict.get('exclude_color', 'grey'), 'marker':'o', 'linestyle':'none'}
            handles.append(Line2D([], [], **plotarg))
            labels.append(self.dict.get('label_exclude', 'Excluded'))

        self.legendarg = self.dict.get('legendarg', {})
        if len(labels)>0:
            ncols = 1
            if len(labels) > 7:
                # ncols = int(len(labels)/4)+1
                ncols = 4
            if 'ncols' not in self.legendarg and ncols > 1:
                self.legendarg['ncols'] = ncols
            if 'title' not in self.legendarg:
                self.legendarg['title'] = title
            self.legends.update(**{str(l):h for l,h in zip(labels, handles)})

        if 'xlabel' not in self.dict:
            ax.set_xlabel(self.xcol)

        self.ax.set_aspect(self.dict.get('aspect', 'equal'))

        if self.dict.get('xscale') == 'log' and not self.yerror:
            self.dict['yscale'] = 'log'

        self.logscale = self.dict.get('yscale') == 'log'

        if self.logscale:
            # assert self.vmin > 0 and self.vmax > 0, f'log scale is not supported with non-positive values in the data for {self.fullname}.'
            self.xmin = self.vmin * 0.9
            self.xmax = self.vmax * 1.1

        if 'xlim' not in self.dict:
            self.ax.set(xlim=(self.xmin, self.xmax))
        else:
            self.xmin, self.xmax = self.dict['xlim']

        if 'ylim' not in self.dict:
            if self.yerror:
                self.ax.set(ylim=(self.emin - 0.02 * (self.emax - self.emin), self.emax + 0.02 * (self.emax - self.emin)))
            else:
                self.ax.set(ylim=(self.xmin, self.xmax))



        self.hist_bins = None
        if self.dict.get('histogram'):
            bbox = self.ax.get_position()
            w = bbox.x1 - bbox.x0
            h = bbox.y1 - bbox.y0
            if w > h:
                h1 = 0.1 * h /w
                w1 = 0.1
            else:
                h1 = 0.1
                w1 = 0.1 * w / h

            # print(w, h, w1, h1)
            # https://stackoverflow.com/questions/37008112/matplotlib-plotting-histogram-plot-just-above-scatter-plot

            self._y = self.x

            bins = max(min(int(len(self.x) / 10), 55), 9)
            if self.dict.get('xscale') == 'log' or self.dict.get('yscale') == 'log':
                bins = np.logspace(np.log10(self.xmin), np.log10(self.xmax), bins)
            else:
                bins = np.linspace(self.xmin, self.xmax, bins)

            self.ax_xDist = self.fig.add_subplot(sharex=self.ax)
            self.ax_xDist.set_position([bbox.x0, bbox.y1, bbox.x1 - bbox.x0, h1])

            _, _, self.xhist = self.ax_xDist.hist(self.x,bins=bins,align='mid',color='lightgrey')
            self.ax_xDist.axis('off')

            # self.ax_xCumDist = self.ax_xDist.twinx()
            # self.ax_xCumDist.hist(x,bins=bins,cumulative=True,histtype='step',density=True,color='r',align='mid')
            # self.ax_xCumDist.tick_params('y', colors='r')
            # self.ax_xCumDist.set_ylabel('cumulative',color='r')

            self.ax_yDist = self.fig.add_subplot(sharey=self.ax)
            self.ax_yDist.set_position([bbox.x1, bbox.y0, w1, bbox.y1-bbox.y0])
            _, _, self.yhist = self.ax_yDist.hist(self.y,bins=bins,orientation='horizontal',align='mid',color='lightgrey')
            self.ax_yDist.axis('off')

            self.hist_bins = bins
            # self.ax_yDist.set(xlabel='count')
            # self.ax_yCumDist = self.ax_yDist.twinx()
            # self.ax_yCumDist.hist(x,bins=100,cumulative=True,histtype='step',density=True,color='r',align='mid',orientation='horizontal')
            # self.ax_yCumDist.tick_params('x', colors='r')
            # self.ax_yCumDist.set_xlabel('cumulative',color='r')

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def update_plot(self,):
        # print(self.dat)
        self._bookmark = self.labels
        self.ax.set_aspect(self.dict.get('aspect', 'equal'))
        self.resize_fig()
        for iplot, l in enumerate(self.labels):
        # for iplot, f in enumerate(self.feats):

            print(f'    Writing {self.fullname} {iplot+1:<4} {l}')

            self._y = self.dat.loc[l]
            self._x = self._y.index

            # update statistics
            err = self.err.loc[l]
            errfact = self.errfact

            # update scatter
            xy = np.array([self.x, err*errfact if self.yerror else self.y]).T
            self.sc.set_offsets(xy)


            # update histogram
            if is_true(self.hist_bins):
                n, _ = np.histogram(self.x, self.hist_bins)
                for count, rect in zip(n, self.xhist.patches):
                    rect.set_height(count)

                n, _ = np.histogram(self.y, self.hist_bins)
                for count, rect in zip(n, self.yhist.patches):
                    rect.set_width(count)

            # update regressions
            mask = ~np.isnan(err)
            x = np.log10(self.x) if self.logscale else self.x
            y = np.log10(self.y) if self.logscale else self.y
            mask = (~np.isnan(x))&(~np.isnan(y))&(~np.isinf(x))&(~np.isinf(y))

            regeq = []
            if is_true(self.regline):
                if self.segmented:
                    regeq.append('Regression Lines:')
                    groups = self.groups.values
                    for i, f in enumerate(self.factors):
                        xseg = x[mask&(groups == f)]
                        yseg = y[mask&(groups == f)]
                        if len(yseg) > 1:
                            xmin = np.min(xseg)
                            xmax = np.max(xseg)
                            a, b = np.polyfit(xseg, yseg, 1)
                            if self.logscale:
                                xx = np.array([(xmin*0.9), (xmax*1.1)])
                                yy = a*xx+b
                                self.regline[f].set_data(10**xx, 10**yy)
                                regeq.append(f'  {f}: log(y) = {a:.3g} log(x) + {b:.4g}')
                            else:
                                dx = xmax - xmin
                                xx = np.array([xmin-0.1*dx, xmax+0.1*dx])
                                self.regline[f].set_data(xx, a*xx+b)
                                regeq.append(f'  {f}: y = {a:.3g} x + {b:.4g}')
                else:
                    regeq.append('Regression Line:')
                    xseg = x[mask]
                    yseg = y[mask]
                    if len(yseg) > 1:
                        xmin = np.min(xseg)
                        xmax = np.max(xseg)
                        a, b = np.polyfit(xseg, yseg, 1)
                        if self.logscale:
                            xx = np.array([10**(xmin*0.95), 10**(xmax*1.05)])
                            yy = 10**(a*xx+b)
                            self.regline.set_data(xx, yy)
                            regeq.append(f'  log(y) = {a:.3g} log(x) + {b:.4g}')
                        else:
                            dx = xmax - xmin
                            xx = np.array([xmin-0.05*dx, xmax+0.05*dx])
                            self.regline.set_data(xx, a*xx+b)
                            regeq.append(f'  y = {a:.3g} x + {b:.4g}')


            if is_true(self.ci):
                cidn, ciup, median = np.quantile(y - x, list(self.percentile)+[0.5])
                if self.logscale:
                    cidn, ciup, median = 10**cidn, 10**ciup, 10**median
                if self.yerror:
                    ci = [[[self.xmin, cidn], [self.xmin, ciup], [self.xmax, ciup], [self.xmax, cidn]]]
                    self.median.set_data([self.xmin, self.xmax], [median, median])
                else:
                    if self.logscale:
                        ci = [[[self.xmin, self.xmin*cidn], [self.xmin, self.xmin*ciup], [self.xmax, self.xmax*ciup], [self.xmax, self.xmax*cidn]]]
                        self.median.set_data([self.xmin, self.xmax], [self.xmin*median, self.xmax*median])
                    else:
                        ci = [[[self.xmin, self.xmin+cidn], [self.xmin, self.xmin+ciup], [self.xmax, self.xmax+ciup], [self.xmax, self.xmax+cidn]]]
                        self.median.set_data([self.xmin, self.xmax], [self.xmin+median, self.xmax+median])

                self.ci.set_verts(ci)

            if is_true(self.stattext):
                aerr = err.abs()
                stattext = [
                    f'Statistics:',
                    f'  MinErr    : {np.min(err*errfact) :10.6G}',
                    f'  AvgErr    : {np.mean(err*errfact):10.6G}',
                    f'  MaxErr    : {np.max(err*errfact) :10.6G}',
                    f'  RMSE      : {np.mean(err**2)**0.5:10.6G}',
                    f'  AvgAbsErr : {np.mean(aerr)       :10.6G}',
                    f'  R-squared : {r2(x, y)            :10.6G}',
                ]
                    # f'Agreement : {r2(self.x, self.y)     :10.6G}\n' + \

                self.stattext.set_text("\n".join(left_adjust(stattext+regeq)))
            if is_true(self.exclude):
                xexclude = self.exclude[self.xcol]
                yexclude = self.exclude[l]
                self.scexclude.set_offsets(np.array([xexclude, (yexclude-xexclude)*errfact if self.yerror else yexclude ]).T)

            self.saveplot(iplot)

        return

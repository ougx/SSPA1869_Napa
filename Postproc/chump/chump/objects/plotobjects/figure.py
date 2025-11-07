
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib import ticker
from matplotlib.animation import FFMpegWriter, HTMLWriter, PillowWriter
from matplotlib.ticker import StrMethodFormatter

from ...utils.plotting import ax_set_kwargs, resize_fig, zorders
from ...utils.misc import (concise_timefmt, get_index_names, get_prettylabel, warn_deprecation,
                           is_true, iterate, iterate_dicts, merge_id_df, get_unique_unsorted)
from ..mobject import mobject
from .. import plotobjects

mpl.rcParams['axes.formatter.useoffset'] = False

# indextable,
# indexnames (used for bookmarks, log); should be the columns in the indextable
# datatable

class figure(mobject):

    def __init__(self,name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)
        self._bookmark = None
        self.ax1 = None
        self.timefmt = '%Y-%m-%d'
        if 'showlegend' in self.dict:
            warn_deprecation(self, '`showlegend` is deprecated since v20240615. Please use `legend` instead.')
            self.legend = self.dict.get('showlegend', self.legend)

        self.legends = {}
        self.Elements = []
        self._indices = None
        self.moviewriter = None
        self.droplevel = []
        self.legendarg = self.dict.get('legendarg', {})
        self.inset = self.dict.get('inset', {})
        self.grouping = self.dict.get('grouping', '')
        """define inset plot """

        self.ysymmetric = self.dict.get('ysymmetric', False)

    @property
    def idcol(self):
        if 'idcol' in self.dict:
            return self.dict['idcol']
        elif is_true(self.indexnames):
            return getattr(self, 'indexnames', [''])[0]
        else:
            raise ValueError(f'Unknown idcol for {self}')

    def setup_ax(self, fig=None, projection=None):
        width   = self.dict.get('width', 1.0)
        height  = self.dict.get('height', 1.0)
        figsize = self.dict.get('figsize', (8, 6))

        if fig is None:
            fig = plt.figure(figsize=figsize, ) # subplotpars={'wspace':0,'hspace':0.}
            fig.subplots_adjust(0.01, 0.01, 0.99, 0.99, 0.01, 0.01)

            if 'logo' in self.dict:
                logo = self.dict['logo'].copy()
                l = plt.imread(logo.pop('source'))
                fig.figimage(l, **logo)

        # ax = fig.add_subplot(projection=projection, )
        ax = fig.add_subplot(projection=None, )
        ax.name = self.fullname

        if 'bounds' in self.dict:
            # https://stackoverflow.com/questions/30030328/correct-placement-of-colorbar-relative-to-geo-axes-cartopy
            bounds = np.array(self.dict['bounds'])
            if len(fig.axes) > 0:
                # fig.canvas.draw() not used so it wont mess up the calculation
                bbox0 = fig.axes[0].get_position()
                width  = bounds[2] * bbox0.width
                height = bounds[3] * bbox0.height
                x0 = bbox0.x0 + bounds[0] * bbox0.width
                y0 = bbox0.y0 + bounds[1] * bbox0.height
                bounds = [x0, y0, width, height]
        else:
            bounds = [0,0,width,height]
        ax.set_position(bounds)


        if self.dict.get('showborder', False):
            border = self.dict.get('borderarg', {})
            if 'visible' not in border:
                border['visible'] = True
            if 'linewidth' not in border:
                border['linewidth'] = 1
            if 'color' not in border:
                border['color'] = 'k'
            if 'zorder' not in border:
                border['zorder'] = zorders['axborder']

            ax.spines.left.set  (**border)
            ax.spines.right.set (**border)
            ax.spines.top.set   (**border)
            ax.spines.bottom.set(**border)


        props = {k:v for k, v in self.dict.items() if k in ax_set_kwargs}
        ax.set(**props)

        if 'xfmt' in self.dict:
            ax.xaxis.set_major_formatter(StrMethodFormatter(self.dict['xfmt']))

        if 'yfmt' in self.dict:
            ax.yaxis.set_major_formatter(StrMethodFormatter(self.dict['yfmt']))

        if 'x_datefmt' in self.dict:
            xfmt = mdates.DateFormatter(self.dict['x_datefmt'])
            ax.xaxis.set_major_formatter(xfmt)

        if 'ticklabel_format' not in self.dict:
            ax.ticklabel_format(**self.dict.get('ticklabel_format', {}))

        if 'title' in self.dict:
            if isinstance(self.dict['title'], str):
                self.title = ax.text(0.5, 0.95, self.dict['title'], ha='center', va='top', zorder=zorders['axtitle'], transform=ax.transAxes) #weight='bold',
            if isinstance(self.dict['title'], dict):
                text = self.dict['title'].pop('text')
                ha = self.dict['title'].pop('ha') if 'ha' in self.dict['title'] else 'center'
                va = self.dict['title'].pop('va') if 'va' in self.dict['title'] else 'top'
                self.title = ax.text(0.5, 0.95, text, transform=ax.transAxes, ha=ha, va=va, **self.dict['title']) #weight='bold',

        self.text = []
        self.adjusttext = []
        for k, v in self.dict.items():
            if k.startswith('text'):
                for tt in iterate_dicts(v):
                    t = tt.copy()
                    if 'transform' not in t:
                        t['transform'] = ax.transAxes
                    else:
                        t['transform'] = getattr(ax, t['transform'])
                    x = t.pop('x')
                    y = t.pop('y')
                    adjust = t.pop('adjust') if 'adjust' in t else (x > 0 and x < 1 and y > 0 and y < 1 and type(self).__name__!='mapplot' and type(self).__name__!='vplot')
                    text = t.pop('text')
                    text = ax.text(x, y, text, **t, )
                    text._orig = text.get_text()
                    self.text.append(text)
                    # if adjust:

            if k.startswith('annotate'):
                for tt in iterate_dicts(v):
                    text = ax.annotate(**tt)
                    text._orig = text.get_text()
                    self.text.append(text)

            if k.startswith('axhline'):
                for tt in iterate_dicts(v):
                    ax.axhline(**tt)

            if k.startswith('axvline'):
                for tt in iterate_dicts(v):
                    ax.axvline(**tt)

            if k.startswith('axhspan'):
                for tt in iterate_dicts(v):
                    ax.axhspan(**tt)

            if k.startswith('axvspan'):
                for tt in iterate_dicts(v):
                    ax.axvspan(**tt)

        self.fig = fig
        self.ax = ax
        for e in self.plotdata:
            if e.dict.get('second_y', False):
                if self.ax1 is None:
                    self.ax1 = self.ax.twinx()
                    self.ax1.set_position(bounds)
                    self.ax1.name = self.fullname + '_2'
                    secondy_arg = self.dict.get('secondy_arg', {})
                    if 'ylim2' in self.dict:
                        secondy_arg['ylim'] = self.dict['ylim2']
                    self.ax1.set(**secondy_arg)
                    # self.ax1.patch.set_visible(False)
                    # self.ax1.set_frame_on(False)
                    self.ax1.grid(False)
                e.ax = self.ax1
                e.fig = self
            else:
                e.ax = self.ax
                e.fig = self

        for kk, vv in self.inset.items():
            for k, sp in vv.items():
                pass
                sp.initialize_plot(self.fig, )
                self.text.extend(sp.text)
                for p in sp.plotdata:
                    p.legend = False
                self.plotdata.extend(sp.plotdata)
                self.droplevel.extend(sp.droplevel)


    def _loaddata(self):
        print(f'  Creating {self.fullname}')
        plotdatas = []
        for k, v in self.dict.items():
            if k.startswith('plotdata'):
                plotdatas.extend(iterate_dicts(v))

        idcol = None
        self.plotdata = []
        for plotdat in plotdatas:
            for k, kv in plotdat.items():
                for v in iterate(kv):
                    e = getattr(self.parent, k)[v]
                    if is_true(getattr(e, 'idcol', None)):
                        idcol = e.idcol
                    self.plotdata.append(e)


        self.exttable = False
        if 'exttable' in self.dict:
            exttable = []
            self.exttable = True
            for k, kv in self.dict['exttable'].items():
                for v in iterate(kv):
                    tab = getattr(self.parent, k)[v]
                    exttable.append(tab.dat)

            self.exttable = pd.concat(exttable)


        for subplot in dir(plotobjects):
            if subplot.startswith('_') or subplot == 'figure':
                continue
            if subplot in self.dict:
                if subplot in self.inset:
                    if isinstance(self.inset[subplot], list):
                        self.inset[subplot].append(self.dict[subplot])
                    elif isinstance(self.inset[subplot], str):
                        self.inset[subplot] = [self.inset[subplot], self.dict[subplot]]

        for subplot in iterate_dicts(self.inset):
            for k, kv in subplot.items():
                subs = {}
                for v in iterate(kv):
                    sp = getattr(self.parent, k)[v]
                    if isinstance(sp, dict):
                        sp = getattr(plotobjects, k)(v, self.parent)
                    subs[v] = sp
                self.inset[k] = subs


    def get_index_row(idx):
        pass

    def add_elements(self):
        pass


    def initialize_plot(self, fig=None):
        self._loaddata()
        self.setup_ax(fig=fig)
        self.add_elements()
        self.plot_legend()


    def plot_legend(self):
        # print(self.name, "legend", self.legend, self.legends)
        if self.legend:
            legendarg = self.legendarg
            if 'fontsize' not in legendarg:
                legendarg['fontsize'] = 'small'
            zorder = legendarg.pop('zorder') if 'zorder' in legendarg else zorders['legend']
            self.ax.legend(self.legends.values(), self.legends.keys(), **legendarg).set_zorder(zorder)
        else:
            if "legend_" in dir(self.ax):
                l = self.ax.legend_
            else:
                l = self.ax.get_legend()
            if l:
                l.remove()

    @property
    def bookmark(self):
        if self._bookmark is None:
            if is_true(self.indices):
                index_names = get_index_names(self.indices)[::-1]
                if not is_true(index_names):
                    self._bookmark = None
                    return
                for g in iterate(self.grouping):
                    if g and g not in index_names:
                        index_names.append(g)
                index_names = index_names[::-1]
                self._indices = self.indices.reset_index()
                self._indices[index_names] = self._indices[index_names].fillna('NULL')
                self._indices = self._indices.set_index(index_names).sort_index()

                if 'time' in index_names:
                    self.timefmt = concise_timefmt(self._indices.index.get_level_values('time'))

                nrow = len(self._indices)
                if len(index_names)>1:
                    bookmark = {}
                    for il, df in self._indices.reset_index().groupby(index_names[:-2] or [False]*nrow):
                        b = bookmark
                        if il != False:
                            for n,k in zip(index_names[:-2],  iterate(il)):
                                k = get_prettylabel(n, k, self.timefmt)
                                if k not in b:
                                    b[k] = {}
                                b = b[k]

                        for il1, df1 in df.groupby(index_names[-2]):
                            k = get_prettylabel(index_names[-2], il1, self.timefmt)
                            b[k] = [get_prettylabel(index_names[-1], v, self.timefmt) for v in df1[index_names[-1]]]
                    self._bookmark = bookmark
                else:
                    self._bookmark = [get_prettylabel(index_names[-1], v, self.timefmt) for v in self._indices.index]
            else:
                return None

        return self._bookmark

    @property
    def flatbookmark(self):
        if type(self.bookmark) is dict:
            flatbookmark = []
            for k, v in self.bookmark.items():
                for vv in v:
                    # flatbookmark.append(str(k).strip().title()+"    "+str(vv).strip())
                    flatbookmark.append(str(k).strip()+"    "+str(vv).strip())
            return flatbookmark
        else:
            return self.bookmark


    @property
    def indices(self):
        # return a dataframe wit the indexes in all elements
        # this should be called after the elements have been added
        if self._indices is None:
            indices = None
            index_names = []
            for e in self.plotdata:
                if e.nplot == 0: continue
                index_names.extend(get_index_names(e.dat))
                indices = pd.concat([indices, pd.DataFrame(True, index=e.dat.index, columns=[e.name]).reset_index()])

            if is_true(indices):
                indices = indices.dropna(how='all').reset_index(drop=True) #.fillna(method='pad')
                index_names = get_unique_unsorted(index_names).tolist()
                if is_true(self.droplevel):
                    for droplevel in iterate(self.droplevel):
                        if droplevel in index_names:
                            index_names.remove(droplevel)
                        else:
                            raise ValueError(f'{droplevel} dimension is not found in {self}')
                # indices.loc[:,~indices.columns.isin(index_names)] = indices.loc[:,~indices.columns.isin(index_names)].fillna(False)
                for c in indices.columns:
                    indices.loc[pd.isna(indices[c]), c] = False

                if index_names:
                    for c in indices:
                        if c in index_names:
                            continue
                        if indices[c].dtype.kind in 'M':
                            indices[c] = (indices[c] == False) | \
                                         (indices[c].isna())
                    # print("indices", indices)
                    indices = indices.groupby(index_names, sort=False).any()
                else:
                    indices = pd.DataFrame()
            else:
                indices = pd.DataFrame()
            if is_true(self.exttable):
                indices = merge_id_df(indices, self.exttable, newindex='left')

            # print(self.name, indices)
            for g in iterate(self.grouping):
                if g and g not in index_names:
                    index_names.append(g)

            self._indices = indices
            if index_names:
                self._indices = indices.reset_index().set_index(index_names[::-1]).sort_index()
        return self._indices

    def setup_movie(self):
        writemovie = self.dict.get('writemovie', False)
        if writemovie:
            if type(writemovie is bool):
                writemovie = self.fullname.replace(":",".") + '.mp4'
            if writemovie.endswith('mp4'):
                moviewriter = FFMpegWriter(**self.dict.get('moviearg', {}))
            elif writemovie.endswith('gif'):
                moviewriter = PillowWriter(**self.dict.get('moviearg', {}))
            else:
                moviewriter = HTMLWriter(**self.dict.get('moviearg', {}))
            moviewriter.setup(self.fig, writemovie, dpi=self.dpi)
            self.moviewriter = moviewriter


    def write_pdf(self, pdf, dpi, verbose):
        self.dpi = self.dict.get('dpi', dpi)
        self.dpi = self.dpi or 200
        self.fig.set(dpi=self.dpi)
        self.pdf = pdf

        self.resize_fig()
        self.setup_movie()
        self.update_plot()

        if self.moviewriter is not None:
            self.moviewriter.finish()
            print('  Animation has been be made.')
        return self.bookmark


    def resize_fig(self):
        resize_fig(self.fig)


    def update_plot(self, *args, **kwargs):

        nplot = 0
        for e in self.plotdata:
            nplot = max(nplot, e.nplot)

        if 'aspect' in kwargs:
            self.ax.set_aspect(kwargs.pop('aspect'))

        index_names = get_index_names(self.indices)
        if nplot>0:
            iplot = -1
            if len(self.indices) > 0:
                # print(self.indices)
                for idx, indice in self.indices.iterrows():
                    iplot += 1
                    idx = {k:v for k,v in zip(index_names,iterate(idx))}
                    print(f'    Writing {self.fullname} {iplot+1:<4} {self.flatbookmark[iplot]}')
                    for e in self.Elements:
                        e.update_plot(idx, indice[e.name])
                    self.saveplot(iplot)
            else:
                print(f'    Writing {self.fullname}')
                for e in self.Elements:
                    e.update_plot(slice(None))
                self.saveplot(0)
        else:
            self.saveplot(0)


    def saveplot(self, iplot):
        ysymmetric = self.ysymmetric
        if 'ylim' not in self.dict and 'extent' not in self.dict:
            self.ax.relim()
            dataLim = self.ax.dataLim
            # if np.isinf(dataLim.y0) or np.isinf(dataLim.y1):
            #     self.ax.set_ybound([1,5])
            avg = (dataLim.y1 + dataLim.y0) / 2
            if dataLim.y1 == dataLim.y0:
                dataLim.y0 = avg - 0.5
                dataLim.y1 = avg + 0.5
            if (dataLim.y1 - dataLim.y0) / abs(dataLim.y0) < 0.001:
                dy = abs(dataLim.y0) * 0.001
                dataLim.update_from_data_y([dataLim.y0-dy, dataLim.y1+dy])
            if ysymmetric:
                vmax = max(abs(dataLim.y0), abs(dataLim.y1))
                dataLim.update_from_data_y([-vmax, vmax])
            yrange = abs(self.dict.get('yrange', 0))
            if dataLim.y1 - dataLim.y0 < yrange:
                y2 = (dataLim.y1 + dataLim.y0) / 2
                dataLim.update_from_data_y([y2-yrange/2, y2+yrange/2])
            self.ax.autoscale_view(True,False,True)
            if 'ticklabel_format' not in self.dict:
                if (abs(dataLim.y1) > 1e5 and abs(dataLim.y0) > 1e5) or (abs(dataLim.y0) < 1e-4 and abs(dataLim.y1) < 1e-4):
                    self.ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(minor_thresholds=(np.inf, np.inf)))


        if 'ylim2' not in self.dict and 'extent' not in self.dict and self.ax1 is not None:
            self.ax1.relim()
            dataLim1 = self.ax1.dataLim
            if ysymmetric:
                vmax = max(abs(dataLim1.y0), abs(dataLim1.y1))
                dataLim1.update_from_data_y([-vmax, vmax])
            self.ax1.autoscale_view(True,False,True)


        # update text in the figure
        if len(self.indices)>0:
            t = self.indices.reset_index().iloc[iplot].to_dict()
            t['idx']=iplot+1
            for text in iterate(self.text):
                if "{" in text._orig and "}" in text._orig:
                    text.set_text(eval("f'"+text._orig+"'", dict(t)))
                else:
                    text.set_text(text._orig)

            # adjust text location if it is inside the axes
            if is_true(self.adjusttext):
                adjust_text(self.adjusttext, ax=self.ax)

        if self.dict.get('writepng', False):
            self.write_png(iplot)
        if self.moviewriter is not None:
            self.moviewriter.grab_frame()
        self.pdf.savefig(self.fig, dpi=self.dpi, bbox_inches='tight')



    def write_png(self, iplot=0):
        pngname = type(self).__name__ + ',' + self.name
        if self.flatbookmark:
            pngname += f',{self.flatbookmark[iplot]}'
        pngname = pngname.strip().replace(':', '-').replace('/', '_').replace('\\', '_') + '.png'
        self.fig.savefig(pngname, dpi=self.dpi, bbox_inches='tight')

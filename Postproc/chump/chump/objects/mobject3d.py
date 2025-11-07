import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LogNorm
from shapely.geometry import LineString
from contourpy import contour_generator
from ..utils.constants import CONTOUR_NLEVEL
from ..utils.misc import (get_index_names, get_prettylabel, is_true, warn_deprecation,
                          iterate_dicts, concise_timefmt)
from ..utils.plotting import line_kwargs, marker_kwargs, add_cax
from .mobject2d import mobject2d


class mobject3d(mobject2d):
    """`mobject3d` represents grid data structures.
    """

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)
        self._x = None
        self._y = None
        self._xvert = None
        self._yvert = None
        self._geom = None
        self._is_structuredgrid = None

        self.plottype: str = self.dict.get('plottype', 'cgrid')
        """plotting type of the grid data, must be 'contour', 'grid', 'cgrid', 'colorgrid', 'colormesh' or 'colorflood'.
        """

        self.norm: str = Normalize if self.dict.get('norm', 'Normalize').lower() == 'normalize' else LogNorm
        """the color scale used for color grid, either 'Normalize' or 'LogNorm', default is 'Normalize' which is linear scale color map.
        """

        self.legend: bool = self.dict.get('legend', {})
        """whether to plot the legend of this object, default is `false`.

        Example:
        >>> legend = true
        """

        if 'colorbar' in self.dict:
            warn_deprecation(self, '`colorbar` is deprecated since v20231214. Please use `colorbararg` instead.')
            self.dict['colorbararg'] = self.dict.pop('colorbar')

        self.colorbararg: dict = self.dict.get('colorbararg', {})
        """the color bar argument for color grid plot.

        Example:
        >>> colorbararg = {shrink=0.88, pad=0.01, label='Simulated groundwater elevation (feet)'}
        """

        self.clabelarg: dict = self.dict.get('clabel', {'fontsize':'x-small'})
        """the contour label argument for contour plot.

        Example:
        >>> clabel = {}
        """

        self.levels: dict = self.dict.get('levels', None)
        """levels used for contour plots.

        Example:
        >>> levels = [0,1,2,5,10,20,50,100,200,300,500]
        """

        self.dlevel: dict = self.dict.get('dlevel', None)
        """level intervel for contour plots.

        Example:
        >>> dlevel = 10
        """

        self.label = self.dict.get('label', self.name)
        """set data label/name used in legend.
        """

        self.writeshp = self.dict.get('writeshp', False)
        """if true, write shapefile of the object.

        Example:
        >>> writeshp = true
        """

        assert self.plottype in ['contour', 'grid', 'cgrid', 'colorgrid', 'colormesh', 'colorflood'], f'Unkown plotting types for {self.fullname}'

    def _loaddata(self, ignoreindex=False):

        # print_callstack()
        super()._loaddata(ignoreindex)
        self._nplot = len(self._dat)

    @property
    def is_structuredgrid(self, ):
        """@private
        """
        if self._is_structuredgrid is None:
            self._loaddata()
        return self._is_structuredgrid

    @property
    def x(self, ):
        """@private
        """
        if self._x is None:
            self._loaddata()
        return self._x

    @property
    def geom(self, ):
        """@private
        """
        if self._geom is None:
            self._loaddata()
        return self._geom

    @property
    def xvert(self, ):
        """@private
        """
        if self._xvert is None:
            self._loaddata()
        return self._xvert

    @property
    def nx(self, ):
        """@private
        """
        if self._xvert is None and self._x is None:
            self._loaddata()
        if self._x is not None:
            return self._x.shape[-1]
        elif self._xvert is not None:
            return self._xvert.shape[-1] - 1

    @property
    def y(self, ):
        """@private
        """
        if self._y is None:
            self._loaddata()
        return self._y

    @property
    def yvert(self, ):
        """@private
        """
        if self._y is None:
            self._loaddata()
        return self._yvert

    @property
    def ny(self, ):
        """@private
        """
        if self._yvert is None and self._y is None:
            self._loaddata()
        if self._y is not None:
            return self.y.shape[0]
        elif self._yvert is not None:
            return self._yvert.shape[0] - 1


    def set_grid(self):
        """@private
        """
        model = self._getobj('model')
        self.gridtype = model.gridtype
        self.is_layered = model.is_layered
        self._geom = model.dat.geometry

        if self.gridtype == 'fe':
            self._is_structuredgrid = False
            self._x = model.vert.x
            self._y = model.vert.y
        else:
            if model.is_structuredgrid:
                self._is_structuredgrid = True
                self._x = model.dat.x.values.reshape([model.nrow, model.ncol])
                self._y = model.dat.y.values.reshape([model.nrow, model.ncol])
                self._xvert = model.xvert
                self._yvert = model.yvert
            else:
                self._is_structuredgrid = False
                self._x = model.dat.x
                self._y = model.dat.y


    def plot(self, *args, **kwargs):
        """@private
        """
        self.clabel = None
        self.contour = None
        self.cgrid = None
        self.node = None
        nplot = self.nplot
        assert self.ax is not None, f'Axes is not defined for {self.fullname}.'
        if self.plottype == 'contour':
            legend = self.plot_contour()
        else:
            legend = self.plot_cgrid()
        return legend

    def vplot(self, vgrid=None, *args, **kwargs):
        """@private
        """
        assert self.ax is not None, f'Axes is not defined for {self.fullname}.'
        self.clabel = None
        self.contour = None
        self.cgrid = None
        self.node = vgrid[['layer','node']]

        nplot = self.nplot

        if self.plottype == 'contour':
            legend = self.vplot_contour(vgrid=vgrid)
        else:
            legend = self.vplot_cgrid(vgrid=vgrid)
        return legend


    def update_plot(self, idx, hasData=True):
        """@private
        """
        if not hasData:
            return
        iidx = pd.Series({k:v for k,v in idx.items() if k in get_index_names(self.dat)})
        if self.plottype == 'contour':
            self.update_contour(iidx)
        else:
            self.update_grid(iidx)

        if self.writeshp:
            self._writeshp(iidx)


    def vupdate(self, idx, hasData=True):
        """@private
        """
        if not hasData:
            return
        iidx = pd.Series({k:v for k,v in idx.items() if k in get_index_names(self.dat)})
        if self.plottype == 'contour':
            self.vupdate_contour(iidx)
        else:
            self.vupdate_grid(iidx)


    def plot_cgrid(self, ):
        """@private
        """
        self.vmin = self.plotarg.pop('vmin') if 'vmin' in self.plotarg else np.nanmin(self.dat.values)
        self.vmax = self.plotarg.pop('vmax') if 'vmax' in self.plotarg else np.nanmax(self.dat.values)
        norm=self.norm(self.vmin, self.vmax)
        if self.is_structuredgrid:
            self.cgrid = self.ax.pcolormesh(self.xvert, self.yvert, np.zeros([self.ny, self.nx]), norm=norm, **self.plotarg)
        else:
            gpd.GeoSeries(self.geom).plot(ax=self.ax, norm=norm, **self.plotarg)
            self.cgrid = self.ax.collections[-1]

        self.colorbar = None
        if is_true(self.colorbararg):
            cax = add_cax(self.ax, self.colorbararg)
            self._colorbar = self.ax.figure.colorbar(self.cgrid, cax=cax, **self.colorbararg)


    def update_grid(self, idx=None, ):
        """@private
        """
        if self.dat.index.nlevels > 1:
            dat = self.dat.xs(tuple(idx), level=tuple(idx.index), ).values
        else:
            dat = self.dat.loc[idx.values].values

        if self.gridtype == 'fe':
            model = self._getobj('model')
            dat = model.get_elem_val( dat)

        zz = dat.flatten()
        if not self.is_layered:
            if self.x.index.nlevels > 1:
                mask = np.array([False] * len(self.x))

                for k, v in idx.items():
                    if k in self.x.index.names:
                        mask[self.x.index.get_level_values(k) == v] = True

            ncell = np.count_nonzero(mask)
            z = np.empty_like(mask, dtype='float')
            z[~mask] = np.nan
            z[ mask] = zz[:ncell]
            zz = z
        self.cgrid.set(array=np.ma.masked_invalid(zz), cmap=self.cgrid.cmap)


    def vplot_cgrid(self, vgrid=None):
        """@private
        """
        dat = self.dat.loc[:, np.unique(self.node['node'])]
        self.vmin = self.plotarg.pop('vmin') if 'vmin' in self.plotarg else np.nanmin(dat.values)
        self.vmax = self.plotarg.pop('vmax') if 'vmax' in self.plotarg else np.nanmax(dat.values)
        if 'aspect' not in self.plotarg:
            aspect = {'aspect':None}
        else:
            aspect = {}
        vgrid.plot(ax=self.ax, norm=self.norm(self.vmin, self.vmax), **self.plotarg, **aspect)
        self.cgrid = self.ax.collections[-1]
        if is_true(self.colorbararg):
            cax = add_cax(self.ax, self.colorbararg)
            self._colorbar = self.ax.figure.colorbar(self.cgrid, cax=cax, **self.colorbararg)


    def vupdate_grid(self, idx):
        """@private
        """
        if self.dat.index.nlevels > 1:
            dat = self.dat.xs(tuple(idx), level=tuple(idx.index), )
        else:
            dat = self.dat.loc[idx.values]


        if self.node is not None:
            dat = [dat.loc[lay, node] for lay, node in self.node.values]

        self.cgrid.set(array=np.ma.masked_invalid(dat).flatten(), cmap=self.cgrid.cmap, ) #norm=self.norm(self.vmin, self.vmax)


    def plot_contour(self, ax=None, idx=0):
        """@private
        """
        legend = None
        if self.legend:
            plot_kwargs = marker_kwargs + line_kwargs
            plot_kw = {kw.rstrip('s'): self.plotarg[kw] for kw in self.plotarg if kw.rstrip('s') in plot_kwargs}
            legend = {self.label: Line2D([], [], **plot_kw)}
            self.legend = True
        return legend

    def vplot_contour(self, grid=None):
        """@private
        """
        self.nlay = int(grid['layer'].max())
        plotargs = iterate_dicts(self.plotarg)
        if len(plotargs) != self.nlay:
            plotargs = plotargs[:1] * self.nlay

        legend = {}
        self.contour = []
        for ilay, plotarg in enumerate(plotargs):
            glay = grid[grid['layer']==ilay+1].sort_values('x0')
            xx = glay.x0.values
            xx = [xx[0]] + np.repeat(xx[1:],2).tolist() + [xx[-1] + glay.iloc[-1]['length']]
            self.contour.append(self.ax.plot(xx,[0]*len(xx),**plotarg)[0])
            legend[self.label + ' Layer ' + str(ilay+1)]=self.contour[-1]

        return legend


    def update_contour(self, idx=None, nodes=None):
        """@private
        """
        if self.dat.index.nlevels > 1:
            dat = self.dat.xs(tuple(idx), level=tuple(idx.index), )
        else:
            dat = self.dat.loc[idx.values]

        plotarg = self.plotarg.copy()
        if self.node is not None:
            dat.loc[:, self.node] = np.nan

        if is_true(self.levels):
            plotarg['levels'] = self.levels
        elif is_true(self.dlevel):
            dlevel = np.abs(self.dlevel)
            dmin = np.nanmin(dat)
            dmax = np.nanmax(dat)
            if self.limits:
                vmin, vmax = self.limits
                if vmax < dmin:
                    vmax = dmax
                if vmin > dmax:
                    vmin = dmin
            else:
                vmin, vmax = dmin, dmax
            levels = np.arange(vmin, vmax+dlevel, dlevel)
            plotarg['levels'] = levels

        if 'levels' not in plotarg:
            plotarg['levels'] = CONTOUR_NLEVEL

        if is_true(self.contour):
            self.contour.remove()

        zz = dat.values.flatten()
        if self.is_structuredgrid:
            self.contour = self.ax.contour(self.x, self.y, zz.reshape(self.y.shape), **plotarg)
        else:
            xx = self.x
            yy = self.y
            if xx.index.nlevels > 1:
                for k, v in idx.items():
                    if k in xx.index.names:
                        xx = xx.xs(v, level=k)
                        yy = yy.xs(v, level=k)

            xx = xx.values.flatten()
            yy = yy.values.flatten()
            # mask = ~np.isnan(zz)
            self.contour = self.ax.tricontour(xx, yy, zz[:len(xx)], **self.plotarg)

        self.contourlines = {"Level":[], "geometry":[]}
        for l, p in zip(self.contour.levels, self.contour.get_paths()):
            vv = p.vertices
            iv = p.codes
            if iv is None:
                continue
            if 1 in iv[1:]:
                vs = np.split(vv, np.where(iv[1:]==1)[0]+1)
            else:
                vs = [vv, ]
            for v in vs:
                self.contourlines["Level"].append(l)
                self.contourlines["geometry"].append(LineString(v))

        self.clabel = self.contour.clabel(**self.clabelarg)
        self.levels = self.contour.levels
        # self.contour = self.contour.collections


    def vupdate_contour(self, idx=None, ):
        """@private
        """

        if self.dat.index.nlevels > 1:
            dat = self.dat.xs(tuple(idx), level=tuple(idx.index), )
        else:
            dat = self.dat.loc[idx.values]
        if self.node is not None:
            dat =dat.loc[:, self.node]


        for ilay in range(self.nlay):
            xx, _ = self.contour[ilay].get_data()
            self.contour[ilay].set_data(xx, dat.loc[ilay+1])


    def _writeshp(self, idx):
        """@private
        """
        if 'time' in get_index_names(self.dat):
            timefmt = concise_timefmt(self.dat.index.get_level_values('time'))
        else:
            timefmt = None
        filename = get_prettylabel(list(idx.index), list(idx.values), timefmt)
        filename = type(self).__name__ + '_' + self.name+'_'+ filename
        filename = filename.strip().replace(':', '_').replace('/', '_').replace('\\', '_') + '.shp'

        if self.contour:
            gpd.GeoDataFrame(self.contourlines, crs=self.crs).to_file(filename)

        if self.cgrid:
            gpd.GeoDataFrame({self.name:self.cgrid.get_array().flatten()}, crs=self.crs, geometry=self.geom).to_file(filename)

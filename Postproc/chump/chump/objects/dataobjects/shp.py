
import geopandas as gpd
import numpy as np
from adjustText import adjust_text
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pandas.api.types import is_numeric_dtype

from ...utils.plotting import line_kwargs, marker_kwargs, patch_kwargs, zorders
from ...utils.misc import get_index_names, is_true, warn_deprecation
from .table import table

#TODO 1: add legend for catgorical data
#TODO 2: colors defined by a column
#TODO 3: label same color as the points

_gdf_func = ['dissolve', 'explode', 'sjoin', 'clip',]
_geometry_func = ['buffer', 'simplify', 'affine_transform', 'rotate', 'scale', 'scale', 'skew', 'translate']
_geometry_prop_ = ['centroid', 'concave_hull', 'convex_hull', ]

class shp(table):
    '''
    define a shapefile object and how it will be plotted
    '''

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        # data
        self.source    = self.source

        self.a_scale   = self.a_scale
        self.a_offset  = self.a_offset
        self.limits    = self.limits
        self.abslimits = self.abslimits
        self.add       = self.add
        self.subtract  = self.subtract
        self.mul       = self.mul
        self.div       = self.div
        self.calculate = self.calculate
        self.extfunc   = self.extfunc

        # plotting
        self.plotarg   = self.plotarg

        self.size: str = self.dict.get('size')
        """name of the column representing the symbol size when plotting the shapefile
        """

        self.sizescale: float = self.dict.get('sizescale', 1.0)
        """a factor to be multiplied with the `size` column. Default is 1.0.
        """

        self.sizelegendarg: float = self.dict.get('sizelegendarg', {})
        """argument used for the sizes in the legend.
        """

        self.pointlabel: str = self.dict.get('pointlabel')
        """name of the colum used to label the point features when plotting the shapefile.

        Example:
        >>> pointlabel = 'WellName'
        """

        self.pointlabelarg: str = self.dict.get('pointlabelarg')
        """plotting argument for the point labels

        Example:
        >>> pointlabelarg = {fontsize='x-small'}
        """

        self.plotarg_loc = self.dict.get('plotarg_loc', {})
        """plotting argument for a highlighted site when looping sites in a `tsplot`
        """


        self.dissolve = self.dict.get('dissolve')
        """Dissolve geometries within groupby into single observation. This is accomplished by applying the unary_union method to all geometries within a groupself.
        See https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.dissolve.html.

        Example:
        >>> dissolve = {by='hub8', aggfunc='sum'}
        """

        self.explode = self.dict.get('explode')
        """Explode multi-part geometries into multiple single geometries.
        Each row containing a multi-part geometry will be split into multiple rows with single geometries,
        thereby increasing the vertical size of the GeoDataFrame.
        See https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.explode.html.

        Example:
        >>> explode = true
        """

        self.sjoin = self.dict.get('sjoin')
        """Spatial join with another geospatial object.
        See https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.sjoin.html.

        Example:
        >>> sjoin = {df={shp='modelgrid'}}
        """

        self.clip = self.dict.get('clip')
        """Clip points, lines, or polygon geometries by another geospatial object.
        See https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.clip.html.

        Example:
        >>> clip = {mask={shp='modelarea'}}
        """

        self.buffer = self.dict.get('buffer')
        """Create buffer of distance of the geospatial object.
        See https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.buffer.html.

        Example:
        >>> buffer = {distance=10}
        """

        self.centroid = self.dict.get('centroid', False)
        """whether to get the centroid. default is false
        See https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.centroid.html.

        Example:
        >>> centroid = true
        """

        self.concave_hull = self.dict.get('concave_hull', False)
        """whether to get the concave hull of each geometry. default is false
        See https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.concave_hull.html.

        Example:
        >>> concave_hull = true
        """

        self.convex_hull = self.dict.get('convex_hull', False)
        """whether to get the convex hull of each geometry. default is false
        See https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.convex_hull.html.

        Example:
        >>> convex_hull = true
        """

        self.simplify = self.dict.get('simplify', None)
        """Get a simplified representation of each geometry.
        See https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.simplify.html.

        Example:
        >>> simplify = {tolerance=100}
        """

        self.affine_transform = self.dict.get('affine_transform', None)
        """Get a translated geometry. For 2D affine transformations, the 6 parameter matrix is [a, b, d, e, xoff, yoff]
        See https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.affine_transform.html.

        Example:
        >>> affine_transform = [2, 3, 2, 4, 5, 2]
        """

        self.rotate = self.dict.get('rotate', None)
        """Get a rotated geometry with a angle in degree.
        Positive angles are counter-clockwise and negative are clockwise rotations.
        See https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.rotate.html.

        Example:
        >>> rotate = 90
        >>> rotate = {angle=90, origin=[0,0]}
        """

        self.scale = self.dict.get('scale', None)
        """Get a scaled geometry by different factors along each dimension.
        See https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.scale.html.

        Example:
        >>> scale = [0.01, 0.01]                             # scaled x and y 1%
        >>> scale = {xfact=0.01, yfact=0.01, origin=[0, 0]}  # scaled x and y 1%
        """

        self.skew = self.dict.get('skew', None)
        """Get a skewed geometry sheared by angles (in degrees) along the x and y dimensions.
        See https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.skew.html.

        Example:
        >>> skew = [45, 30]
        >>> skew = {xs=45, ys=30, origin=[100, 100]}
        """

        self.translate = self.dict.get('translate', None)
        """Get a translated geometry by offset along each dimension.
        See https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.translate.html.

        Example:
        >>> translate = [2, 3]
        >>> translate = {xoff=45, yoff=30}
        """

        self.to_file = self.dict.get('to_file')
        """Write the geodataframe to a shapefile.
        See https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.to_file.html.

        Example:
        >>> to_file = 'grid.shp'
        """

        self.zcol = self.dict.get('zcol')
        """set the column name defining the Z position, used in `vplot`.
        """

        # output
        self.writedata = self.writedata
        self.writefunc = self.writefunc


    def _apply_gdffunc(self, ):
        if is_true(self.sjoin):
            self.sjoin['df'] = self._getobj(self.sjoin['df']).dat
        if is_true(self.clip):
            self.clip['mask'] = self._getobj(self.clip['mask']).dat

        for f in _gdf_func:
            if is_true(getattr(self, f, None)):
                if isinstance(getattr(self, f), dict):
                    self._dat = getattr(self.dat, f)(**getattr(self, f))
                elif isinstance(getattr(self, f), list):
                    self._dat = getattr(self.dat, f)(*getattr(self, f))
                else:
                    self._dat = getattr(self.dat, f)(getattr(self, f))

        for f in _geometry_func:
            if is_true(getattr(self, f, None)):
                if isinstance(getattr(self, f), dict):
                    self._dat.geometry = getattr(self.dat, f)(**getattr(self, f))
                elif isinstance(getattr(self, f), list):
                    self._dat.geometry = getattr(self.dat, f)(*getattr(self, f))
                else:
                    self._dat.geometry = getattr(self.dat, f)(getattr(self, f))

        for f in _geometry_prop_:
            if is_true(getattr(self, f, None)):
                self._dat.geometry = getattr(self.dat, f)


    def readdata(self):
        super().readdata()
        self._apply_gdffunc()
        # self._nplot = len(self._dat) if 'plotarg_loc' in self.dict else 0
        self._nplot = 0

    def add_legend(self, ):
        """@private"""
        legend = {}
        # create the legend
        if self.legend:

            # when plotting by "column" these situation will be handled by GeoPandas
            label = self.legendlabel

            geo_type = self.dat.geom_type.iloc[0].lower()
            # check type
            if 'point' in geo_type:
                # collect plot kwargs
                plot_kwargs = marker_kwargs
                plot_kw = {kw: self.plotarg[kw] for kw in plot_kwargs if kw in self.plotarg}
                if 'marker' not in plot_kw:
                    plot_kw['marker']="o"
                plot_kw['linewidth'] = 0
                if 'facecolor' in plot_kw:
                    plot_kw['markerfacecolor'] = plot_kw.pop('facecolor')
                if 'markersize' in plot_kw:
                    plot_kw['markersize'] = 10
                if 'edgecolor' in plot_kw:
                    plot_kw['markeredgecolor'] = plot_kw.pop('edgecolor')
                else:
                    plot_kw['markeredgecolor'] = 'none'
                legend[label] = Line2D([], [], **plot_kw)
            elif 'line' in geo_type:
                # collect plot kwargs
                plot_kwargs = marker_kwargs + line_kwargs
                plot_kw = {kw: self.plotarg[kw] for kw in plot_kwargs if kw in self.plotarg}
                legend[label] = Line2D([], [], **plot_kw)
            elif 'polygon' in geo_type:
                # collect plot kwargs
                plot_kwargs = patch_kwargs + line_kwargs
                plot_kw = {kw: self.plotarg[kw] for kw in plot_kwargs if kw in self.plotarg}
                if 'color' in plot_kw:
                    c = plot_kw.pop('color')
                if 'facecolor' not in plot_kw and 'fc' not in plot_kw:
                    plot_kw['fc'] = c
                if 'edgecolor' not in plot_kw and 'ec' not in plot_kw:
                    plot_kw['ec'] = c
                legend[label] = Patch(**plot_kw)
            else:
                raise TypeError(f'Unknow shp type {shp.geom_type.iloc[0]}')

        return legend

    def plot(self, *args, **kwargs):
        """@private"""
        self.collection = None
        plotarg = self.plotarg.copy()

        if self.size:
            sizevals = self.dat[self.size].abs()
            if self.dat.shape[0] > 20: # five
                sizes = sizevals.quantile([0.1, 0.3, 0.5, 0.7, 0.9])
            elif self.dat.shape[0] > 5:
                sizes = sizevals.quantile([0.05, 0.35, 0.65, 0.95])
            else:
                sizes = np.sort(sizevals)
            # change the size values
            sizes = list(sizes)
            sizevals = sizevals.where(sizevals>sizes[0], sizes[0]).where(sizevals<sizes[-1], sizes[-1])
            plotarg['markersize'] = np.log(sizevals + 1) * self.sizescale

        legend_created = False
        if 'column' in plotarg:
            legend_created = True
            if 'legend' not in plotarg:
                plotarg['legend'] = True
            self.legend=False
            self.colorbar = None
            if self.plotarg.get('categorical', False):
                c = plotarg['column']
                if is_numeric_dtype(self.dat[c]):
                    plotarg['column'] = c + "__"
                    self.dat[c + "__"] = c + ' ' + self.dat[c].astype(str)

        dat = gpd.GeoDataFrame(self.dat)
        if getattr(dat, 'crs', None) is not None and getattr(self.fig, 'crs', None) is not None:
            dat.to_crs(self.fig.crs, inplace=True)
        ax = dat.plot(ax=self.ax, **plotarg)
        self.collection = ax.collections[-1]

        if len(self.plotarg_loc) > 0: # plot the features if it is used to update locations
            plotarg_loc = {k:v for k,v in self.plotarg_loc.items()}
            plotarg_loc['alpha'] = 0
            ax = dat.plot(ax=self.ax, **plotarg_loc)
            self.collection = ax.collections[-1]

        if 'point' in dat.geom_type.iloc[0].lower():
            labelpoint = self.pointlabel
            labelarg = self.pointlabelarg
            if is_true(labelpoint):
                if 'zorder' not in labelarg:
                    labelarg['zorder'] = zorders['labelpoint']
                if 'fontsize' not in labelarg:
                    labelarg['fontsize'] = 'x-small'
                if labelpoint:
                    welllabels = [ax.text(r.x, r.y, r[labelpoint], **labelarg) for i, r in dat.iterrows()]
                    adjust_text(welllabels, ax=self.ax, )

        if 'markersize' in self.plotarg and is_true(self.sizelegendarg):
            labels = [f'{s:.1f}' for s in sizes]
            legs = [ax.scatter([],[], s=np.log(s + 1) * self.sizescale, c='grey') for s in sizes]
            leg_err = ax.legend(legs, labels, ncol=len(sizes), **self.sizelegendarg)
            ax.add_artist(leg_err)

        if is_true(ax.legend_) and legend_created:
            if len(ax.legend_.texts) > 0:
                legend = {}
                for t, e in zip(ax.legend_.texts, ax.legend_.legend_handles):
                    legend[t._text] = e
                ax.legend_.remove()
                return legend
        return self.add_legend()

    def update_plot(self, idx=None, hasData=True):
        if self.collection:
            self.collection.remove()
            self.collection = None
        if not hasData:
            return
        if idx is not None:
            index_names = get_index_names(self.dat)
            id = {k:v for k, v in idx.items() if k in index_names}
            if len(index_names) > 1:
                selected = self.dat.xs(id.values(), level=id.keys())
            else:
                selected = self.dat.loc[id.values()]
            selected = gpd.GeoDataFrame(selected, crs=self.dat.crs)
            if selected.crs is not None and getattr(self.fig, 'crs', None) is not None:
                selected = selected.to_crs(self.fig.crs)
            ax=selected.plot(ax=self.ax, **self.plotarg_loc)
            self.collection = ax.collections[-1]


    def vplot(self, *args, **kwargs):
        """@private"""
        assert is_true(self.zcol), f'Must define the Z column for {self.fullname}'
        shp = self.dat
        if shp.crs is not None and self.crs is not None:
            shp.to_crs(self.crs, inplace=True)
        lengths = shp.length
        xx = np.cumsum([0] + list(lengths))
        if 'color' not in self.plotarg:
            self.plotarg['color'] = 'k'
        if 'linewidth' not in self.plotarg:
            self.plotarg['linewidth'] = 1.5

        plotarg = {k:v for k, v in self.plotarg.items() if (k != 'legend' or 'column' not in self.plotarg)}
        segs = [[[x0,z],[x1,z]] for x0,x1,z in zip(xx[:-1], xx[1:], shp[self.zcol])]
        lines = LineCollection(segs, **plotarg)
        self.ax.add_collection(lines)
        return self.add_legend()


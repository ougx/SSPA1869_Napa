
import geopandas as gpd
import pandas as pd
from matplotlib.colors import Normalize, LogNorm

from ...utils.plotting import add_cax
from ...utils.misc import get_index_names, merge_id_df
from .shp import shp

class shpts(shp):
    '''
    define a shapefile object with time-varying property defined by a tseries object
    '''

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        # data
        self.source    = self.source
        self.tseries: dict   = self.dict.get('tseries')
        """set the `tseries` object for this shapefile.
        The values of the shapefile will be updated using this `tseries` object.

        Example:
        >>> [shpts.ETtseries]
        >>> source = 'ETzone.shp'
        >>> tseries = {tseries='ET'}
        """

        self.norm: str = Normalize if self.dict.get('norm', 'Normalize').lower() == 'normalize' else LogNorm
        """the color scale used for color grid, either 'Normalize' or 'LogNorm', default is 'Normalize' which is linear scale color map.
        """

        if 'colorbar' in self.dict:
            warn_deprecation(self, '`colorbar` is deprecated since v20231214. Please use `colorbararg` instead.')
            self.dict['colorbararg'] = self.dict.pop('colorbar')

        self.colorbararg: dict = self.dict.get('colorbararg', {})
        """the color bar argument.

        Example:
        >>> colorbar = {shrink=0.88, pad=0.01, label='Simulated groundwater elevation (feet)'}
        """

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
        self.plotarg = self.plotarg
        self.size = self.size
        self.sizescale = self.sizescale
        self.sizelegendarg = self.sizelegendarg
        self.pointlabel = self.pointlabel
        self.pointlabelarg = self.pointlabelarg
        self.legend = self.legend
        self.legendlabel = self.legendlabel
        self.dissolve = self.dissolve
        self.explode = self.explode
        self.sjoin = self.sjoin
        self.clip = self.clip
        self.buffer = self.buffer
        self.centroid = self.centroid
        self.concave_hull = self.concave_hull
        self.convex_hull = self.convex_hull
        self.simplify = self.simplify
        self.affine_transform = self.affine_transform
        self.rotate = self.rotate
        self.scale = self.scale
        self.skew = self.skew
        self.translate = self.translate
        self.zcol = self.zcol
        # output
        self.writedata = self.writedata
        self.writefunc = self.writefunc
        self.to_file = self.to_file


    def readdata(self, ):
        super().readdata()
        tseries = self._getobj(self.tseries)
        columns = list(self.dat.columns)
        columns.append(tseries.labels[0])
        self._dat = gpd.GeoDataFrame(merge_id_df(self.dat, tseries.dat, )
                                     ).reset_index().set_index(tseries.timecol).loc[:,columns]
        self._nplot = len(pd.unique(self._dat.index.get_level_values(0)))


    def plot(self):

        super().plot()
        if 'column' not in self.plotarg:
            self.plotarg['column'] = self.dat.columns[-1]
        self.colorbar = None
        self.cax = add_cax(self.ax, self.colorbararg)

    def update_plot(self, idx=None, hasData=True):
        if self.collection:
            self.collection.remove()
            if self.colorbar:
                self.colorbar.remove()
                self.colorbar = None
        if not hasData:
            return

        self.cax.clear()

        if idx is not None:
            index_names = get_index_names(self.dat)
            id = {k:v for k, v in idx.items() if k in index_names}
            if len(index_names) > 1:
                selected = self.dat.xs(id.values(), level=id.keys())
            else:
                selected = self.dat.loc[id.values()]
            plotarg = {k:v for k,v in self.plotarg.items()}
            if ('vmin' not in self.plotarg or 'vmax' not in plotarg):
                vals = selected[self.dat.columns[-1]]
                plotarg['norm'] = self.norm(vals.min(), vals.max())
            ax = gpd.GeoDataFrame(selected).plot(ax=self.ax, **plotarg)
            self.collection = ax.collections[-1]
            self.colorbar = self.ax.figure.colorbar(self.collection, cax=self.cax, **self.colorbararg)


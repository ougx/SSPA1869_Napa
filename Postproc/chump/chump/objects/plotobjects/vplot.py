import geopandas as gpd
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from shapely.geometry import Point, LineString, Polygon

from .figure import figure
from ..dataobjects.shp import shp
from ...utils.misc import iterate, get_prettylabel, concise_timefmt, merge_id_df, get_index_names
from .mapplot import mapplot

from matplotlib.lines import Line2D

from ...utils.plotting import line_kwargs, marker_kwargs
import pandas as pd

class vplot(figure):
    """
    **`vplot`** is a subclass of **`figure`**. It is used to make vertical transect.
    Belows are the options:

    - *profile*: a dictionary define the profile line which will used to intersect with the vgrid to create the transect.
                 It can be points such as `{points=[[x0,y0],[x1,y1], .. [xn,yn]]}` or a table-like object such as `{shp='streams'}`.
                 If it is the profile is a shp object and has the columns of `length`, `row` and `column` for structured vgrid or `node` for unstructured vgrid, these columns will be used to
    - *model*: a dictionary define a model object (such as MODFLOW) that defines the 3D vgrid, e.g. model = {mf='model2000'}.
    - *plotdata*: dictionary obsts, simts, pstres, mflst
    - [*showprofile*]: a booleam value whether to plot the profile; the `zcol` must be defined.
    - [*profilelabel*]: a name for the profile in the legend.
    - [*zcol*]: the column nmae in the profile table to define the Z position.
    - [*dist_scale*]: the scaling factor for distance/length, for example, 1/5280 to convert feet to mile or 0.001 for conversion from meter to kilometer.
    - [*mapplot*]: the name of a **`mapplot`** object to show the profile locations.
    - [*showgrid*]: whether to show the vgrid.
    - [*plotarg_grid*]: a dictionary of plotting arguments for the vgrid

    """
        # - [*distcol*]: the column nmae in the profile table to define distance for each section from the starting point of the profile.
        #            If not defined, it is the cumulative length calculated from the geometry

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)
        self._profile = None
        self.axmap = None
        self._vgrid = None
        self.droplevel = ['layer']

    def readdata(self):
        self.model = None
        for k, v in self.dict.get('model').items():
            self.model = getattr(self.parent, k)[v]
            break

        for k, v in self.dict.get('profile').items():
            if k == 'points':
                profile = gpd.GeoDataFrame(geometry=[LineString(v)])
                self.dict['showprofile'] = False
                hascell = False
                haslength = False
            else:
                profile = getattr(self.parent, k)[v].dat
                columns = profile.columns
                hascell = False
                haslength = 'length' in columns
                if self.model.is_structuredgrid:
                    hascell = ('row' in columns) and ('column' in columns)
                else:
                    hascell = 'node' in columns
            break

        # get the intersecting cells if necessary
        if not (hascell and haslength):
            assert profile.geom_type.isin(['LineString',]).all(), f'Profile must be LineString features for {self.fullname}.'
            if not hascell:
                # profile = profile[[c for c in profile.columns if c.lower() not in ('row', 'column', 'node')]]
                cline = profile.geometry.iloc[0]

                for ilay in range(max(self.model.nlay,1)+1):
                    profile.drop('botm'+str(ilay), axis=1, errors='ignore', inplace=True)
                profile.drop(['row', 'column', 'node', 'layer'], axis=1, errors='ignore', inplace=True)
                profile = gpd.overlay(profile, self.model.dat.reset_index())
                haslength = False

            if not haslength:
                profile['length'] = profile.length

        # reorder and filter small sections
        profile['length'] *= self.dict.get('dist_scale', 1.0)
        profile = profile[profile['length'] / profile['length'].sum() > 0.001]
        dist = np.cumsum(profile['length'])
        profile['x1'] = dist
        profile['x0'] = profile['x1'] - profile['length']
        self._profile = profile


    @property
    def profile(self):
        if self._profile is None:
            self._loaddata()
        return self._profile

    def setup_ax(self, fig=None):
        '''
        create map for an axes
        '''

        super().setup_ax(fig,)
        if 'xlabel' not in self.dict:
            self.ax.set(xlabel='Distance')
        if 'ylabel' not in self.dict:
            self.ax.set(xlabel='Elevation')


    @property
    def vgrid(self):
        if self._vgrid is None:
            # set up vgrid
            geometry = []
            if self.model.is_layered:
                rows = []
                for ilay in range(self.model.nlay):
                    for i, r in self.profile[['x0', 'length', f'botm{ilay}', f'botm{ilay+1}', 'node']].iterrows():
                        x0 = r.iloc[0]
                        x1 = x0 + r.iloc[1]
                        y0 = r.iloc[2]
                        y1 = r.iloc[3]
                        r['layer'] = ilay+1
                        rows.append([ilay+1,int(r['node'])])
                        geometry.append(Polygon([(x0,y0),(x0,y1),(x1,y1),(x1,y0)]))

                vgrid = gpd.GeoDataFrame(rows, columns='layer node'.split(), geometry=geometry)
            else:
                geometry = self.profile.apply(lambda r: Polygon([(r['x0'],r['botm0']),(r['x0'],r['botm1']),(r['x0']+r['length'],r['botm1']),(r['x0']+r['length'],r['botm0'])]), axis=1)
                vgrid = gpd.GeoDataFrame(self.profile[['layer', 'node']], geometry=geometry)

            vgrid = vgrid.sort_values(['layer', 'node'])
            centroid = vgrid.centroid
            vgrid['x'] = centroid.x
            vgrid['y'] = centroid.y
            self._vgrid = vgrid
        return self._vgrid

    def add_elements(self):

        self.readdata()

        self.axmap = None
        if 'mapplot' in self.dict:
            self.map = mapplot(self.dict['mapplot'], self.parent)
            # for i in range(999):
            #     if 'plotdata' + str(i) not in self.dict:
            #         self.dict()
            self.map.initialize_plot(self.fig, )
            self.axmap = self.map.ax
            self.map.legend['Transect'] = self.profile.plot(ax=self.axmap, **self.plotarg).collections[-1]
            self.text.extend(self.map.text)
            self.plotdata.extend(self.map.plotdata)


        for this_element in self.plotdata:
            if this_element.ax == self.ax:
                legend = this_element.vplot(vgrid=self.vgrid)
                if legend is not None:
                    self.legends.update(**legend)
            if this_element.nplot > 0:
                self.Elements.append(this_element)

        zorder = max([_.zorder for _ in self.ax.get_children()])
        if self.dict.get('showgrid'):
            zorder += 1
            plotarg = self.dict.get('plotarg_grid', {}).copy()
            plotarg['zorder'] = zorder
            if 'aspect' not in plotarg:
                plotarg['aspect'] = None
            self.vgrid.plot(ax=self.ax, column='layer', **plotarg)

        # show profile
        if self.dict.get('showprofile', False):
            assert 'zcol' in self.dict, f'Must define `zcol` to show the profile {self.fullname}.'
            zorder += 1
            zcol = self.profile[zcol]
            xx = []
            zz = []
            for i, (x0,l, z) in enumerate(self.profile[['x0', 'length', zcol]].values):
                xx.extend([x0, x0+l])
                zz.extend([z, z])

            plotarg = self.plotarg.copy()
            plotarg['zorder'] = zorder
            if 'aspect' not in plotarg:
                plotarg['aspect'] = None
            gpd.GeoSeries([LineString(np.array([xx, zz]).T)]).plot(ax=self.ax, **plotarg)
            plot_kwargs = marker_kwargs + line_kwargs
            plot_kw = {kw: self.plotarg[kw] for kw in plot_kwargs if kw in plotarg}

            self.legends[self.dict.get('profilelabel', 'Profile')] = Line2D([], [], **plot_kw)


    def update_plot(self, *args, **kwargs):

        # self.ax.set_aspect(self.dict.get('aspect', 'auto'))
        # self.create_bookmark()
        # self.resize_fig()
        # self.setup_movie()
        index_names = get_index_names(self.indices)
        if len(self.indices)>0:
            iplot = -1
            for idx, indice in self.indices.iterrows():
                iplot += 1
                idx = {k:v for k,v in zip(index_names,iterate(idx))}
                print(f'    Writing {self.fullname} {iplot+1:<4} {self.flatbookmark[iplot]}')
                for e in self.Elements:
                    if e.ax == self.ax:
                        e.vupdate(idx, indice[e.name])
                    else:
                        e.update_plot(idx, indice[e.name])

                self.saveplot(iplot)
        else:
            self.saveplot(0)

import numpy as np
import pandas as pd
import geopandas as gpd

from ..objects.mobject import mobject
from ..utils.misc import warn_deprecation
from ..utils.io import readarray
from ..utils.math import bilinear_interpolation_factors, InterpolationCoeffs
from ..utils.spatial import transform_grid
from shapely.geometry import Point


def search_intersect_cells(grid, cell):
    """@private"""
    cells_adjacent = gpd.sjoin(grid, gpd.GeoDataFrame(geometry=[cell.geometry]), )
    nvert = len(cells_adjacent)
    xx = cells_adjacent['x']
    yy = cells_adjacent['y']
    nodes = cells_adjacent.index
    return nvert, xx, yy, nodes


class prepare(mobject):
    """
    `prepare` is a command used to calculate the horizontal interpolation weights based on well location and model grid for the `prepare` objects defined in the configuration file.
    For structured grids, bilinear interpolation is used.
    For unstructured grids, kriging interpolation is used.

    Example:
    >>> chump.exe prepare example.toml
    """


    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        self.model: dict = self.dict['model']
        """ `model` set the model object.

        Example:
        >>> model = {mf = 'gwf'}
        >>> model = {iwfm = 'C2VSimCG'}
        """


        if 'wells' in self.dict:
            warn_deprecation(self, '`wells` is deprecated since v20231214. Please use `sites` instead.')
            self.dict['sites'] = self.dict.pop('wells')

        self.sites: dict = self.dict['sites']
        """ `sites` set the object representing wells/sites.

        Example:
        >>> sites = {csv = 'obswells'}
        >>> sites = [{csv = ['obswells0', 'obswells1']}, {shp = 'obswells2'}]
        """


    def run(self):

        print(f'Prepare hydrograph well file {self.name} ...')

        wells  = self._getobj(self.sites)
        model = self._getobj(self.model)


        nl, nr, nc = max(model.nlay, 1), max(model.nrow, 1), model.ncol

        wellprop = wells.dat
        if wells.crs is not None and model.crs is not None:
            wellprop = wellprop.to_crs(model.crs)
            wellprop['x'] = wellprop.geometry.x
            wellprop['y'] = wellprop.geometry.y

        ibound = model.ibound

        wellcolumns = list(wellprop.columns.str.lower())

        needlayer = True
        collay = None
        layer = None

        if 'layer' in wellcolumns:
            collay = wellcolumns.index('layer')
            layer = wellprop.iloc[:, collay].values.astype(int)

        needlayer = False
        if needlayer and 'avgmethod' in wellcolumns:
            avgmethod = wellprop.iloc[:, wellcolumns.index('avgmethod')]
            needlayer = any(avgmethod.str.lower()=='layer')


        if needlayer and model.nlay > 0 and ibound is not None:
            pass
        else:
            needlayer = False

        wellprop['nInterp'] = 0

        if model.is_structuredgrid:
            if 'x' not in wellprop.columns or 'y' not in 'x' not in wellprop.columns:
                assert wellprop.geometry.geom_type.iloc[0] == 'Point', f'x or y is not in the table of {wells.fullname} and it does not have Point geometry'
                wellprop['x'] = wellprop.geometry.x
                wellprop['y'] = wellprop.geometry.y

            wellprop['localx'], wellprop['localy'] = transform_grid(
                wellprop['x'], wellprop['y'], xoff=model.xoff, yoff=model.yoff, angrot=model.angrot, inverse=True)

            ibound = ibound.reshape([nl, nr, nc])
            if layer is None:
                layer = np.ones(len(wellprop), dtype=int)

            wellprop1 = bilinear_interpolation_factors(wellprop['localx'], wellprop['localy'], model.dx, model.dy, layer, ibound)
            indexname = wellprop.index.name
            wellprop = pd.concat([wellprop.reset_index(), wellprop1], axis=1)
            wellprop.set_index(indexname, inplace=True)

            wellprop ['Node']  = (wellprop['Row']  -1) * nc + wellprop['Column']
            wellprop ['Node1'] = (wellprop['Row1'] -1) * nc + wellprop['Column1']
            wellprop ['Node2'] = (wellprop['Row2'] -1) * nc + wellprop['Column2']
            wellprop ['Node3'] = (wellprop['Row3'] -1) * nc + wellprop['Column3']
            wellprop ['Node4'] = (wellprop['Row4'] -1) * nc + wellprop['Column4']
            wellprop['nInterp'] = 4

        else:
            grid = model.dat
            for i, r in wellprop.iterrows():
                r.index = [ii.lower() if isinstance(ii, str) else ii  for ii in r.index ]
                if not model.is_layered:
                    grid = model.dat.xs(r['layer'], level='layer')
                cell = gpd.sjoin(grid, gpd.GeoDataFrame(geometry=[Point(r.x, r.y)], crs=model.crs)).iloc[0]
                if model.gridtype=='fd':
                    nvert, xx, yy, nodes = search_intersect_cells(grid, cell)
                else:
                    nvert = cell.nvert
                    nodes = cell['vert1':'vert'+str(nvert)].values.astype(int)
                    xx = model.vert.iloc[nodes-1].x
                    yy = model.vert.iloc[nodes-1].y

                print(f'    Interpolating {i} from nodes: {list(nodes)}')
                weights = InterpolationCoeffs(nvert, r.x, r.y, xx.values, yy.values)

                j = 0
                for n, w in zip(nodes, weights):
                    if w == 0:
                        continue
                    j += 1
                    wellprop.loc[i, 'Weight'+str(j)] = w
                    wellprop.loc[i, 'Node'  +str(j)] = n

                wellprop.loc[i, 'nInterp'] = j

        for c in wellprop:
            for w in ['layer', 'row', 'column', 'node', 'ninterp', 'ibound']:
                if str(c).lower().startswith(w):
                    wellprop[c] = wellprop[c].fillna(0).astype(int, errors='ignore')
            if str(c).lower().startswith('weight'):
                wellprop[c] = wellprop[c].fillna(0)

        wellprop.drop('geometry',axis=1).to_csv(f'{self.name}.csv')

        print(f'  New wellprop created at {self.name}.csv')

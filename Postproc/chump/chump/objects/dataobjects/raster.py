import numpy as np
import pandas as pd
from osgeo import gdal

from .modelarray import modelarray


class raster(modelarray):
    """read a raster dataset e.g. *.tif. For other parameters, see `.modelarray`."""
    def readdata(self):
        src = gdal.Open(self.source)
        # GT(0) x-coordinate of the upper-left corner of the upper-left pixel.
        # GT(1) w-e pixel resolution / pixel width.
        # GT(2) row rotation (typically zero).
        # GT(3) y-coordinate of the upper-left corner of the upper-left pixel.
        # GT(4) column rotation (typically zero).
        # GT(5) n-s pixel resolution / pixel height (negative value for a north-up image).
        x0, dx, dxdy, y0, dydx, dy = src.GetGeoTransform()
        dat = src.ReadAsArray()
        if src.RasterCount > 1:
            for i in range(src.RasterCount):
                dat[i] = np.where(dat==src.GetRasterBand(i+1).GetNoDataValue(), np.nan, dat[i])
        else:
            dat = np.where(dat==src.GetRasterBand(1).GetNoDataValue(), np.nan, dat)
        nrow, ncol= dat.shape[-2:]
        nlay = src.RasterCount

        self._dat = pd.DataFrame(dat.reshape([nlay, -1]), index=range(1, nlay+1)) #TODO now just assume it is on layer
        self._dat.index.name = 'layer'

        xx, yy = np.meshgrid(np.arange(0, ncol+1), np.arange(0, nrow+1))
        self._xvert = x0 + xx * dx   + yy * dxdy
        self._yvert = y0 + xx * dydx + yy * dy

        self._x = (self._xvert[1:, 1:] + self._xvert[1:, :-1]) / 2
        self._y = (self._yvert[1:, 1:] + self._yvert[:-1, 1:]) / 2

        self._nplot = nlay
        self._is_structuredgrid = True
        self.is_layered = True
        self.gridtype = 'fd'

        del src



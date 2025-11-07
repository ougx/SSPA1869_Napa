
from ...utils.misc import is_true
from ..dataobjects import shp

import numpy as np

class model(shp):
    """`model` is the parent class for all the model objects."""
    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        self.model_ws = self.dict.get('model_ws', '.')
        """set the model working directory. Default is '.'
        """

        self._nlay = None
        self._nrow = None
        self._ncol = None
        self._vert = None
        self._ibound = None

        self.gridtype = 'fd'  # finite difference
        """@private"""

        self.xmin   = self.dict.get('xmin', 0.0)
        """set the X coordinate (in CRS unit) of the lowerleft corner of the model grid, default is 0
        """

        self.ymin   = self.dict.get('ymin', 0.0)
        """set the Y coordinate (in CRS unit) of the lowerleft corner of the model grid, default is 0
        """

        self.rotation = self.dict.get('rotation', 0.0)
        """set the rotation (in degree) of the model grid, default is 0
        """

        self.lfactor = self.dict.get('lfactor', 1.0)
        """set length unit conversion factor from model to coordinate reference system

        Example:
        >>> lfactor = 0.3048
        """

        self.writeshp = self.dict.get('writeshp', False)
        """set to true to write a shapefile of the model grid
        """

    def _loaddata(self, ):
        print(f'  Loading {self.fullname}')
        self._nplot = 0


    def _writeshp(self):
        if self.writeshp:
            filename = type(self).__name__ + '_' + self.name
            filename = filename.strip().replace(':', '_').replace('/', '_').replace('\\', '_') + '.shp'
            self.dat.to_file(filename, )


    @property
    def nlay(self, ):
        """@private"""
        if self._nlay is None:
            self._loaddata()
        return self._nlay

    @property
    def nrow(self, ):
        """@private"""
        if self._nrow is None:
            self._loaddata()
        return self._nrow

    @property
    def ncol(self, ):
        """@private"""
        if self._ncol is None:
            self._loaddata()
        return self._ncol

    @property
    def ncell(self, ):
        """@private"""
        return max(self.nlay,1)*max(self.nrow,1)*max(self.ncol,1)

    @property
    def ncpl(self, ):
        """@private"""
        return max(self.nrow,1)*max(self.ncol,1)

    @property
    def xoff(self, ):
        """@private"""
        return self.xmin

    @property
    def yoff(self, ):
        """@private"""
        return self.ymin

    @property
    def angrot(self, ):
        """@private"""
        return self.rotation

    @property
    def is_structuredgrid(self, ):
        """@private"""
        return self.nlay > 0 and self.nrow > 0

    @property
    def ibound(self, ):
        """@private"""
        if self._ibound is None:
            if self.is_layered:
                self._ibound = np.array([self.dat[f'ibound{ilay+1}'].values for ilay in range(self.nlay)])
            else:
                self._ibound = self.dat.ibound.values
        return self._ibound

    @property
    def is_layered(self, ):
        """@private"""
        return self.nlay > 0

    @property
    def vert(self):
        """@private"""
        if not is_true(getattr(self, '_vert', None)):
            self._loaddata()
        return self._vert

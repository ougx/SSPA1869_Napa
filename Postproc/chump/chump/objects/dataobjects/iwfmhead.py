import pandas as pd

from ...utils.io import read_iwfmhead
from .modelarray import modelarray


class iwfmhead(modelarray):
    """reads IWFM groundwater ouput file; see `.modelarray` for the parameters
    """

    def set_grid(self):

        model = self._getobj('model')

        self._is_structuredgrid = False
        self.gridtype = model.gridtype


        self.is_layered = True
        self._geom = model.dat.geometry
        self._x = model.vert.x
        self._y = model.vert.y


    def readdata(self):
        self._dat = read_iwfmhead(self.source,)

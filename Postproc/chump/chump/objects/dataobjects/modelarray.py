import numpy as np
import pandas as pd

from ...utils.io import readarray
from ...utils.misc import get_index_names
from ..mobject3d import mobject3d


class modelarray(mobject3d):
    """reads a model array file in text format"""

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)
        # data
        self.source = self.dict.get('source')
        """set the file path of the file
        """

        self.model = self._getobj('model')
        """set the model object

        Example:
        >>> model = {mf='gwf'}
        """

        self.valwidth = self.dict.get('valwidth', None)
        """set the width occupied by for each value. default is None

        Example:
        >>> valwidth = 10
        """

        self.dtype = self.dict.get('dtype', float)
        """set the data type. default is float

        Example:
        >>> dtype = 'int'
        """

        self.skiprows = self.dict.get('skiprows', 0)
        """set the number of line to skip before reading data. default is 0.

        Example:
        >>> skiprows = 10
        """

        self.a_scale     = self.a_scale
        self.a_offset    = self.a_offset
        self.t_scale     = self.t_scale
        self.t_offset    = self.t_offset
        self.limits      = self.limits
        self.abslimits   = self.abslimits
        self.minlayer    = self.minlayer
        self.maxlayer    = self.maxlayer
        self.mintime     = self.mintime
        self.maxtime     = self.maxtime
        self.lfunc       = self.lfunc
        self.tfunc       = self.tfunc
        self.resample    = self.resample
        self.add         = self.add
        self.subtract    = self.subtract
        self.mul         = self.mul
        self.div         = self.div
        self.calculate   = self.calculate
        self.extfunc     = self.extfunc

        # plotting
        self.plottype    = self.plottype
        self.plotarg     = self.plotarg
        self.norm        = self.norm
        self.legend      = self.legend
        self.colorbararg = self.colorbararg
        self.clabelarg   = self.clabelarg
        self.levels      = self.levels
        self.dlevel      = self.dlevel
        self.label       = self.label

        # output
        self.writedata   = self.writedata
        self.writeshp    = self.writeshp
        self.writefunc   = self.writefunc


    def readdata(self):

        array = readarray(**self.dict)
        model = self._getobj('model')
        if model.is_layered:
            array = array.reshape([-1, model.ncpl])
            array = pd.DataFrame(array, index=pd.Index(range(1, array.shape[0]+1), name='layer'))
        else:
            array = pd.DataFrame(array, index=model.dat.index)

        array.columns = range(1, array.shape[1]+1)
        self._dat = array


    def extract_nodes(self, nodes):
        """@private"""
        return self.dat.loc[:, nodes]


    def _finalizedata(self):
        if self._is_structuredgrid is None:
            self.set_grid()

        model = self._getobj('model')
        ibounds = model.dat.loc[:,[c.startswith('ibound') for c in model.dat.columns]]
        dats = []
        for il, c in enumerate(ibounds.columns):
            ib = np.where(ibounds[c]==0, np.nan, 1.0).reshape([1, -1])
            if 'layer' in get_index_names(self.dat):
                if il + 1 in self.dat.index.get_level_values('layer'):
                    dats.append(self.dat.xs(il+1, level='layer', drop_level=False) * ib)
            else:
                dats.append(self.dat * ib)
                break
        self._dat = pd.concat(dats).sort_index()
        super()._finalizedata()

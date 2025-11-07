
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from ...utils.io import agg_hds
from ...utils.misc import iterate
from .mfcbb import mfcbb
import flopy

class mfflux(mfcbb):
    """UNDER DEVELOPMENT"""

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)
        self.dict['identifier'] = ['FLOW RIGHT FACE', 'FLOW FRONT FACE', 'FLOW LOWER FACE']
        self.dict['plottype'] = 'quiver'

        self._qx = None  # horizontally
        self._qy = None  # vertically
        self._qz = None  # vertically

    def readdata(self):

        self.set_grid()
        mf = self._getobj('model')
        assert mf.is_structuredgrid, 'mfflux for unstructured grid is currently not supported'


        with flopy.utils.CellBudgetFile(self.dict['source']) as cbc: # TODO write IO
            self._qx =  cbc.get_data(text='FLOW RIGHT FACE', full3D=True)
            self._qy = -cbc.get_data(text='FLOW FRONT FACE', full3D=True)
            self._qz = -cbc.get_data(text='FLOW LOWER FACE', full3D=True)
            self._times = cbc.get_times() * self.dict.get('t_scale', 1) + self.dict.get('t_offset', 0)

        # set stepping
        ntime, nlay = self._qx.shape[:2]

        xinterval = 1 if mf.ncol < 30 else int(mf.ncol / 30)
        yinterval = xinterval
        xinterval = self.dict.get('xinterval', xinterval)
        yinterval = self.dict.get('yinterval', yinterval)

        size = ntime * nlay

        self._qx = self._qx[:,:,::yinterval,::xinterval].reshape([size, -1])
        self._qy = self._qy[:,:,::yinterval,::xinterval].reshape([size, -1])
        self._qz = self._qz[:,:,::yinterval,::xinterval].reshape([size, -1])

        self._x = self._x[::yinterval,::xinterval]
        self._y = self._y[::yinterval,::xinterval]

        self.plotarg['angles'] = 'xy'

        start_date = self.dict.get('start_date', mf.start_date)
        time_unit  = self.dict.get('time_unit', mf.time_unit)
        self._dat = pd.concat(
            {start_date + pd.Timedelta(t, time_unit): pd.Series(0, index=range(1, nlay+1)) for t in self._times},
            names = ['time', 'layer']
        )



    def plot(self, *args, **kwargs):
        self.quiver = self.ax.quiver(self.x, self.y, self._qx[0], self._qy[0], **self.plotarg)


    def update_plot(self, idx=0, hasData=True):
        if not hasData:
            return
        print(f'    Updating {self.name} {idx} {self.get_label(idx)}')
        self.quiver.set_UVC(self._qx[idx], self._qy[idx])


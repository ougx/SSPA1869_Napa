import pandas as pd
import os
import flopy
import numpy as np

from .mfbin import mfbin


class mfcbb(mfbin):
    """read MODFLOW cell-by-cell budget file; also see `.mfbin` for the parameters
    """

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        # initialize the parameters that can be used
        self.text = self.dict.get('text', None)
        """set text identifier to extract the flux; used before MF6"""

        self.package = self.dict.get('package', None)
        """set the first (flow-from) pckage name to extract the flux; used for MF6"""

        self.package2 = self.dict.get('package2', None)
        """set the second (flow-to) pckage name to extract the flux; used for MF6"""

        # initialize the parameters that can be used
        self.precision = self.dict.get('precision', 'auto')
        """Precision of floating point budget data in the file. Default is 'auto'."""


    def readdata(self):

        mf = self._getobj('model')
        with flopy.utils.CellBudgetFile(self.source, precision=self.precision) as cbc:
            dat = cbc.get_data(text=self.text, paknam=self.package, paknam2=self.package2, full3D=True)
            times = np.array(cbc.get_times()) * self.t_scale + self.t_offset
            if self.start_date:
                times = [self.start_date + pd.Timedelta(t, self.time_unit) for t in times]
            if dat[-1].ndim < 3:
                nlay = 1
            else:
                nlay = dat[-1].shape[0]

            self._dat = pd.concat(
                {t:pd.DataFrame(dat[i].reshape([nlay, -1]), index=pd.Index(range(1, nlay+1), name='layer')) for i, t in enumerate(times)},names=['time'])


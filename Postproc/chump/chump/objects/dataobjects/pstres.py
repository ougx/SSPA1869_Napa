from io import StringIO
import pandas as pd
import numpy as np

from ...utils.datfunc import exttable
from .tseries import tseries

class pstres(tseries):
    """
    reads a PEST residual file (.res or .rei) See `.tseries` for the parameters
    """

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

    def readdata(self, ):

        with open(self.source) as f:
            res = [l.strip() for l in f]
        if self.source.lower().strip().endswith('.rei'):
            res = res[4:]
        self._dat = pd.read_csv(StringIO('\n'.join(res)), sep=' ', skipinitialspace=True).rename(columns={
                            'Name':'obsname', 'Group':'pestObsGroup', 'Weight':'pestObsWeight', 'Measured':'Observed', 'Modelled':self.name
                        })


        # get the external naming table
        if self.exttable:
            exttable(self, 'exttable', on='obsname')

        self._labels = ['Observed', self.name]

        if self.timecol in self._dat.columns:
            self._dat[self.timecol] = self._parse_time_col(self._dat[self.timecol])
            self._process_duplicatedtime()

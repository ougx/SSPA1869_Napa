from io import StringIO
import pandas as pd
import numpy as np
from ...utils.datfunc import exttable
from .tseries import tseries

class hobout(tseries):
    """
    reads output of the MODFLOW HOB package. See `.tseries` for the parameters
    """

    def readdata(self, ):

        self._labels = ['Simulated', 'Observed', ]

        with open(self.source) as f:
            res = [l.strip() for l in f]

        df = pd.read_csv(self.source, sep=' ', skipinitialspace=True, skiprows=1, header=None).iloc[:,:3]
        df.columns = self._labels + ['obsname']

        df['obsname'] = df['obsname'].astype(str).str.lower()

        # get the external naming table
        self._dat = df
        if self.exttable:
            exttable(self, 'exttable', on='obsname')

        # calculate the mean if there is overlapping time for the same well
        self._dat[self.timecol] = self._parse_time_col(self._dat[self.timecol])

        # calculate the mean if there are identical times
        self._process_duplicatedtime()

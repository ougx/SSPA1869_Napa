from io import StringIO

import pandas as pd
import numpy as np
from ...utils.io import readsfr2df
from ...utils.misc import iterate

from ...utils.datfunc import exttable
from .tseries import tseries


class sfrout(tseries):
    """define a SFR output object that reads SFR output file. See `.tseries` for other parameters.
    """

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        self.source    = self.source

        if 'idcol' not in self.dict:
            self.dict['idcol'] = 'segment'
        """set the output unit. Default is 'segment'."""


        self.valcol: str = self.dict.get('valcol', 'qaq')
        """a string defining the value columns in the data source. options include:

        - 'qaq': stream leakage
        - 'qin': stream inflow
        - 'qout': stream outflow
        - 'overland': stream overland flow
        - 'precip': stream precipitation
        - 'et': stream ET
        - 'stage': stream stage
        - 'depth': stream water depth
        - 'width': stream width
        - 'condutance': stream conductance
        - 'gradient': gradient between stream stage and groundwater head

        Default is 'qaq'.

        Example:
        >>> valcol = 'overland'
        """

        self.aggfunc = self.dict.get('aggfunc', 'sum')
        """apply aggregation function based on the site/well id. Default is 'sum'.

        Example:
        >>> aggfunc = 'mean'
        """

    def readdata(self, ):
        # read as dataframe
        df = readsfr2df(self.source)

        # get the time from modflow
        mf = self._getobj('model')
        totim = mf.totim
        self._dat = pd.merge(df, totim[['period', 'step', 'time']], on=['period', 'step'])

        # get the external naming table
        if self.exttable:
            exttable(self, 'exttable', on=['segment', 'reach'])

        self._labels = [c for c in iterate(self.valcol)]

        # calculate the mean if there is overlapping time for the same well
        self._dat[self.timecol] = self._parse_time_col(self._dat[self.timecol])

        self._process_duplicatedtime()

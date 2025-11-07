from io import StringIO

import pandas as pd
import numpy as np
from ...utils.io import read_swat_output
from ...utils.misc import iterate, get_index_names
from ...utils.datfunc import exttable
from .tseries import tseries


class swatoutput(tseries):
    """read SWAT model's output.hru, output.rch, output.sub etc. See `.tseries` for the parameters.
    """

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        self.source    = self.source
        self.model     = self._getobj('model')
        """set the SWAT model object

        Example:
        >>> model = {swat='PlatteRiver'}
        """


    def readdata(self, ):
        # get the swat cio
        swat = self._getobj('model')
        # read as dataframe
        df = read_swat_output(self.source, int(swat.cio["IYR"]), int(swat.cio["NYSKIP"]), int(swat.cio["IPRINT"]))
        self.timecol = 'time'

        # get the external naming table
        self._dat = df
        if self.exttable:
            exttable(self, 'exttable', on=['segment', 'reach'])

        self._labels = df.columns
        if self.valcol:
            self._labels = [c for c in iterate(self.valcol)]

        # calculate the mean if there is overlapping time for the same well
        self._process_duplicatedtime()

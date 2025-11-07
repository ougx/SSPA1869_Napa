import pandas as pd
import os
import flopy
import re
from .tseries import tseries

class mflst(tseries):
    """read water budget in a MODFLOW list file. See .tseries for the parameters.
    """


    def readdata(self):
        lst = self.source

        assert os.path.exists(lst), f'{lst} does not exists.'
        with open(lst) as f:
            mf6 = f.readline().strip() == 'MODFLOW 6'
        lst = flopy.utils.Mf6ListBudget(lst, ) if mf6 else flopy.utils.MfListBudget(lst,)
        incremental, cumulative = lst.get_dataframes(
            start_datetime=None, diff=True,
        )

        if self.dict.get('volume', False):
            df = cumulative.diff()
            df.iloc[0] = cumulative.iloc[0]
        else:
            df = incremental

        self.timecol = df.index.name or 'index'
        self._dat = df.reset_index()
        self._dat[self.timecol] = self._parse_time_col(self._dat[self.timecol])

        df = self._dat.set_index(self.timecol)
        if 'sto-ss' in df.columns and 'sto-sy' in df.columns:
            df['sto'] = df['sto-ss'] + df['sto-sy']
            # df.drop(['sto-ss', 'sto-sy'], axis=1)

        df = df[[c for c in df.columns if not ('total' in c.lower() or 'in-out' in c.lower() or 'percent' in c.lower())]]
        df.columns = [re.sub(" +", " ", c.replace("_", " ")).title() for c in df.columns]

        df = df.stack().swaplevel().to_frame()
        df.columns = ['flux']
        df.index.names = ['package', 'time']
        self._dat = df

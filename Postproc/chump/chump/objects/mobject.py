from copy import copy

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype, is_string_dtype
from ..utils.misc import iterate, is_true

class mobject():
    """Data objects extract and store spatial and temporal information of diverse formats,
    and convert them into a uniform data abstraction represented as Pandas dataframes
    with highly efficient indexing/slicing capability. Three top data classes,
    `mobject`, `chump.objects.mobject2d`, and `chump.objects.mobject3d`, are available to
    represent data with various degrees of dimensionality.

    The `mobject` class serves as the root superclass for all other derived subclasses.
    Below are some properties that can be set for all the objects in `chump`.
    """

    def __init__(self, name, mplot, *args, **kwargs):
        """
        @private
        """
        self.name = name
        """
        @private
        """
        self.type = type(self).__name__
        """
        @private
        """
        self.parent = mplot
        """
        @private
        """
        self.verbose = mplot.verbose
        """
        @private
        """
        self.exclude = None
        """
        @private
        """
        self._dat = None
        self._nplot = None

        self.time_unit: str = self.dict.get('time_unit',
                                       self.parent.dict.get('time_unit', 'D') if self.parent is not None else 'D')
        '''
        time unit for the object. default is "days". If not set, it will use the global setting.

        '''

        self.start_date: str | datetime = pd.Timestamp(self.dict['start_date']) if 'start_date' in self.dict else mplot.start_date
        '''
        starting date for time related data. The time will be parsed into timestamps using the starting date and time unit.
        If not set, it will use the global setting.
        '''

        self.epsg: int = self.dict.get('epsg', mplot.epsg)
        '''
        set the coordinate reference system for the object using EPSG. e.g. `epsg = 4326`. If not set, it will use the global setting.
        '''

        self.crs: str = self.dict.get('crs', mplot.crs)
        '''
        set the coordinate reference system using strings. See details in https://geopandas.org/en/stable/docs/user_guide/projections.html.
        '''

        if self.crs is None and self.epsg is not None:
            self.crs = f"epsg:{self.epsg}"

        self.plotarg = copy(self.dict.get('plotarg', {}))
        '''
        plotting argument for the object
        '''

        self.legend = self.dict.get('legend', False)
        '''
        whether to show the legend
        '''

        self.timecol: str = self.dict.get('timecol', 'time')
        """set the time column. Default is 'time'.

        Example:
        >>> timecol = 'date'
        """

        self.t_scale: float = self.dict.get('t_scale', 1.0)
        """set a scaling factor, default is 1.0. final time values = original time values * `t_scale` + `t_offset`

        Example:
        >>> t_scale = 86400 # convert days to seconds
        """

        self.t_offset: float = self.dict.get('t_offset', 0.0)
        """set a time offset, default is 0.0. final time values = original time values * `t_scale` + `t_offset`

        Example:
        >>> t_offset = -365 # move time ahead by one year
        """

        self._parse_plotarg()


    def _getobj(self, key: str or dict, fromeq=False):
        """
        get an object
        """
        if isinstance(key, dict):
            for k, v in key.items():
                return getattr(self.parent, k)[v]
        else:
            if key == '':
                return None
            elif fromeq:
                k, v = key.split('.', 1)
                return self._getobj({k:v})
            else:
                assert key in self.dict, f'{key} not in {self}'
                if self.dict[key] == '':
                    return None
                assert isinstance(self.dict[key], dict), f'{key}:{self.dict[key]} is not a dict in {self}'
                return self._getobj(self.dict[key])
        raise ValueError(f'No {k} object named "{v}" is defined.')

    def plot(self):
        """@private"""
        pass

    @property
    def fullname(self):
        """
        @private
        """
        return f"{self.type}.{self.name}"

    @property
    def dict(self):
        """
        @private
        """
        if getattr(self, "_dict", None) is None:
            olddict = self.parent.dict[self.type].get(self.name, {})
            self._dict = {}
            for k, v in olddict.copy().items():
                if k != 'parent':
                    self._dict[k] = v
                else:
                    parent = self._getobj({self.type:olddict['parent']})
                    if parent is not None:
                        for kk, vv in parent.dict.items():
                            if kk not in self._dict:
                                self._dict[kk] = vv
        return self._dict

    @property
    def nplot(self):
        """
        @private
        """
        if self._nplot is None:
            self._loaddata()
        return self._nplot

    @property
    def dat(self, ):
        """
        @private
        """
        if self._dat is None:
            self._loaddata()
        return self._dat

    def _parse_plotarg(self, ):
        if 'legend' in self.plotarg:
            self.legend = self.plotarg.pop('legend')

        if 'cmap' in self.plotarg:
            cmaps = self.plotarg['cmap'].split(',')
            if len(cmaps) > 1:
                self.plotarg['cmap'] = plt.get_cmap(cmaps[0], int(cmaps[1]) if cmaps[1] else None)


    def _parse_time_col(self, times, to_numeric=False):

        if getattr(self, 'verbose') > 5:
            print(self.fullname, 'timecol   ', self.timecol)
            print(self.fullname, 'start_date', self.start_date)
            print(self.fullname, 'time_unit ', self.time_unit)
            print(self.fullname, 't_scale   ', self.t_scale)
            print(self.fullname, 't_offset  ', self.t_offset)

        if not is_numeric_dtype(times):
            return pd.to_datetime(times)

        if is_true(self.start_date):
            self.start_date = pd.Timestamp(self.start_date)

            if to_numeric:
                if is_datetime64_dtype(times):
                    return [(t - self.start_date)/pd.Timedelta(1, self.time_unit) for t in iterate(times)]
            else:
                if is_numeric_dtype(times):
                    times = times * self.t_scale + self.t_offset
                    times = [self.start_date + pd.Timedelta(t, self.time_unit) for t in iterate(times)]
                    return pd.to_datetime(times)
        return times


    def _loaddata(self):
        self._nplot = 0
        self._dat = []
        return

    def __str__(self):
        mydict = pd.Series(self.dict).to_string().replace("\n", "\n  ") + "\n"
        return f'{self.fullname}\n  {mydict}'

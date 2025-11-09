from collections import OrderedDict

import numpy as np
import pandas as pd

from ...utils.misc import iterate, is_true, get_index_names, get_prettylabel
from .borelog import borelog
from .figure import figure
from .mapplot import mapplot


class tsplot(figure):
    '''
    *plotdata*: dictionary obsts, simts, pstres, mflst
    [*plotdata**]: addtional dictionary obsts, simts, pstres, mflst
    *features*: dictionary one or more table, shp or csv to define the property of the plotting features, e.g. `features = {table=['wellwest', 'welleast']}`.
    [*featnames*]: a list of feature names for the time series plot, for example some well names of interest
    [*legendarg*]: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
    '''

    def _loaddata(self):
        super()._loaddata()

        self.ax1 = None
        self.axmap = None
        self.droplevel = ['time']

        end_date = None#pd.Timestamp('1700-01-01')
        start_date = None#pd.Timestamp('2260-12-31')
        # locations = []
        for ts in self.plotdata:
            times = ts.dat.index.get_level_values(ts.timecol)
            if end_date is None:
                end_date = times.max()
                start_date = times.min()
            else:
                end_date = max(end_date, times.max())
                start_date = min(start_date, times.min())

        self.end_date = end_date
        self.start_date = start_date


    def add_elements(self):

        self.tsmap = None
        if 'mapplot' in self.dict:
            self.tsmap = mapplot(self.dict['mapplot'], self.parent)
            for i in range(99):
                if 'plotdata'+str(i) not in self.tsmap.dict:
                    plotdatakey = 'plotdata'+str(i)
                    self.tsmap.dict[plotdatakey] = []
            self.tsmap.initialize_plot(self.fig, )
            self.tsmap.dict.pop(plotdatakey)
            for ts in self.plotdata:
                # add location to plotdata so it will show up in the map legend
                if 'location' in ts.dict:
                    ts.axmap = self.tsmap
            self.text.extend(self.tsmap.text)

        for e in self.plotdata:
            if e.ax == self.ax or e.ax == self.ax1:
                legend = e.plot()
                if legend is not None:
                    self.legends.update(**legend)
            if e.nplot > 0:
                self.Elements.append(e)

        self.ax.set_xlabel(self.dict.get('xlabel', 'Time'))
        if is_true(self.tsmap):
            self.tsmap.plot_legend()

        if 'xlim' not in self.dict:
            self.ax.set(xlim=(
                self.start_date-0.02*(self.end_date-self.start_date),
                self.end_date+0.02*(self.end_date-self.start_date))
            )

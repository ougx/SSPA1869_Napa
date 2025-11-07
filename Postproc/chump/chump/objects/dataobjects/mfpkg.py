import os

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Patch
import flopy

from .mfbin import mfbin

# TODO refine
class mfpkg(mfbin):
    """UNDER DEVELOPMENT

    read a MODFLOW package
    """
    _defaultcolors = {
        'riv':'green', 'sfr':'green', 'str':'grren',
        'chd':'blue',
        'wel':'red', 'mnw1':'red', 'mnw2':'red', 'mnw':'red',
        'drn':'yellow',
        'ghb':'c',
        'lak':'lightpurple'
    }

    def readdata(self):

        self.color = self.plotarg.get('color', self._defaultcolors[self.dict['package'].lower()])
        self.plotarg['color'] = self.color

        mf = self._getobj('model')

        found_package = False
        with open(mf.namefile_fullpath) as f:
            mfname_lines = list(filter(lambda x: not (x.startswith('#') or x.strip() == ''), f.readlines()))
            for l in mfname_lines:
                if l.lower().lstrip().startswith(self.dict['package'].lower()):
                    found_package = True
                    break

        assert found_package, f"Package {self.dict['package']} is not in Model {mf.name}."

        p,n,f = l.split()[:3]

        pkg = getattr(flopy.modflow, f'Modflow{self.dict["package"].capitalize()}').load(
            os.path.join(mf.dict.get('model_ws', '.'), f), mf.mf, #check=False
        )

        periods = self.dict.get('periods', range(1, mf.mf.nper+1))
        if isinstance(periods, int):
            periods = [periods]

        self._x = mf.xvertices
        self._y = mf.yvertices
        self._z = mf.zvertices
        self._dat = {}

        for i in periods:
            p = self.pkg.stress_period_data[i-1]
            r = np.full([mf.nlay, mf.nrow, mf.ncol], np.nan)
            r[p['k'],p['i'],p['j']] = 1
            self._dat[mf.totim[i-1]] = pd.DataFrame(r.reshape([mf.nlay, -1]), index=range(1, mf.nlay+1))
            self._dat[mf.totim[i-1]].index.name = 'layer'

        self._dat = pd.concat(self._dat, names=['time'])
        self._dat.dropna(how='all', inplace=True)
        self._nplot = self._dat.shape[0]

    def plot(self, ax=None, idx=0):
        super().plot(ax=ax, idx=idx)
        return {self.name: Patch(color=self.color)} if self.dict.get('legend', False) else {}

    def vplot(self, ax=None, rows=None, cols=None, lengths=None, idx=0):
        super().vplot(ax=ax, rows=rows, cols=cols, lengths=lengths, idx=idx)
        return {self.name: Patch(color=self.color)} if self.dict.get('legend', False) else {}

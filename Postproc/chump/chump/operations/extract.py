import warnings

import numpy as np
import pandas as pd
from ..objects.mobject import mobject
from ..utils.misc import (is_true, iterate, iterate_dicts, merge_id_df, get_index_names, interpolate_missing_data,
    impute_time, warn_deprecation)
from ..utils.constants import navalue
from ..utils.math import merr, mae, rmse, maxae, std
from ..utils.io import read_ts_nodes



def headbyweight(heads, weights, q=0):
    '''
    q is an arbitrary pumping assumed to be used for a monitoring well when a water sample is collected
    '''

    mask = pd.isna(heads)
    weights = np.where(mask, 0.0, 1.0) * weights # when the sum weight is not 1.0

    return ((heads.where(~mask, 0.0) * weights).sum(axis=1) - q) / weights.sum(axis=1)


class extract(mobject):
    """ `extract` is a command used to retrive model results at specified wells/sites matching point measurements for the `extract` objects defined in the configuration file.

    Example:
    >>> chump.exe extract example.toml
    """

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)
        self._dat = None
        self._stat = None

        if 'wells' in self.dict:
            warn_deprecation(self, '`wells` is deprecated since v20231214. Please use `sites` instead.')
            self.dict['sites'] = self.dict.pop('wells')

        self.sites = self.dict['sites']
        """ `sites` set the object representing wells/sites.

        Example:
        >>> sites = {csv = 'obswells'}
        >>> sites = [{csv = ['obswells0', 'obswells1']}, {shp = 'obswells2'}]
        """

        #self.model = self.dict.get('model', None)
        #""" `model` set the model object for this extraction.
        #This is only needed when the grid or parameter
        #is needed when computing the representative values of the simulation at the sites.
        #For example, the thickness and hydraulic conductivity are needed
        #when computing the transmissivity weighted groundwater head for a multi-layer well.
#
        #Example:
        #>>> model = {mf = 'gwf'}
        #>>> model = {iwfm = 'C2VSimCG'}
        #"""

        self.sim = self.dict['sim']
        """ `sim` set the objects representing model results.
        It can be the model full-grid output (e.g. `..objects.dataobjects.mfbin`, `..objects.dataobjects.iwfmhead`) or
        a MF6 `OBS` output file (`..objects.dataobjects.tseries`).

        Example:
        >>> sim = {mfbin = 'TCEconc'}
        >>> sim = {iwfmhead = 'svsimhead'}
        """

        self.sim2 = self.dict.get('sim2', None)
        """ `sim2` set the object representing simulated heads
        that is needed when computing simulated concentration in a multilayer well using the `mass` or `massts` method.

        Example:
        >>> sim  = {mfbin = 'TCEconc'}
        >>> sim2 = {mfbin = 'heads'}
        """

        self.obs = self.dict.get('obs', None)
        """ `obs` set the objects representing cooresponding observation data. Default is None.

        If an observation timestmap is not included in the model simulation,
        linear interpolation is perfomed to get the simulated values for the missing timestamps.

        Example:
        >>> obs = {tseries = 'obsgwlevel'}
        >>> obs = {tseries = ['obsgwlevel0', 'obsgwlevel1']}
        """

        self.alllayer = self.dict.get('alllayer', False)
        """ if `alllayer` is ture, extract model results for all the layers at site. Default is false.
        """

        self.obtimeonly = self.dict.get('obtimeonly', False)
        """ if `obtimeonly` is ture, write data at the observation times only to another file.
        When `obtimeonly` is ture, `obs` must be set.
        """

        self.log = self.dict.get('log', False)
        """ if `log` is ture, add log10(data) columns to the data table. Default is false.
        """

        self.diff = self.dict.get('diff', False)
        """ if `diff` is ture, add difference (current row - privous row) columns to the data table. Default is false.
        """

        self.extractq = self.dict.get('extractq', None)
        """ for `avgMethod == "massts"`, pumping rate time seris can be specified using `extractq`.
        The column name must be 'extractq' in the time series.

        Example:
        >>> extractq  = {tseries = 'PumpingRate'}
        """

    @property
    def stat(self):
        """@private
        """
        if self._stat is None:
            self._loaddata()
        return self._stat


    def _loaddata(self, ):

        self.wellnames = []
        sites = []
        for kkv in iterate_dicts(self.sites):
            for k, kv in kkv.items():
                for v in iterate(kv):
                    tab = self._getobj({k: v})
                    sites.append(tab.dat)
                    self.wellnames.append(tab.dat.index)
        self.wells = pd.concat(sites)
        self.wellnames = np.unique(np.concatenate(self.wellnames))
        if is_true(self.obs):
            obs = []
            for kkv in iterate_dicts(self.obs):
                for k, kv in kkv.items():
                    for v in iterate(kv):
                        # print(k, kv, v)
                        o = self._getobj({k: v})
                        od = o.dat.copy()
                        od.columns = ['Observed']
                        obs.append(od)
            self.obs = pd.concat(obs, )
            self.wellnames = np.intersect1d(self.obs.index.get_level_values(0).unique(), self.wellnames)
            self.obs   = self.obs.loc[self.wellnames]
            if self.obs.index.names is not None:
                indexnames = self.obs.index.names
                self.obs = self.obs.rename_axis([self.wells.index.name,] + indexnames[1:])

        self.wells = self.wells.loc[self.wellnames].copy()
        well_columns = {c.lower() if isinstance(c, str) else c:c for c in self.wells.columns}
        sim = self._getobj('sim')
        model = None
        if getattr(sim, 'model', None):
            model = sim._getobj('model')

        if 'layer' not in well_columns.values():
            self.wells['layer'] = 1
            well_columns['layer'] = 'layer'

        if 'avgmethod' not in well_columns.values():
            if is_true(model):
                self.wells['avgmethod'] = 'layer'
            else:
                self.wells['avgmethod'] = 'none'
            well_columns['avgmethod'] = 'avgmethod'

        # if sim2 is not defined, mass balace method cannot be used
        avgmethod = well_columns['avgmethod']
        if self.sim2 is None:
            self.wells.loc[self.wells[avgmethod].astype(str).str.startswith('mass'), avgmethod] = 'trans'

        if 'ninterp' not in well_columns:
            self.wells['ninterp'] = 1
            self.wells['node1'] = 1.0
            self.wells['weight1'] = 1.0
            well_columns['ninterp'] = 'ninterp'
            well_columns['node1']   = 'node1'
            well_columns['weight1'] = 'weight1'

        if self.alllayer:
            self.wells[well_columns['avgmethod']] = 'alllayer'

        # get the unique nodes
        maxiterp = int(self.wells.loc[:, well_columns['ninterp']].max())
        nodes = []
        for i in range(maxiterp):
            nodes.append(self.wells.loc[:, well_columns['node'+str(i+1)]].dropna())
        nodes = np.unique(np.concatenate(nodes))
        nodevalues= sim.extract_nodes(nodes)
        nodevalues2 = None
        if any(self.wells[well_columns['avgmethod']].str.startswith('mass')):
            nodevalues2= self._getobj('sim2').extract_nodes(nodes)

            time1 = nodevalues.index.get_level_values('time')
            layer1 = nodevalues.index.get_level_values('layer')

            time2 = nodevalues2.index.get_level_values('time')
            layer2 = nodevalues2.index.get_level_values('layer')

            times  = np.unique(np.concatenate([time1, time2]))

            if any(self.wells[well_columns['avgmethod']]=='massts'):
                extractq = impute_time(self._getobj('extractq').dat, times, interp=False)

            nodevalues  = impute_time(nodevalues,  times, )
            nodevalues2 = impute_time(nodevalues2, times, )


        results = {}
        for wellname, wellrow in self.wells.iterrows():
            print(f'      Extracting {wellname}')

            avgmethod = wellrow[well_columns['avgmethod']].lower()
            ninterp   = int(wellrow[well_columns['ninterp']])
            weights   = wellrow[[well_columns['weight'+str(i+1)] for i in range(ninterp)]].values[np.newaxis]
            nodes     = wellrow[[well_columns['node'  +str(i+1)] for i in range(ninterp)]].values

            if avgmethod == 'none':
                hh = (nodevalues.loc[:, nodes] * weights).sum(axis=1)

            elif avgmethod == 'layer':
                # get the layer
                layer = wellrow[well_columns['layer']]
                hh = headbyweight(nodevalues.xs(layer, level='layer')[nodes], weights)

            else:
                hh = headbyweight(nodevalues.loc[:, nodes], weights).unstack('layer')

                if avgmethod == 'alllayer':
                    # get the layer
                    hh.columns = [self.name + f'_Layer{int(c):02d}'  for c in hh.columns]
                    results[wellname] = hh

                elif avgmethod in ['thickness', 'trans', 'mass', 'massts']:
                    assert 'scntop' in well_columns and  'scnbot' in well_columns, f"Well {wellname} is defined by {avgmethod} without scntop or/and scntop"
                    assert not (pd.isna( wellrow[well_columns['scntop']]) or pd.isna( wellrow[well_columns['scnbot']])), \
                                f"Well {wellname} is defined by {avgmethod} with NAN scntop or/and scntop"


                    hk = 1.0
                    if model.gridtype == 'fe':
                        botm = 0
                        for n, w in zip(nodes, weights):
                            botm += model.vert.loc[n,'botm0':'botm'+str(model.nlay)] * w
                        if avgmethod in ['trans', 'mass']:
                            hk = 0.0
                            for n, w in zip(nodes, weights):
                                hk += model.vert.loc[n:,'hk1':'hk'+str(model.nlay)] * w

                    else:
                        node =  int(wellrow[well_columns['node']])
                        botm = model.dat.loc[node, [f'botm{ilay}' for ilay in range(model.nlay+1)]]
                        if avgmethod != 'thickness':
                            hk = model.dat.loc[node, [f'hk{ilay+1}' for ilay in range(model.nlay)]]


                    # find layers
                    scntop = wellrow[well_columns['scntop']]
                    scnbot = wellrow[well_columns['scnbot']]

                    layTop = np.searchsorted(-botm[1:-1], -scntop) + 1
                    layBot = np.searchsorted(-botm[1:-1], -scnbot) + 1
                    botm   = np.array(botm)
                    hk     = np.array(hk)

                    botm[layTop-1] = scntop
                    botm[layBot]   = scnbot
                    thick = -np.diff(botm)

                    if avgmethod != 'thickness':
                        thick = thick * hk

                    hh = hh.loc[:,layTop:layBot]
                    vweights = thick[layTop-1:layBot][np.newaxis]

                    if not avgmethod.startswith('mass'):
                        hh = headbyweight(hh, vweights, )
                    else:
                        gwe = headbyweight(nodevalues2.loc[:,nodes], weights).unstack('layer').loc[:,layTop:layBot]
                        if avgmethod == 'massts':
                            q = extractq.loc[wellname]['extractq'].loc[gwe.index].values
                        else:
                            q = wellrow[well_columns['extractq']]
                        gwewell = headbyweight(gwe, vweights, q).to_frame('wellhead')
                        qq = (gwe - gwewell.values) * vweights
                        qq.where(qq>0, 0, inplace=True)
                        hh = (hh * qq).sum(axis=1)  / np.where(q>0, qq.sum(axis=1), np.nan)

            results[wellname] = hh.to_frame(name=self.name)

        dat = pd.concat(results, names=[self.wells.index.name, 'time'])

        if self.obs is not None:
            dat = pd.concat([self.obs, dat, ], axis=1).sort_index()

            # fill simulated to observed times

            icol_sim = list(dat.columns).index('Observed') + 1
            valcols = list(dat.columns)[icol_sim:]
            dat = interpolate_missing_data(dat, valcols)
        else:
            icol_sim = 0

        Models = list(dat.columns[icol_sim:])

        self._stat = None
        if is_true(self.obs):

            dat1 = dat.copy()

            err = dat.iloc[:,icol_sim:] - dat.iloc[:,icol_sim-1:icol_sim].values

            if self.parent.err_as_oms:
                err = -err

            if self.log:
                for c in dat1.columns:
                    dat1[c+'_log'] = np.log(dat1[c])
                    err[c+'_log'] = dat1[c+'_log'] - dat1['Observed_log']
                    if is_true(self.parent.err_as_oms):
                        err[c+'_log'] = -err[c+'_log']
                Models = Models + [M+'_log' for M in Models]

            if self.diff:
                for c in dat1.columns:
                    dat1[c+'_diff'] = dat1[c].groupby(level=0).diff()
                for c in err.columns:
                    err[c+'_diff'] = err[c].groupby(level=0).diff()
                Models = Models + [M+'_diff' for M in Models]

            # abs mean error
            stat = err.groupby(level=0).agg([merr, mae, maxae, std, rmse, ]).T.unstack(0).T
            stat.index.names = [get_index_names(self.wells)[0], 'Model']
            self._stat = merge_id_df(stat, self.wells, how='left')
        self._dat = dat

    def run(self):
        """@private
        """
        index_names = get_index_names(self.dat)
        if is_true(self._stat):
            self.stat.to_csv(f'{self.name}-stat.csv', )

        self.dat.to_csv(f'{self.name}.csv')

        if self.obtimeonly and is_true(self.obs):
            dat = dat.dropna(subset=['Observed'])
            dat.set_index(index_names).to_csv(f'{self.name}_obtimeonly.csv')

        print(f'{self.name} extracted!')

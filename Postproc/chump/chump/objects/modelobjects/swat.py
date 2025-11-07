import os
from io import StringIO

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from ...utils.constants import parameters, lineHruSub
from ...utils.misc import is_true
from .model import model


class swat(model):
    """define a SWAT model object"""

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        self.model_ws = self.dict.get('model_ws', '.')
        """set the model working directory. Default is '.'
        """

        # read the subbasin shapefile
        self.shpsub = self.dict.get('shpsub', None)
        """set a subbasin shapefile object if output unit is subbasin

        Example:
        >>> [shp.subbasin]
        >>> source = 'GIS/subbasin.shp'

        >>> [swat.watershedmodel]
        >>> shpsub = {shp='subbasin'}
        """

        # read the HRU shapefile
        self.shphru = self.dict.get('shphru', None)
        """set a HRU shapefile object if output unit is HRU"""


    def read_cio(self,):
        '''@private
        read SWAT file.cio

        Returns
        -------
        None.

        '''
        self._cio = dict()
        lines_to_skip = tuple(range(7)) + (11, 33) + tuple(range(33, 41)) + (45, 47, 53, 57) + tuple(range(62, 73)) + (77, )
        with open(os.path.join(self.TxtInOut, 'file.cio')) as f:
            for i, l in enumerate(f):
                if i in lines_to_skip:
                    continue
                else:
                    val, key = l.split('|')
                    key = key[:key.index(':')].strip()
                    self._cio[key] = val.strip()


        # self.output_start_date = pd.DateOffset(pd.Timestamp('{}-01-01'.format(int(self.cio["IYR"])+int(self.cio["NYSKIP"]))), day=int(self.cio["IDAF"])-1)
        # self.output_end_date   = pd.DateOffset(pd.Timestamp('{}-01-01'.format(int(self.cio["IYR"])+int(self.cio["NBYR"])-1)), day=int(self.cio["IDAL"])-1)

        self.output_start_date = pd.Timestamp('{}-01-01'.format(int(self._cio["IYR"])+int(self._cio["NYSKIP"]))) + \
                                 pd.Timedelta(0 if self._cio["NYSKIP"]>'0' else int(self._cio["IDAF"]) - 1, 'D')
        self.output_end_date   = pd.Timestamp('{}-01-01'.format(int(self._cio["IYR"])+int(self._cio["NBYR"])-1)) + \
                                 pd.Timedelta(int(self._cio["IDAL"]) -1, 'D')

    @property
    def cio(self):
        """@private"""
        if not is_true(getattr(self, '_cio', None)):
            self._loaddata()
        return self._cio

    def read_input_sub(self):
        """@private
        read the parameters from the SWAT model input files

        """
        TxtInOut = self.TxtInOut

        with open(os.path.join(TxtInOut, 'fig.fig')) as fig:
            figlines = fig.readlines()

        isub = 0
        sub = []
        subhru = []
        submgt = []
        subsol = []

        for il, lfig in enumerate(figlines):
            if not lfig.startswith('subbasin'):
                continue
            i = il + 1
            isub += 1
            print('    Reading subbasin', os.path.join(TxtInOut, figlines[i].strip()))
            with open(os.path.join(TxtInOut, figlines[i].strip()), encoding="latin-1") as fsub:
                sublines = fsub.readlines()

            sub_area = float(sublines[1][:20])
            sub_ = dict(subbasin=isub, area=sub_area)

            for kkv, vkv in parameters['sub'].items():
                sub_[kkv] = float(sublines[vkv[0]-1][vkv[1]-1 : vkv[2]])

            with open(os.path.join(TxtInOut, figlines[i].strip().rstrip('.sub')+'.rte'), encoding="latin-1") as fsub:
                rtelines = fsub.readlines()

            for kkv, vkv in parameters['rte'].items():
                sub_[kkv] = float(rtelines[vkv[0]-1][vkv[1]-1 : vkv[2]])

            sub.append(sub_)
            for ihru, hrufline in enumerate(sublines[lineHruSub-1:]):
                hru_val = dict(subbasin=isub, hru=ihru+1, subarea=sub_area)

                with open(os.path.join(TxtInOut, hrufline[:13])) as fhru:
                    hrulines = fhru.readlines()
                    l0 = hrulines[0]
                    hru_val['landuse'] = l0[l0.index('Luse:')+5:l0.index('Luse:')+10].strip()
                    hru_val['soil']    = l0[l0.index('Soil:')+5:l0.index('Soil:')+12].strip()
                    hru_val['slope']   = l0[l0.index('Slope:')+6:l0.index('/')-2].strip()

                for ifile in range(0, len(hrufline), 13):
                    # print(l[ifile:ifile+13])
                    for pk, kv in parameters.items():
                        if hrufline[ifile:ifile+13].rstrip().lower().endswith(pk):
                            with open(os.path.join(TxtInOut, hrufline[ifile:ifile+13].strip())) as fhru:
                                hrulines = fhru.readlines()
                            if pk == 'sol':
                                hru_val['HYDGRP']  = hrulines[2][24:99].strip()
                                hru_val['TEXTURE'] = hrulines[6][27:99].strip()
                                ilayline = parameters['sollayer']['SOL_Z01'][0] - 1
                                nlayer = int(len(hrulines[ilayline][27:].rstrip('\n')) / 12)
                                hru_val['NLAYER'] = nlayer
                                for ilay in range(nlayer):
                                    sollay = dict(subbasin=isub, hru=ihru+1, ilayer=ilay+1)
                                    for kkv, vkv in parameters['sollayer'].items():
                                        if int(kkv[-2:]) -1 == ilay:
                                            v = hrulines[vkv[0]-1][vkv[1]-1 : vkv[2]].strip()
                                            sollay[kkv[:-2]] = float(v) if v else np.nan
                                    # print(ilay, sollay)
                                    subsol.append(sollay)


                            elif pk == 'mgt':
                                # read operations
                                iop = -9999
                                for i, l in enumerate(hrulines):
                                    iop += 1
                                    if l.strip().lower().startswith('operation schedule'):
                                        iop = 0
                                        continue
                                    if iop > 0:
                                        # (1x,i2,1x,i2,1x,f8.3,1x,i2,1x,i4,1x,i3,1x,i2,1x,f12.5,1x,f6.2,1x,f11.5,1x,f4.2,1x,f6.2,1x,f5.2,i12)
                                        #    mon,  day,   husc,   op,mgt1i,mgt2i,mgt3i,    mgt4,   mgt5,    mgt6,   mgt7,   mgt8,   mgt9,mgt10i
                                        mgtprac = dict(
                                            subbasin=isub, hru=ihru+1, practice=iop,
                                            mon    = int  (l[1 : 3]) if l[1 : 3].strip() != '' else None,
                                            day    = int  (l[4 : 6]) if l[4 : 6].strip() != '' else None,
                                            husc   = float(l[7 :15]) if l[7 :15].strip() != '' else None,
                                            op     = int  (l[16:18]) if l[16:18].strip() != '' else None,
                                            mgt1i  = int  (l[19:23]) if l[19:23].strip() != '' else None,
                                            mgt2i  = int  (l[24:27]) if l[24:27].strip() != '' else None,
                                            mgt3i  = int  (l[28:30]) if l[28:30].strip() != '' else None,
                                            mgt4   = float(l[31:43]) if l[31:43].strip() != '' else None,
                                            mgt5   = float(l[44:50]) if l[44:50].strip() != '' else None,
                                            mgt6   = float(l[51:62]) if l[51:62].strip() != '' else None,
                                            mgt7   = float(l[63:67]) if l[63:67].strip() != '' else None,
                                            mgt8   = float(l[68:74]) if l[68:74].strip() != '' else None,
                                            mgt9   = float(l[75:80]) if l[75:80].strip() != '' else None,
                                            mgt10i = int  (l[80:92]) if l[80:92].strip() != '' else None,
                                        )
                                        submgt.append(mgtprac)

                            for kkv, vkv in kv.items():
                                v = hrulines[vkv[0]-1][vkv[1]-1 : vkv[2]].strip()
                                # print(ihru, kkv, hrulines[vkv[0]-1])
                                hru_val[kkv] = float(v) if v else np.nan
                subhru.append(hru_val)

        self.subbasin = pd.DataFrame(sub,)
        self.subhru = pd.DataFrame(subhru,)
        self.subsol = pd.DataFrame(subsol,)
        self.submgt = pd.DataFrame(submgt,)

    def _loaddata(self, ):
        super()._loaddata()

        self.gridtype = 'hru'  #

        # parse file.cio
        self.TxtInOut = self.model_ws
        self.read_cio()
        # self.read_input_sub()

        # read the subbasin shapefile
        if is_true(self.shpsub):
            self._dat = self._getobj(self.shpsub)

        # read the HRU shapefile
        if is_true(self.shphru):
            self.hru = self._getobj(self.shphru)

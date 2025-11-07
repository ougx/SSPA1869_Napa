import os
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

import flopy

from .model import model
from ...utils.spatial import transform_grid
from ...utils.misc import is_true

_ITMUNI = {"u": 0, "s": 1, "m": 2, "h": 3, "d": 4, "y": 5}
_LENUNI = {"u": 0, "f": 1, "m": 2, "c": 3}


def calc_totim(perioddata):
    """@private
    calculate the time length for the time steps

    Args:
        perioddata (_type_): _description_
    """
    dts = []
    for iper, row in enumerate(perioddata):
        perlen = row[0]
        nstp = int(row[1])
        mult = row[2]
        if mult <= 1.:
            dt = perlen / nstp
        else:
            dt = perlen * (mult - 1) / (mult ** nstp - 1)

        dts += [((iper, istp), iper+1, istp+1, dt*mult**istp) for istp in range(nstp)]

    dts = pd.DataFrame(dts, columns='kstpkper period step dt'.split())
    dts['time'] = dts.dt.cumsum() # endding time of each time step
    return dts.set_index('kstpkper')



def create_dis_grid(dis, xoff=0., yoff=0., angrot=0., lfactor=1.):
    """@private
    test mf2000:
    import flopy
    mf = flopy.modflow.Modflow.load('2020.nam',
                                    model_ws=r'c:/Cloud/OneDrive - S.S. Papadopulos & Associates, Inc/0008-sspaPlot/Models/MF2K_Rotated',
                                    load_only='dis')
    mg = create_dis_grid(mf.dis, 366084, 1523595, -25)
    mg.plot(column='botm0', cmap='jet')
    mg.plot(column='x', cmap='jet')


    test mf6:
    import flopy
    sim = flopy.mf6.MFSimulation.load(sim_ws=r'c:/Cloud/OneDrive - S.S. Papadopulos & Associates, Inc/0008-sspaPlot/Models/MF6_DIS',
                                      load_only=['npf'], verbosity_level=0)
    mf = sim.get_model()
    dis = mf.dis
    mg = create_dis_grid(dis, )
    mg.plot(column='botm0', cmap='jet')
    mg.plot(column='x', cmap='jet')
    """

    xxv = np.cumsum([0] + dis.delr.array.tolist())
    yyv = np.cumsum([0] + dis.delc.array.tolist()[::-1])[::-1]
    ncol = len(xxv) - 1
    nrow = len(yyv) - 1

    xxc = (xxv[:-1] + xxv[1:]) * 0.5
    yyc = (yyv[:-1] + yyv[1:]) * 0.5

    xxv, yyv = transform_grid(*np.meshgrid(xxv, yyv), xoff, yoff, angrot, lfactor)
    xxc, yyc = transform_grid(*np.meshgrid(xxc, yyc), xoff, yoff, angrot, lfactor)


    ic, ir = np.meshgrid(range(ncol), range(nrow))
    ic, ir = ic.flatten(), ir.flatten()

    gdf = gpd.GeoDataFrame({
            'row': ir,
            'column':ic,
            'x': xxc.flatten(),
            'y': yyc.flatten(),
        }, geometry=[
        Polygon([(xxv[iir  , iic  ], yyv[iir  , iic  ]),
                    (xxv[iir  , iic+1], yyv[iir  , iic+1]),
                    (xxv[iir+1, iic+1], yyv[iir+1, iic+1]),
                    (xxv[iir+1, iic  ], yyv[iir+1, iic  ]),]) for iic, iir in zip(ic, ir)])


    gdf['botm0'] = dis.top.array.flatten()
    botm = dis.botm.array
    for ilay in range(botm.shape[0]):
        gdf['botm'+str(ilay+1)] = botm[ilay].flatten()

    gdf.index = np.arange(1, gdf.shape[0]+1)
    gdf.index.name = 'node'
    return gdf, xxv, yyv


def create_disv_grid(disv, xoff=0., yoff=0., angrot=0., lfactor=1.):
    """@private
    test mf6:
    import flopy
    sim = flopy.mf6.MFSimulation.load(sim_ws=r'c:/Cloud/OneDrive - S.S. Papadopulos & Associates, Inc/0008-sspaPlot/Models/MF6_DISV',
                                      load_only=['npf'], verbosity_level=0)
    mf = sim.get_model()
    mg = create_disv_grid(mf.disv, )
    mg.plot(column='botm0', cmap='jet')
    mg.plot(column='x', cmap='jet')
    """

    angrot_radians = np.pi * angrot / 180.

    xx = disv.vertices.array['xv']
    yy = disv.vertices.array['yv']

    # vertex_coords = np.array([
    #     (xx * np.cos(angrot_radians) - yy * np.sin(angrot_radians)) * lfactor + xoff,
    #     (xx * np.sin(angrot_radians) + yy * np.cos(angrot_radians)) * lfactor + yoff
    # ]).T
    vertex_coords = np.array(transform_grid(xx, yy, xoff, yoff, angrot, lfactor)).T

    xs = []
    ys = []
    polys = []
    for cell in disv.cell2d.array:
        xs.append(cell[1])
        ys.append(cell[2])
        polys.append(Polygon([vertex_coords[int(i)] for i in cell.tolist()[4:4+cell[3]]]))

    gdf = gpd.GeoDataFrame(dict(
        x=xs,
        y=ys,
        geometry=polys
    ))

    gdf['botm0'] = disv.top.array
    botm = disv.botm.array
    for ilay in range(disv.nlay.data):
        gdf['botm'+str(ilay+1)] = botm[ilay]

    gdf.index = np.arange(1, gdf.shape[0]+1)
    gdf.index.name = 'node'

    return gdf

def read_gsf(gsffilename):

    """@private
    read a modflow-usg gsf file
    """

    with open(gsffilename, "r") as f:
        # header
        l = '#'
        while l.startswith('#'):
            l = f.readline().lstrip()

        lines = [l] + f.readlines()

    # dimension
    ncell, nlay, iz, ic = [int(n) for n in lines[1].split()]

    #  the number of element vertex definitions
    nvertex = int(lines[2].split()[0])

    #  the coordinates of each vertex
    vertex_coords = [tuple([float(r) for r in l.split()[:2]]) for l in lines[3:3+nvertex]]

    # inode x y z lay m (ivertex(i),i=1,m)
    xs = []
    ys = []
    zs = []
    polys = []
    layers = []
    for cell in range(ncell):
        line = lines[3+nvertex+cell].split()
        x, y, z  = [float(r) for r in line[1:4]]
        lay, m = [int(r) for r in line[4:6]]
        xs.append(x)
        ys.append(y)
        zs.append(z)
        layers.append(lay)
        polys.append(Polygon(pd.unique([vertex_coords[int(i)-1] for i in line[6:6+m]]))) # only used the unique vertex with different horzontal location

    gdf = gpd.GeoDataFrame(dict(
        x=xs,
        y=ys,
        z=zs,
        layer=layers,
        geometry=polys
    ))

    gdf['node'] = gdf.groupby('layer').x.transform(lambda x: np.arange(1, x.shape[0]+1))
    return gdf.set_index(['layer', 'node'])


class mf(model):
    """`mf` define a MODFLOW model object."""
    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        self.model_ws = self.dict.get('model_ws', '.')
        """set the model working directory. Default is '.'
        """

        self.namefile = self.dict.get('namefile', None)
        """set MODFLOW name file if it is not MF6."""

        self.submodel = self.dict.get('submodel', 1)
        """set the MF6 submodel. Default is the first flow/transport model.

        Example:
        >>> submodel = 1  # first model
        >>> submodel = 'gwf1'  # model named gwf1
        '"""

        self.gsf = self.dict.get('gsf', None)
        """set the gsf file name. `gsf` file is needed for unstructured grids."""

        self.version = self.dict.get('version', 'mfnwt').lower()
        """set the MODFLOW version. Default is 'mfnwt'

        Example:
        >>> version = 'mf2k'
        """

        self._totim = None
        self._xvert = None
        self._yvert = None
        self._dx = None
        self._dy = None


    def _loaddata(self, ):
        self._nplot = 0
        if self.version == 'mf6':
            submodels = {}
            submodel = self.submodel
            with open(os.path.join(self.model_ws, 'mfsim.nam')) as f:
                for l in f:
                    if l.lstrip().upper().startswith('BEGIN MODELS'):
                        break

                for l in f:
                    l = l.strip()
                    if l.lstrip().upper().startswith('END MODELS'):
                        break

                    ll = l.strip().split()
                    submodels[ll[2].lower()] = ll[1]
            if len(submodels) == 1:
                self.namefile = ll[1]
                self.submodel = list(submodels.keys())[self.submodel-1]
            else:
                if isinstance(submodel, str):
                    if submodel.lower() in submodels:
                        self.namefile = submodels[submodel.lower()]
                elif isinstance(self.submodel, int):
                    self.namefile = list(submodels.values())[self.submodel-1]
                    self.submodel = list(submodels.keys())[self.submodel-1]
                else:
                    raise ValueError(f'Submodel {submodel} not found in mfsim.')

        if self.verbose > 0:
           if self.version == 'mf6':
               print(f'    Loading MF6 submodel {self.submodel}')
           else:
               assert self.namefile is not None, 'namefile must be defined for non-MF6 models.'
               print(f'    Loading {self.fullname}')
               self.namefile = self.namefile.strip()

        if not(':' in self.namefile or self.namefile.startswith('/')):
            self.namefile_fullpath = os.path.join(self.model_ws, self.namefile)

        if not os.path.exists(self.namefile_fullpath):
            raise FileExistsError(f'Namefile {self.namefile_fullpath} does not exists.')

        with open(self.namefile_fullpath) as f:
            mfname_lines = list(filter(lambda x: not (x.startswith('#') or x.strip() == ''), f.readlines()))

        if self.version == 'mf6':
            # set up grid
            mf6sim = flopy.mf6.MFSimulation.load(
                sim_ws=self.model_ws,
                strict=False,
                verbosity_level=0,
                write_headers=False,
                load_only=['npf', 'sto'],
                verify_data=False
            )
            self._totim = calc_totim(mf6sim.tdis.perioddata.array)
            model = mf6sim.get_model(self.submodel)
            self.packages = [p.upper() for p in model.get_package_list()]
            idomain = None
            if 'DIS' in self.packages:
                dis = model.dis
                self._nlay = dis.nlay.data
                self._nrow = dis.nrow.data
                self._ncol = dis.ncol.data
                self._dx = dis.delr.data
                self._dy = dis.delc.data
                self._dat, self._xvert, self._yvert = create_dis_grid(dis, self.xoff, self.yoff, self.angrot, self.lfactor)
                idomain = dis.idomain.array
            elif 'DISV' in self.packages:
                disv = model.disv
                self._nlay = disv.nlay.data
                self._nrow = 0
                self._ncol = disv.ncpl.data
                self._dat = create_disv_grid(disv, self.xoff, self.yoff, self.angrot, self.lfactor)
                idomain = disv.idomain.array
            elif 'DISU' in self.packages:
                disu = model.disu
                self._nlay = 0
                self._nrow = 0
                self._ncol = disu.nodes.data
                self._dat = read_gsf(self.gsf)
                self._dat['botm0'] = disu.top.array.flatten()
                self._dat['botm1'] = disu.bot.array.flatten()
                idomain = disu.idomain.array
            else:
                raise ValueError(f'Unknow grid type in MF6 {self.name}')
            # set up hydraulic conductivity
            hk = model.npf.k.array
            ss = None
            sy = None
            if 'STO' in self.packages:
                ss = model.sto.ss.array
                if model.sto.sy is not None:
                    sy = model.sto.sy.array
            if model.npf.k33overk:
                an = 1. / model.npf.k33.array
                vk = hk / an
            else:
                vk = model.npf.k33.array
                an = hk / vk
            if self.nlay > 0:
                # dis or disv
                hk =hk.reshape([self.nlay, -1])
                vk =vk.reshape([self.nlay, -1])
                an =an.reshape([self.nlay, -1])
                if ss is not None:
                    ss = ss.reshape([self.nlay, -1])
                    sy = sy.reshape([self.nlay, -1])
                for ilay in range(self.nlay):
                    self._dat['hk'+str(ilay+1)] = hk[ilay].flatten()
                    self._dat['vk'+str(ilay+1)] = vk[ilay].flatten()
                    self._dat['an'+str(ilay+1)] = an[ilay].flatten()
                    if ss is not None:
                        self._dat['ss'+str(ilay+1)] = ss[ilay].flatten()
                    if sy is not None:
                        self._dat['sy'+str(ilay+1)] = sy[ilay].flatten()
                    if is_true(idomain):
                        self._dat[f'ibound{ilay+1}'] = idomain[ilay].flatten()
                    else:
                        self._dat[f'ibound{ilay+1}'] = 1
            else:
                # disu
                self._dat['hk1'] = hk
                if is_true(idomain):
                    self._dat['ibound1'] = idomain.flatten()
                else:
                    self._dat['ibound1'] = 1
        else:

            packages = []
            for l in mfname_lines:
                if l.lstrip()[:3].lower() == 'dis' : packages.append(l.lstrip()[:4].strip().upper())
                if l.lstrip()[:3].lower() == 'bas' : packages.append(l.lstrip()[:4].strip().upper())
                if l.lstrip()[:3].lower() == 'bcf' : packages.append(l.lstrip()[:4].strip().upper())
                if l.lstrip()[:3].lower() == 'lpf' : packages.append(l.lstrip()[:4].strip().upper())
                if l.lstrip()[:3].lower() == 'upw' : packages.append(l.lstrip()[:4].strip().upper())
                if l.lstrip()[:4].lower() == 'zone': packages.append(l.lstrip()[:4].strip().upper())
                if l.lstrip()[:4].lower() == 'pval': packages.append(l.lstrip()[:4].strip().upper())
                if l.lstrip()[:4].lower() == 'mult': packages.append(l.lstrip()[:4].strip().upper())


            if 'DISU' in packages:
                model = flopy.mfusg.MfUsg.load(self.namefile, model_ws=self.model_ws, check=False, load_only=packages)
                self._nlay = 0
                self._nrow = 0
                self._ncol = model.disu.nodes
                self._dat = read_gsf(self.gsf)
                self._dat['botm0'] = model.disu.top.array.flatten()
                self._dat['botm1'] = model.disu.bot.array.flatten()
                perioddata = zip(model.disu.perlen.array,model.disu.nstp.array,model.disu.tsmult.array,)
                self._totim = calc_totim(perioddata)
            else:
                model = flopy.modflow.Modflow.load(
                    self.namefile,
                    version=self.version,
                    model_ws=self.model_ws,
                    load_only=packages,
                    check=False
                )

                self._nlay = model.nlay
                self._nrow = model.nrow
                self._ncol = model.ncol
                self._dx = model.dis.delr.array
                self._dy = model.dis.delc.array

                self._dat, self._xvert, self._yvert = create_dis_grid(model.dis, self.xoff, self.yoff, self.angrot, self.lfactor)
                perioddata = zip(model.dis.perlen.array,model.dis.nstp.array,model.dis.tsmult.array,)
                self._totim = calc_totim(perioddata)

            ss = None
            sy = None
            hk = None
            vk = None
            layvka = None
            if 'LPF' in packages:
                hk = model.lpf.hk.array
                vk = model.lpf.vka.array
                layvka = model.lpf.layvka.array
                if model.lpf.ss is not None:
                    ss = model.lpf.ss.array
                if model.lpf.sy is not None:
                    sy = model.lpf.sy.array
            elif 'UPW' in packages:
                hk = model.upw.hk.array
                vk = model.upw.vka.array
                layvka = model.upw.layvka.array
                if model.upw.ss is not None:
                    ss = model.upw.ss.array
                if model.upw.sy is not None:
                    sy = model.upw.sy.array
            elif 'BCF6' in packages:
                hk = []
                for ilay, laycon in enumerate(model.bcf6.laycon):
                    if laycon==0 or laycon == 2:
                        thick = self._dat[f'botm{ilay}']-self._dat[f'botm{ilay+1}']
                        hk.append(model.bcf6.tran.array[ilay].flatten() / thick.values)
                    else:
                        hk.append(model.bcf6.hy.array[ilay].flatten())
            if layvka is not None:
                an = np.zeros_like(hk, dtype=float)
                for ilay, ivka in enumerate(layvka):
                    if ivka == 0:
                        an[ilay] = hk[ilay] / vk[ilay]
                    else:
                        an[ilay] = vk[ilay]
                        vk[ilay] = hk[ilay] / an[ilay]
            if 'DISU' in packages:
                self._dat['hk1'] = np.concatenate(hk).flatten() if hk.ndim > 1 else hk
                self._dat['ibound1'] = model.bas6.ibound.array.flatten()
            else:
                for ilay in range(model.nlay):
                    self._dat[f'hk{ilay+1}'] = hk[ilay].flatten()
                    self._dat[f'vk{ilay+1}'] = vk[ilay].flatten()
                    self._dat[f'an{ilay+1}'] = an[ilay].flatten()
                    if ss is not None:
                        self._dat['ss'+str(ilay+1)] = ss[ilay].flatten()
                    if sy is not None:
                        self._dat['sy'+str(ilay+1)] = sy[ilay].flatten()
                    self._dat[f'ibound{ilay+1}'] = model.bas6.ibound.array[ilay].flatten()

        for c in self._dat.columns:
            if c.startswith('hk') or c.startswith('vk') or c.startswith('an') or c.startswith('ss') or c.startswith('sy'):
                self._dat[c] = np.where(self._dat["ibound"+c[2:]]!=0, self._dat[c], np.nan)


        if self.crs is not None:
            self._dat.crs = self.crs
        self._writeshp()

    @property
    def totim(self, ):
        """@private"""
        if self._totim is None:
            self._loaddata()
        return self._totim

    @property
    def xvert(self, ):
        """@private"""
        if self.is_structuredgrid:
            return self._xvert
    @property
    def yvert(self, ):
        """@private"""
        if self.is_structuredgrid:
            return self._yvert
    @property
    def dx(self, ):
        """@private"""
        if self.is_structuredgrid:
            if self._dx is None:
                self._loaddata()
            return self.lfactor * self._dx
    @property
    def dy(self, ):
        """@private"""
        if self.is_structuredgrid:
            if self._dy is None:
                self._loaddata()
            return self.lfactor * self._dy

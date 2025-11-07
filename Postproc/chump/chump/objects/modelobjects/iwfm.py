import os
from io import StringIO

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from ...utils.io import read_iwmffile
from .model import model


class iwfm(model):
    """define an IWFM model object."""

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        self.model_ws = self.dict.get('model_ws', '.')
        """set the model working directory. Default is '.'
        """

        self.simfile = self.dict['simfile']
        """set the simulation file of IWFM."""
        if not(':' in self.simfile or self.simfile.startswith('/')):
            self.simfile = os.path.join(self.model_ws, self.simfile)

    def get_elem_val(self, nodevalues):
        """@private"""
        return np.sum([nodevalues[self.dat['vert' + str(i+1)].values-1] * \
                       self.dat['frac' + str(i+1)].values.reshape([-1, 1]) for i in range(4)],
                      axis=0)


    def _loaddata(self, ):
        super()._loaddata()

        self.gridtype = 'fe'  # finite element

        # parse the simfile
        simfile = self.simfile
        simdir = os.path.dirname(simfile)
        lines = read_iwmffile(simfile)

        self.preprocbin = os.path.join(simdir,lines[3].strip().split()[0])
        self.gwfile     = os.path.join(simdir,lines[4].strip().split()[0])
        self.streamfile = os.path.join(simdir,lines[5].strip().split()[0])

        self.read_preprocbin()

        if self.crs is not None:
            self._dat.crs = self.crs
        self.writeshp()



    def read_preprocbin(self):
        '''@private
        Class_AppGrid.f90@ReadProcessedAppGridData
        '''

        with open(self.preprocbin, 'rb') as f:

            ## Grid

            ### diminsions
            NNodes, NElements, NFaces, NSubregions, NBoundaryFaces  = np.fromfile(f, 'i4', 5)
            x     = np.fromfile(f, 'f8', NNodes) * self.lfactor
            y     = np.fromfile(f, 'f8', NNodes) * self.lfactor

            nvert = np.fromfile(f, 'i4', NElements)
            ivert = np.fromfile(f, 'i4', NElements * 4).reshape([-1, 4])

            self._nrow = 0
            self._ncol = NElements
            xy = np.array([x, y]).T
            nivert = np.hstack([nvert.reshape([NElements, -1]), ivert])
            geometry = np.apply_along_axis(lambda r: Polygon(xy[r[1:r[0]+1]-1]), 1, nivert)

            self._dat = gpd.GeoDataFrame(nivert, index=range(1, NElements+1), columns=['nvert'] + ['vert'+str(i+1) for i in range(4)], geometry=geometry)
            centroid = self._dat.centroid
            self._dat['x'] = centroid.x
            self._dat['y'] = centroid.y
            self._dat.index.name = 'node'       # TODO: for now called it "node" to be consistent with MF6

            ### NNodes of AppNode
            for i in range(NNodes):
                ID                   = np.fromfile(f, 'i4', 1)[0]
                Area                 = np.fromfile(f, 'f8', 1)[0]
                BoundaryNode         = np.fromfile(f, 'i4', 1)[0]
                NConnectedNode       = np.fromfile(f, 'i4', 1)[0]
                NFaceID              = np.fromfile(f, 'i4', 1)[0]
                iSizeSurroundingElem = np.fromfile(f, 'i4', 1)[0]
                iSizeConnectedNode   = np.fromfile(f, 'i4', 1)[0]

                SurroundingElement   = np.fromfile(f, 'i4', iSizeSurroundingElem)
                ConnectedNode        = np.fromfile(f, 'i4', iSizeConnectedNode)
                FaceID               = np.fromfile(f, 'i4', NFaceID)
                ElemID_OnCCWSide     = np.fromfile(f, 'i4', NFaceID)
                IrrotationalCoeff    = np.fromfile(f, 'f8', NFaceID)

            ### NElements of AppElement
            Subregions          = np.zeros(NElements, dtype=int)
            VertexAreaFractions = np.zeros([NElements, 4])
            for i in range(NElements):
                ID                   = np.fromfile(f, 'i4', 1)[0]
                Subregion            = np.fromfile(f, 'i4', 1)[0]
                Area                 = np.fromfile(f, 'f8', 1)[0]
                NFaceID              = np.fromfile(f, 'i4', 1)[0]
                nVertexArea          = np.fromfile(f, 'i4', 1)[0]
                nIntegral_DELShpI_DELShpJ     = np.fromfile(f, 'i4', 1)[0]
                nIntegral_Rot_DELShpI_DELShpJ = np.fromfile(f, 'i4', 1)[0]

                FaceID               = np.fromfile(f, 'i4', NFaceID)
                VertexArea           = np.fromfile(f, 'f8', nVertexArea)
                VertexAreaFraction   = np.fromfile(f, 'f8', nVertexArea)
                Integral_DELShpI_DELShpJ        = np.fromfile(f, 'f8', nIntegral_DELShpI_DELShpJ)
                Integral_Rot_DELShpI_DELShpJ    = np.fromfile(f, 'f8', nIntegral_Rot_DELShpI_DELShpJ)

                Subregions[i] = Subregion
                VertexAreaFractions[i, :nVertexArea] = VertexAreaFraction

            self._dat['subregion'] = Subregions
            self._dat[['frac'+str(i+1) for i in range(4)]] = VertexAreaFractions

            ### nface of AppFace
            for i in range(NFaces):
                node1, node2, element1, element2 = np.fromfile(f, 'i4', 4)
                length                           = np.fromfile(f, 'f8', 1)[0]
                BoundaryFace                     = np.fromfile(f, 'i4', 1)[0]

            BoundaryFaceList = np.fromfile(f, 'i4', NBoundaryFaces)

            ### NSubregions of AppSubregion
            regions = []
            for i in range(NSubregions):
                ID                   = np.fromfile(f, 'i4' , 1)[0]
                Name                 = np.fromfile(f, 'S50', 1)[0].decode()
                NRegionElements      = np.fromfile(f, 'i4' , 1)[0]
                NNeighborRegions     = np.fromfile(f, 'i4' , 1)[0]

                Area                 = np.fromfile(f, 'f8' , 1)[0]
                RegionElements       = np.fromfile(f, 'i4' , NRegionElements)

                RegionNo             = np.fromfile(f, 'i4' , NNeighborRegions)
                NRegBndFace          = np.fromfile(f, 'i4' , NNeighborRegions)
                RegBndFace           = np.fromfile(f, 'i4' , sum(NRegBndFace))
                regions.append(Name.strip())

            ## Stratigraphy
            NLayers        = np.fromfile(f, 'i4' , 1)[0]
            TopActiveLayer = np.fromfile(f, 'i4' , NNodes)
            ActiveNode1D   = np.fromfile(f, 'i4' , NNodes * NLayers).reshape([NLayers, NNodes])
            GSElev         = np.fromfile(f, 'f8' , NNodes)
            TopElev        = np.fromfile(f, 'f8' , NNodes * NLayers).reshape([NLayers, NNodes])
            BottomElev     = np.fromfile(f, 'f8' , NNodes * NLayers).reshape([NLayers, NNodes])

            botm = np.vstack([TopElev, BottomElev[-1:,]]).T

            self._nlay = NLayers
            self._vert = pd.concat([pd.DataFrame(xy, columns=['x', 'y']), pd.DataFrame(botm, columns=['botm'+str(i) for i in range(NLayers+1)])], axis=1)
            self._vert.index = range(1, NNodes+1)
            self._vert.index.name = 'node'


    def read_gwfile(self):
        """@private"""
        lines = read_iwmffile(self.gwfile)
        self.ngwhyd = int(lines[21].strip().split()[0])

        lines = lines[24+self.ngwhyd:]

        self.ngwf = int(lines[0].strip().split()[0])

        lines = lines[2+self.ngwf:]

        self.ngwpargrp = int(lines[0].strip().split()[0])

        FX,FKH,FS,FN,FV,FL = lines[1].strip().split()[:6]
        TUNITKH = lines[2].strip().split()[0]
        TUNITV  = lines[3].strip().split()[0]
        TUNITL  = lines[4].strip().split()[0]

        lines = lines[5:]

        for i in range(self.ngwpargrp):
            groupnodes = lines[0]
            NDP = int(lines[1].strip().split()[0])
            NEP = int(lines[2].strip().split()[0])
            lines = lines[2+NEP+NDP:]

        if self.ngwpargrp == 0:
            parameters = np.loadtxt(StringIO(''.join([l[15:] for l in lines[:self.nlay*self.ncpl]])))
            self.pars = pd.DataFrame(parameters, columns='hk ss sy vka vk'.split(), index=pd.MultiIndex.from_product(range(1, self.ncpl+1), range(1, self.nlay+1)))


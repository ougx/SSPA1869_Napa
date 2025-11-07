# plot objects
from .mf import mf
from .iwfm import iwfm
from .swat import swat


_readme = """Model objects are in fact a special type of data objects. A model object parses the model
input files and stores the model discretization and parameterization information through its `loaddata()` function.
The model grid is stored as a GeoPandas’ GeoDataFrame, which is a subclass of the Pandas’ DataFrame with a “geometry”
column that store the geometry of each element such as a grid cell or a HRU (hydrologic response unit).
In the current version, there are three model subclass objects: `.mf`, `.iwfm` and `.swat`.
Using the FloPy package, the `mf` object handles different versions of MODFLOW including
MODFLOW 2000/2005, MODFLOW-NWT, MODFLOW-USG and MODFLOW 6.
The `.iwfm` and `.swat` objects represent the IWFM and SWAT models, respectively.
An IWFM parser is developed to read the IWFM simulation file, preprocessor binary file and groundwater input file.
The preprocessor binary file is created using the IWFM Pre-processor as the first step of running IWFM.
The binary file includes information of the finite element (FE) grid and aquifer stratigraphy.
The swatResultReader (https://github.com/ougx/swatResultReader) is integrated to read input and output files of
the SWAT model. Besides the model files, the SWAT geospatial information is defined by a subbasin shapefile.
These shapefiles are usually created during the preprocessing phase using GIS tools such as ArcSWAT.

"""

__doc__ = _readme + '\n'.join([f'        - {__name__}.'+k for k in dir() if not k.startswith('_')])

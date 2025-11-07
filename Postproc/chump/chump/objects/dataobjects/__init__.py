# data objects

from .csv import csv
from .img import img
from .iwfmhead import iwfmhead
from .line import line
from .point import point
from .modelarray import modelarray
from .mflst import mflst
from .mfpkg import mfpkg
from .mfbin import mfbin
# from .mfbints import mfbints
from .mfcbb import mfcbb
from .mfflux import mfflux
from .raster import raster
from .shp import shp
from .shpts import shpts
from .swatoutput import swatoutput
from .pstres import pstres
from .hobout import hobout
from .scalebar import scalebar
from .tseries import tseries
from .sfrout import sfrout
from .table import table
from .northarrow import northarrow
from .well import well

_readme = """Data objects extract and store spatial and temporal information of diverse formats,
and convert them into a uniform data abstraction represented as Pandas dataframes with highly efficient
indexing/slicing capability. Three top data classes, `.mobject`, `.mobject2d`, and `.mobject3d`, are
available to represent data with various degrees of dimensionality. The `.mobject` class serves as the
root superclass for all other derived subclasses. `.mobject2d` and `.mobject3d` classes correspond to tabular
and gridded data structures, respectively. These object classes possess identical functions across their
respective subclasses, with run-time polymorphism being a key feature of CHUMP's implementation.
For example, the `loaddata()` function is used to read and parse input files for all the objects.
CHUMP can read a large number of file types covering a wide range of geospatial and model data.
This helps users readily convey the model results more accurately and intuitively.

"""

__doc__ = _readme + '\n'.join([f'        - {__name__}.'+k for k in dir() if not k.startswith('_')])

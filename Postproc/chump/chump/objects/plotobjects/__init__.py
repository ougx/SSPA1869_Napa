
# plot objects
from .borelog import borelog
from .edfplot import edfplot
from .figure import figure
# from .lstplot import lstplot
from .mapplot import mapplot
from .scatterplot import scatterplot
from .vplot import vplot
from .tsplot import tsplot

_readme = """Visualization objects are used to create images or animations based on the data objects.
Currently, five major visualization types are supported in CHUMP. The products of the visualization objects are either PDF pages or PNG files. Named bookmarks for each page
are added in the PDF. When there are multiple rows (such as different layers or times) in the plotting data,
CHUMP will loop through the data rows and create the PDF page or PNG file for each row. If the augment writemovie
is specified, an animation will be automatically produced.

"""

__doc__ = _readme + '\n'.join([f'        - {__name__}.'+k for k in dir() if not k.startswith('_')])

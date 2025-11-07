
from .extract import extract
from .interp import interp
from .plot import plot
from .prepare import prepare
from .stat import stat


_readme = """Operations set up the workflow of CHUMP.

"""

__doc__ = _readme + '\n'.join([f'        - {__name__}.'+k for k in dir() if not k.startswith('_')])

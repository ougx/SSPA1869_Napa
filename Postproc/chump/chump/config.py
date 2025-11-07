import os
import sys

import tomli
# import yaml
import encodings.idna # import for pyinstaller

#import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
from .objects import dataobjects, modelobjects
from . import operations
from copy import copy

from .utils.misc import iterate, concat_dicts
from .utils.io import pretty_print_dict

sys.setrecursionlimit(150000)
plt.ioff()

class config():
    """
    The primary objective driving the development of CHUMP is to streamline
    post-processing by automating repetitive tasks, while ensuring applicability.
    The interface for user input and production of output is simplified through
    the use of a configuration file. This technique is a form of configurable programming,
    frequently employed in computer engineering to overcome limitations of traditional
    graphical user interfaces (GUIs). A significant benefit of this approach is that users
    can adjust settings through the configuration file without the need for script rewrites
    or application recompilations. The configuration file is an easy-to-use
    ASCII input file that can be created by non-programmers.

    YAML (https://yaml.org/) and TOML (https://toml.io/) are two human-readable data serialization
    formats supported by CHUMP. The configuration file written in these two formats can be parsed
    and stored as a hierarchical dictionary structure. This dictionary structure is then used to
    define various data objects with different levels of complexity. One notable advantage of this
    approach is the seamless unpacking of the dictionary with double asterisks as arbitrary keyword
    arguments in Python functions. This facilitates the convenient utilization of the configuration
    data to specify diverse parameters and options in the Python code. Furthermore, both YAML and TOML
    natively support assignment of date/time, array/list, boolean values. In addition, CHUMP allows
    for the implementation of object property inheritance and overwriting by defining the parent object.

    `config` class is the app entry to load configuration file. It also sets up global parameters which
    will be used for all objects if it is not overwritten.
    """

    def __init__(self, args):
        """@private"""

        self.dict = {}
        """@private"""

        self.configfiles = []
        """@private"""

        self.verbose = int(args.verbose)
        """@private"""

        self.include: list = []
        """`inlucde` tells `chump` to insert the contents of other files into the main configuration file.
        This provides flexibility how different section of the configuration file can be organized.

        When `include` is used, it should be placed as the first parameter. Nested `include` (include files which also include other files) is allowed.

        Example:
        >>> include = ['../model/mf.toml', '../obsdata/obs.toml']
        """
        self.configfiles.append(args.config)
        self.dict = self._read_config(args.config)

        self.masterdir = self.dict.get('masterdir', '.')
        '''
        set the the working directory, default is the directory from which the chump is called.

        Example:
        >>> masterdir = '../../model'
        '''
        os.chdir(self.masterdir)

        for k, v in self.dict.items():
            setattr(self, k, copy(v))

        # allocate PDF bookmarks
        self.bookmarks = {}
        """@private"""

        self.epsg = self.dict.get('epsg', None)
        '''
        set the global coordinate reference system for the object using EPSG. e.g. `epsg = 4326`.
        '''

        self.crs = self.dict.get('crs', None)
        '''
        set the global coordinate reference system using a string of certain format. See details in https://geopandas.org/en/stable/docs/user_guide/projections.html.
        '''

        if self.crs is None and self.epsg is not None:
            self.crs = f"epsg:{self.epsg}"

        plotstyle = self.dict.get('plotstyle', 'ggplot')
        '''
        set the matplotlib plotting style, see [style reference](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html) for available styles. default value is `'ggplot'`.
        '''
        plt.style.use(plotstyle)

        self.err_as_oms = self.dict.get('err_as_oms', True)
        """
        When `err_as_oms = true`, `residual = observed - simulated`; otherwise `residual = simulated - observed`; default is `true`.
        """

        self.time_unit = self.dict.get('time_unit', 'D')
        """
        set the global time unit for the object. default is "days".
        """

        self.start_date = None
        """
        set the global starting date for time related data. The time will be parsed into datetime formats using the starting date and time unit.
        If not set, the time will be in numeric formats.
        """

        if 'start_date' in self.dict:
            self.start_date = pd.Timestamp(self.dict['start_date'])

        self.metadata = self.dict.get('metadata', {})
        """`metadata` set the metadata for all the PDF created from this PDF, if not overwritten.

        Example:
        >>> metadata = {title='SSPA Project 111',
        >>>             author='SSPAmodler',
        >>>             subject='Groundwater model results'}
        """
        # pretty_print_dict(self.dict)
        self._initialize_elements()

        if self.verbose > 1:
            print('='*30, 'Full dict', '='*30)
            pretty_print_dict(self.dict)


    def _read_config(self, cfile):
        """
        parse configuration file
        """

        if self.verbose > 0:
            print(f'Loading {cfile} ...')
        if cfile.lower().endswith('toml'):
            with open(cfile, "rb") as f:
                config_dict = tomli.load(f)
        # elif cfile.lower().endswith('yml') or cfile.lower().endswith('yaml'):
        #     with open(cfile, "r") as f:
        #         config_dict = yaml.safe_load(f)
        else:
            raise IOError('Unsupported configuration file format.')

        if self.verbose > 5:
            print('='*30, cfile, '='*30)
            pretty_print_dict(config_dict)

        while 'include' in config_dict:
            for c in iterate(config_dict.pop('include')):
                self.configfiles.append(c)
                config_dict = concat_dicts(config_dict, self._read_config(c))

        if 'pdf' in config_dict:
            if 'plot' in config_dict:
                config_dict['plot'].update(config_dict.pop('pdf'))
            else:
                config_dict['plot'] = config_dict.pop('pdf')

        return config_dict


    def _initialize_elements(self):
        """
        initialize the elements
        """
        for o in [dataobjects, modelobjects]:
            for e in dir(o):
                if e.startswith("__"): continue
                element = getattr(self, e, None)
                if element is not None:
                    for k in element.keys():
                        element[k] = getattr(o, e)(k, self)


    def _check_writable_file(self, ):

        try:
            with open(self.outfile, 'wb') as f:
                f.write(b'test')
            if self.outfile.endswith('.pdf'):
                with open(self.outfile.replace('.pdf', '._pdf'), 'wb') as f:
                    f.write(b'test')
        except:
            print(f'{self.outfile} is not writable. Please check if it is closed.')
            return
        finally:
            if os.path.exists(self.outfile):
                os.remove(self.outfile)
            if os.path.exists(self.outfile.replace('.pdf', '._pdf')):
                os.remove(self.outfile.replace('.pdf', '._pdf'))

    def runCmd(self, opt, name):
        """
        @private entry to prepare well file based on well location and model grid
        """
        if opt == 'plot':
            self.outfile = f'{name}.pdf'
        else:
            self.outfile = f'{name}.csv'

        self._check_writable_file()

        getattr(operations, opt)(name, self).run()

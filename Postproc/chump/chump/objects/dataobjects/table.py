from ..mobject2d import mobject2d
from ...utils.io import read_table
import pandas as pd

class table(mobject2d):
    """
    generic table object with the `id` column.
    """

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        self.source: str = self.dict.get('source', None)
        """set the file path of this data. """

        self.sheet = self.dict.get('sheet')
        """`sheet` set the sheet name when reading data from an Excel workbook. Default is None.
        """

        self.sep = self.dict.get('sep', ',')
        """`sep` set delimiter in the data file. Default is ",".
        """

        self.legendlabel = self.dict.get('legendlabel', self.plotarg.get('label', self.name))
        """set the name displayed in the legend, default is the object name
        """

    def readdata(self, ):
        df = read_table(self.source, self.sheet, crs=self.crs, sep=self.sep)

        if self.idcol:
            self.dimcols.append(self.idcol)
            df.set_index(self.idcol, drop=True, inplace=True)

        self._dat = df
        # self._nplot = len(pd.unique(self._dat.index.get_level_values(self.idcol)))

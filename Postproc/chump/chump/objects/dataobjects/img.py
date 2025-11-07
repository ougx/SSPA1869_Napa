from matplotlib import image

from ..mobject import mobject


class img(mobject):
    """define an image object that reads *.jpg, *.png etc."""
    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        self.source: str = self.dict.get('source', None)
        """set the file path of this data. """

        self.extent: str = self.dict.get('extent', None)
        """set the image extent as [left, right, bottom, top]. """

        self.plotarg = self.plotarg

        self._dat = None

    def _loaddata(self):
        self._dat = image.imread(self.source)
        self._nplot = 0

    def plot(self, *args, **kwargs):
        """@private"""
        array = self.dat
        plotarg = {k:v for k, v in self.plotarg.items() if k != 'extent'}
        self.ax.imshow(array, extent=self.extent, **plotarg)


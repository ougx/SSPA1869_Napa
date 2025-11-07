
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib import patheffects

from ..mobject import mobject



class line(mobject):
    """define a line object by a series of coordinates"""
    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)
        self.xy = np.array(self.dict['xy'])
        """set the coordinates of the line

        Example:
        >>> xy = [[1,1],[2,2],[4,4]]
        """

        self.plotarg = self.plotarg

    def plot(self, *args, **kwargs):
        ls = self.ax.plot(*self.xy.T, **self.plotarg)
        if self.legend:
            return {getattr(ls[0], 'label', self.name): ls[0]}


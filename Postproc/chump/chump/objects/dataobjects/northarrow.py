
from matplotlib import patheffects

from ..mobject import mobject
from ...utils.plotting import zorders

class northarrow(mobject):
    """define a north arrow"""

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        self.xy = self.dict['xy']
        """set the coordinates to place the north arrow"""

        self.ha = self.dict.get('ha', 'center')
        """horizontal alignment, default is 'center'."""

        self.va = self.dict.get('va', 'center')
        """vertical alignment, default is 'center'."""

        self.color = self.dict.get('color', 'k')
        """set color, default is 'black'."""

        self.foreground = self.dict.get('foreground', 'w')
        """set foreground color, default is 'white'."""

        self.fontsize = self.dict.get('fontsize', 'x-large')
        """set fontsize, default is 'x-large'."""

        self.plotarg = self.plotarg

    def plot(self, *args, **kwargs):
        self._nplot = 0

        plotarg = self.plotarg.copy()
        if 'ha' not in plotarg:
            plotarg['ha'] = self.ha

        if 'va' not in plotarg:
            plotarg['va'] = self.va

        if 'color' not in plotarg:
            plotarg['color'] = self.color

        if 'fontsize' not in plotarg:
            plotarg['fontsize'] = self.fontsize

        if 'zorder' not in plotarg:
            plotarg['zorder'] = zorders['northarrow']

        if 'foreground' not in plotarg:
            foreground = self.foreground
        else:
            foreground = plotarg.pop('foreground')

        if 'path_effects' not in plotarg:
            plotarg['path_effects'] = [patheffects.withStroke(linewidth=5, foreground=foreground)]

        self.text = self.ax.text(*self.xy , u'\u25B2\nN', **plotarg)

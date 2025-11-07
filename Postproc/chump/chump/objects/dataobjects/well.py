
from ...utils.plotting import welllog
from ..mobject import mobject


class well(mobject):
    """plot a vertical pumping/monitoring well"""

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        self.xy = self.dict['xy']
        """set the location of the well top """

        self.radius = self.dict.get('radius', 1)
        """set the radius of the well top. default is 1. """

        self.depth = self.dict.get('depth')
        """set the well depth. if not set, it will use `scnbot`."""

        self.scntop = self.dict.get('scntop')
        """set the screen top depth"""

        self.scnbot = self.dict.get('scnbot')
        """set the screen bottom depth"""

        self.scnhatch = self.dict.get('scnhatch','---')
        """set the hatch pattern of the screen intervel, default is '--'"""

        self.ec = self.dict.get('ec', 'k')
        """set edge color, default is 'black'"""

        self.fc = self.dict.get('fc', 'w')
        """set the face color, default is 'white'"""

        self.zorder = self.dict.get('zorder', 999999)
        """set the zorder. default is 999999."""

    def _loaddata(self):
        self._nplot = 0
        if not self.depth:
            self.depth = self.scnbot


        self._dat = welllog(
            x=self.xy[0],
            top=self.xy[1],
            bot=self.xy[1]-self.depth,
            scntop=self.xy[1]-self.scntop,
            scnbot=self.xy[1]-self.scnbot,
            ec=self.ec,
            fc=self.fc,
            zorder=self.zorder,
            hatch=self.scnhatch,
            )

    def plot(self, *args, **kwargs):
        self.dat.plot(ax=self.ax)

    def vplot(self, *args, **kwargs):
        self.dat.plot(ax=self.ax)

    def update_plot(self, idx=None, hasData=True):
        pass

    def vupdate(self, idx=None, hasData=True):
        pass

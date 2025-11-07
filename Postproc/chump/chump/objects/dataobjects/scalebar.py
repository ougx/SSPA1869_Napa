from matplotlib import patheffects
from ..mobject import mobject
from ...utils.plotting import zorders, RotationAwareAnnotation2

class scalebar(mobject):
    """draw the scale bar."""

    def __init__(self, name, mplot, *args, **kwargs):
        super().__init__(name, mplot, *args, **kwargs)

        self.xy = self.dict['xy']
        """set the coordinates to place the scale bar"""

        self.xy1 = self.dict.get('xy1', [1, 0])
        """set the relative offset point used to define the direction of the scalebar. Default is [1, 0].
        The scale bar will be through `xy` and `xy1`."""

        self.unit = self.dict.get('unit', 'mile')
        """set the unit displaying in the scale bar."""

        self.unit_factor = self.dict.get('unit_factor', 5280)
        """set the scale factor converting from the displaying unit to the coordinate system unit. Default is 5280."""

        self.length = self.dict['length']
        """set the length of the scale bar in the unit displaying in the scale bar."""

        self.color = self.dict.get('color', 'k')
        """set color, default is 'black'."""

        self.foreground = self.dict.get('foreground', 'w')
        """set foreground color, default is 'white'."""

        self.pad = self.dict.get('pad', 5)
        """points between the scale bar label and ruler. default is 5."""

        self.zorder = self.dict.get('zorder', zorders['scalebar'])
        """set zorder of the scale bar. Default is 99999995."""

        self.fontsize = self.dict.get('fontsize', 'large')
        """set fontsize, default is 'large'."""

        self.linewidth = self.dict.get('linewidth', 3)
        """set line width of the scale bar ruler. Default is 3."""

    def _loaddata(self):
        self._nplot = 0



    def plot(self, ):
        '''@private
        https://stackoverflow.com/questions/32333870/how-can-i-show-a-km-ruler-on-a-cartopy-matplotlib-plot
        '''
        x, y = self.xy
        x1, y1 = self.xy1


        len_ = (x1 ** 2 + y1 ** 2) ** 0.5
        length1 = self.length * self.unit_factor

        x0 = -x1 * length1 * 0.5 / len_ + x
        x2 =  x1 * length1 * 0.5 / len_ + x

        y0 = -y1 * length1 * 0.5 / len_ + y
        y2 =  y1 * length1 * 0.5 / len_ + y

        buffer = [patheffects.withStroke(linewidth=5, foreground="w")]
        self.ax.plot([x0, x2], [y0, y2], color=self.color, linewidth=self.linewidth, zorder=self.zorder, path_effects=buffer)

        buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
        RotationAwareAnnotation2(str(self.length) + ' ' + str(self.unit), xy=(x, y), p=(x+x1, y+y1),
                                ax=self.ax, xytext=(0, self.pad), textcoords="offset points", fontsize=self.fontsize,
                                va="bottom", ha='center', zorder=self.zorder, path_effects=buffer, )

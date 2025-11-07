
import cartopy.crs as ccrs
from cartopy.io.img_tiles import GoogleTiles, Stamen

from .figure import figure
from ...utils.misc import is_true
from ...utils.plotting import zorders

class mapplot(figure):
    '''

    '''


    def setup_ax(self, fig=None):
        '''
        create map for an axes

        show_border
        epsg

        '''
        ### map ###
        crs = None
        if is_true(self.epsg):
            crs = ccrs.epsg(int(self.epsg))
        elif is_true(self.crs):
            crs = self.crs

        super().setup_ax(fig, crs)

        extent = self.dict.get('extent', getattr(self.parent, 'extent', None))
        assert extent is not None, f'extent is not defined for {self}\n{self.parent.dict}'

        self.ax.set_xlim(extent[:2])
        self.ax.set_ylim(extent[2:])

        if self.dict.get('showgrid', False):
            self.ax.grid(visible=True, )
            pass
        else:
            self.ax.grid(False)
            self.ax.xaxis.set_visible(False)
            self.ax.yaxis.set_visible(False)

    def add_elements(self):
        basemap = self.dict.get('basemap', None)
        if basemap is not None:
            level = self.dict.get('zoomlevel', 13)
            if basemap == 'terrain':
                self.ax.add_image(Stamen('terrain-background'), level, zorder=zorders['basemap'])
            elif basemap == 'street':
                self.ax.add_image(GoogleTiles(style="street"), level, zorder=zorders['basemap'])
            elif basemap == 'satellite':
                self.ax.add_image(GoogleTiles(style="satellite"), level, zorder=zorders['basemap'])

        self.Elements = []

        for this_element in self.plotdata:
            legend = this_element.plot()
            if legend is not None:
                self.legends.update(**legend)
            if is_true(this_element.nplot):
                self.Elements.append(this_element)


    def update_plot(self, ):
        super().update_plot(aspect='equal')

import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse, Rectangle
from matplotlib._tight_bbox import adjust_bbox


zorders = dict(
    basemap    =0,
    scatter    =20,
    scatter_ci =10,
    scatter_med=30,
    scatter_11 =31,
    scatter_reg=32,
    axtitle    =99999999,
    axborder   =99999998,
    legend     =99999997,
    northarrow =99999996,
    scalebar   =99999995,
    labelpoint =99999994,
    statstable =99999990,
)


markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', 'P', '*', 'h', 'H', 'X', 'd']
patch_kwargs = ['alpha', 'color', 'facecolor', 'edgecolor', 'ec', 'fc', 'edgewidth', 'linewidth']
marker_kwargs = ['alpha', 'marker', 'markersize', 'markeredgewidth', 'markeredgecolor', 'markerfacecolor', 'markerfacecoloralt', 'markeredgecolor'] + patch_kwargs
line_kwargs =  ['alpha', 'linewidth', 'linestyle', 'linewidths']
ax_set_kwargs = [
    'adjustable', 'agg_filter', 'alpha', 'anchor', 'animated', 'aspect', 'autoscale_on', 'autoscalex_on', 'autoscaley_on',
    'axes_locator', 'axisbelow', 'box_aspect', 'clip_box', 'clip_on', 'clip_path', 'facecolor', 'frame_on', 'gid',
    'in_layout', 'label', 'mouseover', 'navigate', 'path_effects', 'picker', 'position', 'prop_cycle', 'rasterization_zorder',
    'rasterized', 'sketch_params', 'snap', 'subplotspec', 'title', 'transform', 'url', 'visible',
    'xbound', 'xlabel', 'xlim', 'xmargin', 'xscale', 'xticklabels', 'xticks',
    'ybound', 'ylabel', 'ylim', 'ymargin', 'yscale', 'yticklabels', 'yticks',
     'zorder'
]
collection_prop = ['agg_filter','alpha','animated','antialiased',
    'capstyle','clim','clip_box','clip_on','clip_path',
    'cmap','color','edgecolor','facecolor','gid','hatch',
    'in_layout','joinstyle','label','linestyle','linewidth',
    'mouseover','norm','offset_transform','offsets',
    'path_effects','paths','picker','pickradius',
    'rasterized','sizes','sketch_params','snap',
    'transform','url','urls','visible','zorder'
]

def get_catgorical_cmap(vals, cmap=None, ):
    cats = pd.factorize(vals, sort=True)
    colorarray = cats[0]
    factors = list(cats[1])
    if isinstance(factors[0], float):
        factors = [int(v) if float.is_integer(v) else v for v in factors]

    n = len(factors)
    if n <= 12:
        cmap=cmap or 'Paired'
    elif n <= 20:
        cmap=cmap or 'tab20'
    else:
        cmap=cmap or 'turbo'
    cmap = plt.get_cmap(cmap, n)
    return cmap, colorarray/n, factors

def left_adjust(texts):
    width = 0
    ls = np.asarray([len(t) for t in texts])
    width = (ls).max()
    npads = width - ls
    return [t + ' '*npad for t, npad in zip(texts, npads)]

def get_groupcolors(ngroup, cmap='tab20', igroup=None):
    cmap = plt.get_cmap(cmap, ngroup)
    return cmap.colors if igroup is None else cmap.colors[igroup]

def quickplot(dat, plottype='plot', name='debug.png', plotarg={}):
    """for debug

    Args:
        dat (_type_): _description_
        plottype (str, optional): _description_. Defaults to 'plot'.
        name (str, optional): _description_. Defaults to 'debug.png'.
        plotarg (dict, optional): _description_. Defaults to {}.
    """
    fig, ax = plt.subplots()
    getattr(ax, plottype)(dat, **plotarg)
    fig.savefig(name, bbox_inches='tight', dpi=100)


def resize_fig2(fig, pad_inches=None, ):
    """ from matplotlib.backend_bases.print_figure
        https://github.com/matplotlib/matplotlib/issues/25608
    """

    renderer = fig.canvas.get_renderer()
    bbox_inches = fig.get_tightbbox(renderer, )
    if pad_inches is None:
        pad_inches = rcParams['savefig.pad_inches']
    bbox_inches = bbox_inches.padded(pad_inches)
    restore_bbox = adjust_bbox(fig, bbox_inches, fig.canvas.fixed_dpi)
    return restore_bbox

def resize_fig(fig, left_pad=1, bottom_pad=0.5, top_pad=0.3, right_pad=0.7):

    # get the original figure width and height
    fw, fh = fig.get_size_inches()
    r = fig.canvas.get_renderer()

    bounds = []
    bboxs = []
    for ax in fig.axes:
        # ax.apply_aspect()
        bbox = ax.get_position()
        bounds.append([bbox.x0, bbox.y0, bbox.x1, bbox.y1])
        bboxs.append(bbox)
        for tt in ax.texts:
            bb = tt.get_window_extent(r)
            x0 = bb.x0 / fig.dpi / fw
            x1 = bb.x1 / fig.dpi / fw
            y0 = bb.y0 / fig.dpi / fh
            y1 = bb.y1 / fig.dpi / fh
            bounds.append([x0, y0, x1, y1])
        for tt in [ax.xaxis.label, ax.yaxis.label, ax.title, ]:
            if tt.get_text() == '': continue
            bb = tt.get_window_extent(r)
            x0 = bb.x0 / fig.dpi / fw
            x1 = bb.x1 / fig.dpi / fw
            y0 = bb.y0 / fig.dpi / fh
            y1 = bb.y1 / fig.dpi / fh
            bounds.append([x0, y0, x1, y1])

    bounds = np.array(bounds)
    xmin = bounds[:,0].min()
    ymin = bounds[:,1].min()
    xmax = bounds[:,2].max()
    ymax = bounds[:,3].max()

    xscale = xmax - xmin
    yscale = ymax - ymin

    # calculate the new figure width and height
    fw = fw * xscale + left_pad + right_pad
    fh = fh * yscale + bottom_pad + top_pad

    newx0 = left_pad   / fw
    newy0 = bottom_pad / fh

    xscale = (1 - newx0 - right_pad / fw) / xscale
    yscale = (1 - newy0 - top_pad   / fh) / yscale

    fig.set_size_inches(fw, fh)
    for bbox, ax in zip(bboxs, fig.axes):
        x0 = newx0 + (bbox.x0 - xmin) * xscale
        y0 = newy0 + (bbox.y0 - ymin) * yscale
        w  = bbox.width  * xscale
        h  = bbox.height * yscale
        ax.set_position([x0, y0, w, h])


class welllog(PatchCollection):
    def __init__(self, x=0, top=5, bot=None, scntop=None, scnbot=None, radius=1, ec='k', fc='w', zorder=1, *args, **kwargs):

        if bot is None and scntop is not None and scnbot is not None:
            bot = scnbot

        self.welltop = Ellipse([x, top], radius*2, radius)
        self.casing  = Rectangle([x-radius, bot], radius * 2, top-bot)
        self.wellbot = Ellipse([x, bot], radius*2, radius)
        self.welltop.set_zorder(zorder+0.1)
        self.casing .set_zorder(zorder)
        self.wellbot.set_zorder(zorder)

        self.patches = [self.casing, self.welltop, self.wellbot]
        hatch = '--'
        if 'hatch' in kwargs:
            hatch = kwargs.pop('hatch')
        self.screen = None
        if scntop is not None and scnbot is not None:
            self.screen  = Rectangle([x-radius, scnbot], radius * 2, scntop-scnbot)
            self.screen .set_zorder(zorder+0.1)
            self.screen.set_hatch(hatch)
            self.patches.append(self.screen)

        for p in self.patches:
            p.set_ec(ec)
            p.set_fc(fc)

    def plot(self, ax=None):
        for p in self.patches:
            ax.add_patch(p)

    def set_facecolor(self, color='k'):
        for p in self.patches:
            p.set_fc(color)

    def set_edgecolor(self, color='k'):
        for p in self.patches:
            p.set_ec(color)

    def remove(self):
        for p in self.patches:
            p.remove()

    def set_dimension(self, x=None, top=None, bot=None, scntop=None, scnbot=None, radius=None):

        if bot is None and scnbot is not None:
            bot = scnbot

        x_orig, top_orig = self.welltop.center
        radius_orig      = self.welltop.get_height()
        bot_orig         = self.casing.get_y()

        scn_bot_orig     = self.screen.get_y()
        scn_height_orig     = self.screen.get_height()
        scn_top_orig = scn_bot_orig + scn_height_orig

        x = x or x_orig
        top = top or top_orig
        bot = bot or bot_orig
        scntop = scntop or scn_top_orig
        scnbot = scnbot or scn_bot_orig
        radius = radius or radius_orig

        self.welltop.update(dict(center=[x, top], width=radius*2, height=radius))
        self.casing .update(dict(xy=[x-radius, bot], width=radius*2, height=top-bot))
        self.wellbot.update(dict(center=[x, bot], width=radius*2, height=radius))
        if self.screen is not None:
            self.screen .update(dict(xy=[x-radius, scnbot], width=radius*2, height=scntop-scnbot) )


class RotationAwareAnnotation2(mtext.Annotation):
    '''
    https://stackoverflow.com/questions/19907140/keeps-text-rotated-in-data-coordinate-system-after-resizing
    '''

    def __init__(self, s, xy, p, ax, pa=None, **kwargs):
        self.ax = ax
        self.p = p
        self.pa = pa or xy
        kwargs.update(rotation_mode=kwargs.get("rotation_mode", "anchor"))
        mtext.Annotation.__init__(self, s, xy, **kwargs)
        self.set_transform(mtransforms.IdentityTransform())
        if 'clip_on' in kwargs:
            self.set_clip_path(self.ax.patch)
        self.ax._add_text(self)

    def calc_angle(self):
        p = self.ax.transData.transform_point(self.p)
        pa = self.ax.transData.transform_point(self.pa)
        ang = np.arctan2(p[1]-pa[1], p[0]-pa[0])
        return np.rad2deg(ang)

    def _get_rotation(self):
        return self.calc_angle()

    def _set_rotation(self, rotation):
        pass

    _rotation = property(_get_rotation, _set_rotation)


def add_cax(ax, colorbararg={}):

    if 'location' not in colorbararg:
        if colorbararg.get('orientation') == 'horizontal':
            colorbararg['location'] = 'bottom'
        else:
            colorbararg['location'] = 'right'

    if 'fraction' not in colorbararg:
        colorbararg['fraction'] = 0.03
    if 'shrink' not in colorbararg:
        colorbararg['shrink'] = 1.0
    if 'pad' not in colorbararg:
        colorbararg['pad'] = 0.01

    if colorbararg['location'] in ['right', 'left']: colorbararg['orientation'] = 'vertical'
    if colorbararg['location'] in ['top', 'bottom']: colorbararg['orientation'] = 'horizontal'


    if 'anchor' in colorbararg:
        colorbararg['anchor'] = tuple(colorbararg['anchor'] )
    else:
        if colorbararg['location'] == 'right': colorbararg['anchor'] = (0.0, 0.5)
        if colorbararg['location'] == 'left' : colorbararg['anchor'] = (1.0, 0.5)
        if colorbararg['location'] == 'top'  : colorbararg['anchor'] = (0.5, 0.0)
        if colorbararg['location'] == 'bottom': colorbararg['anchor'] = (0.5, 1.0)

    if 'panchor' in colorbararg:
        colorbararg['panchor'] = tuple(colorbararg['panchor'])
    else:
        if colorbararg['location'] == 'right': colorbararg['panchor'] = (1.0, 0.5)
        if colorbararg['location'] == 'left' : colorbararg['panchor'] = (0.0, 0.5)
        if colorbararg['location'] == 'top'  : colorbararg['panchor'] = (0.5, 1.0)
        if colorbararg['location'] == 'bottom': colorbararg['panchor'] = (0.5, 0.0)

    bbox0 = ax.get_position()

    cax = ax.figure.add_subplot()
    panchor = colorbararg.pop('panchor')
    anchor  = colorbararg.pop('anchor')
    fraction = colorbararg.pop('fraction')
    shrink = colorbararg.pop('shrink')
    location = colorbararg.pop('location')
    pad = colorbararg.pop('pad')
    orientation = colorbararg['orientation']
    if orientation == 'vertical':
        width = fraction * bbox0.width
        height = shrink * bbox0.height
        x0 = bbox0.x0 + panchor[0]*bbox0.width+pad*bbox0.width
        if location =='left':
            x0 -= (2*pad*bbox0.width + anchor[0]*width)
        y0 = bbox0.y0 + panchor[1]*bbox0.height - anchor[1]*height
    else:
        width = shrink * bbox0.width
        height = fraction * bbox0.height
        y0 = bbox0.y0 + panchor[1]*bbox0.height+pad*bbox0.height
        if location =='bottom':
            y0 -= (2*pad*bbox0.height + anchor[1]*height)
        x0 = bbox0.x0 + panchor[0]*bbox0.width - anchor[0]*width

    cax.set_position([x0,y0,width,height])
    return cax

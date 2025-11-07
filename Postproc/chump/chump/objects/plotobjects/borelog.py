
import matplotlib.pyplot as plt
import numpy as np

from ..mobject import mobject


class borelog(mobject):

    # def plot_well_profile(ax=None,
    #                       welltop=None, wellbotm=None, screenelev=[()], layerelev=[], meanwl=None,
    #                       wellcolor='k', screencolor='g', layercolor='r', waterlevelcolor='b', gsecolor='brown',
    #                       label_left=15,
    #                      ):
    #     '''
    #     plot the well screen and model layers
    #     90% water level interval
    #     surface elevation
    #     model surface elevation

    #     Returns
    #     -------
    #     None.

    #     '''
    #     well_center = -0.7
    #     well_width = 0.6

    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=(5, 10))

    #     # plot the well line
    #     ax.plot([well_center, well_center], [welltop, wellbotm], color=wellcolor)

    #     ax.plot(well_center, welltop, marker='_', markersize=20, color=gsecolor)
    #     ax.annotate(f'GSE={welltop:.1f}', xy=(well_center, welltop), xycoords='data', xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize='small')

    #     # plot screen
    #     for scn in screenelev:

    #         if not scn:
    #             break

    #         ax.bar(well_center, scn[0]-scn[1], width=well_width, bottom=scn[1], color=screencolor, alpha=0.7)#, hatch='--')
    #         ax.annotate(f'Scn={scn[0]:.1f}\n      ~{scn[1]:.1f}', xy=(well_center, (scn[0]+scn[1])*0.5),
    #                     xycoords='data', xytext=(label_left, 0), textcoords='offset points', ha='left', va='center', fontsize='small')

    #     # plot layers
    #     for layer in layerelev:
    #         ax.plot(well_center, layer, marker='o', markersize=3, color=layercolor, alpha=0.5)

    #     if meanwl:
    #         ax.plot(well_center, meanwl, marker=7, markersize=5, color='b', alpha=0.9)
    #         ax.plot(well_center, meanwl, marker="_", markersize=5, color='b', alpha=0.9)


    #     ax.annotate(f'Aquif. Botm\n   {layer:.1f}', xy=(well_center, layer),
    #                 xycoords='data', xytext=(0, -5), textcoords='offset points', ha='center', va='top', fontsize='small')


    #     ax.set_yticks(layerelev)
    #     ax.set_xlim(-1, 1)
    #     ax.xaxis.set_ticklabels([])
    #     ax.yaxis.set_ticklabels([])


    def initialize_plot(self, fig=None,
        well_center=0.2, well_width=0.6,
        wellcolor='k', layercolor='r', screencolor='g'):
        '''
        create map for an axes
        '''
        ### map ###

        if fig is None:
            figsize = self.dict.get('figsize', (8, 6))
            self.fig = plt.figure(figsize=figsize, ) # subplotpars={'wspace':0,'hspace':0.}
            self.fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        else:
            self.fig = fig

        self.ax = self.fig.add_subplot()

        self.ax.set_xlim(-1, 1)
        self.ax.xaxis.set_ticklabels([])
        self.ax.yaxis.set_ticklabels([])


        self.wellcenter = well_center
        self.welltop = 0
        self.wellbotm = 0

        self.wellline = self.ax.plot([self.wellcenter, self.wellcenter], [self.wellbotm, self.welltop], color=wellcolor)[0]
        self.topline = self.ax.plot(self.wellcenter, self.welltop,  marker='_', markersize=10, color=wellcolor)[0]
        self.botline = self.ax.plot(self.wellcenter, self.wellbotm, marker='_', markersize=10, color=wellcolor)[0]

        # plot layers
        self.layerelev = [0, 1]
        # self.layerdivide = self.ax.plot([self.wellcenter]*len(self.layerelev), self.layerelev, linewidth=0.,
        #                                 marker='o', markersize=3, color=layercolor, alpha=0.5)[0]

        self.ax.set_xticks([])
        mf = self._getobj('model')

        self.layertext = [
            self.ax.text(-0.5, -9000, str(i+1), fontsize='xx-small', ha='center', va='center') for i in range(mf.nlay)
        ]
        self.layerk = [
            self.ax.bar(0., i)[0] for i in range(mf.nlay)
        ]

        if 'kxlevels' in self.dict:
            self.unique_k = np.sort(np.unique(self.dict['kxlevels']))
            ncolor = len(self.unique_k)
        else:
            self.unique_k = np.sort(np.unique(mf.hk))
            ncolor = len(self.unique_k) - 1
        self.cmap_k = plt.get_cmap(self.dict.get('cmap','jet'), ncolor)
        # cmap(np.searchsorted(unique_k, 50)/ncolor, 0.5)
        # f = self.ax.fill_between([0, 1], 0, 1, color='green', alpha=0.5, transform= self.ax.get_yaxis_transform())

        self.ax.yaxis.set_tick_params(width=0.1, length=2)
        # plot screen
        self.screens = []
        self.screenelev = []
        # for scn in self.screenelev:

        #     if not scn:
        #         break

        #     self.screens.append(
        #         self.ax.bar(self.wellcenter, scn[0]-scn[1], width=well_width, bottom=scn[1], color=screencolor, alpha=0.7)#, hatch='--')
        #     )
        #     # ax.annotate(f'Scn={scn[0]:.1f}\n      ~{scn[1]:.1f}', xy=(well_center, (scn[0]+scn[1])*0.5),
        #     #             xycoords='data', xytext=(label_left, 0), textcoords='offset points', ha='left', va='center', fontsize='small')

    def update_plot(self, well, hasData=True):
        # print(well)
        mf = self._getobj('model')
        irow, icol = int(well.row - 1), int(well.column - 1)

        self.layerelev = np.concatenate([[mf.top[irow, icol]], mf.botm[:, irow, icol]])
        self.ax.set_yticks(self.layerelev[::-1])

        self.welltop = well.top if 'top' in well.index else well.scntop
        self.wellbotm = well.botm if 'botm' in well.index else well.scnbot
        self.wellline.set_data([self.wellcenter, self.wellcenter], [self.wellbotm, self.welltop])
        self.topline.set_data(self.wellcenter, self.welltop)
        self.botline.set_data(self.wellcenter, self.wellbotm)
        # self.layerdivide.set_data([self.wellcenter]*len(self.layerelev), self.layerelev)

        for i, t in enumerate(self.layertext):
            t.set_position((-0.5, 0.5*(self.layerelev[i]+self.layerelev[i+1])))

        for i, k in enumerate(self.layerk):
            kk = mf.hk[i, irow, icol]
            c = self.cmap_k(np.searchsorted(self.unique_k, kk)/self.cmap_k.N, alpha=self.dict.get('alpha', 1.0))
            k.set_color(c)
            k.set_bounds(-1, self.layerelev[i+1], 2, self.layerelev[i]-self.layerelev[i+1])

        dy = (self.layerelev[0] - self.layerelev[-1]) * 0.01
        self.ax.set_ylim(min(self.layerelev[-1] - dy, self.wellbotm), max(self.layerelev[0] + dy, self.welltop))

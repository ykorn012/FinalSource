import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt1
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd

class FDC_Graph:

    def plt_show1(self, n, y_act, y_prd, subtitle, q_param):
        plt.figure()
        plt.suptitle(subtitle, fontsize=15)
        plt.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.91)
        plt.plot(np.arange(n), y_act, 'rx--', label='$y_{act}^{' + q_param + '}$', lw=2, ms=5, mew=2)
        plt.plot(np.arange(n), y_prd, 'bx--', label='$y_{pred}^{' + q_param + '}$', lw=2, ms=5, mew=2)
        plt.legend(loc='upper left', fontsize='large')
        plt.xticks(np.arange(0, n + 1, 50))
        plt.xlabel('Run No.')
        plt.ylabel('Actual and Predicted Response')
        plt.show()

    def plt_show2(self, n, y1, y2, Noise):
        plt.figure()
        plt.plot(np.arange(0, n + 1, 1), y1, 'bx-', y2, 'gx--', lw=2, ms=5, mew=2)
        plt.xticks(np.arange(0, n + 1, 5))
        if Noise == False:
            plt.yticks(np.arange(-1.2, 1.3, 0.2))
        else:
            plt.yticks(np.arange(-5.5, 6, 0.5))
        plt.xlabel('Metrology Run No.(z)')
        plt.ylabel('e(z)')
        plt.tight_layout()

    def plt_show3(self, n, y1, y2):
        plt.figure()
        plt.plot(np.arange(n), y1, 'bx-', y2, 'gx--', lw=2, ms=5, mew=2)
        plt.xticks(np.arange(0, n + 1, 5))
        plt.yticks(np.arange(-12, 3, 2))
        plt.xlabel('Metrology Run No.(z)')
        plt.ylabel('e(z)')

    def plt_show4(self, n, y1):
        plt.figure()
        plt.plot(np.arange(n), y1, 'rx-', lw=2, ms=5, mew=2)
        plt.xticks(np.arange(0, n + 1, 5))
        plt.yticks(np.arange(-1.2, 1.3, 0.2))
        plt.xlabel('Metrology Run No.(z)')
        plt.ylabel('e(z)')

    def plt_show5(self, ez_run, N, M, dM, S1, Noise):
        df = pd.DataFrame(ez_run, columns=['q1', 'q2'])
        label = []
        for i in np.arange(0, N + 1, 1):
            if i <= S1 * M:
                label.append(0)
            else:
                label.append(1)
        df['label'] = pd.Series(label)
        # df.loc[251]['label']

        xdata = []
        y1data = []
        y2data = []
        ldata = []

        plt.figure()

        for i in np.arange(0, N + 1, 1):
            if i < S1 * M and i % M == 0:
                xdata.append(i)
                y1data.append(df.loc[i]['q1'])
                y2data.append(df.loc[i]['q2'])
                ldata.append(0)
                # line1.set_xdata(xdata)
                # line1.set_ydata(y1data)
                # line2.set_xdata(xdata)
                # line2.set_ydata(y2data)
            if i >= S1 * M and i % dM == 0:
                xdata.append(i)
                y1data.append(df.loc[i]['q1'])
                y2data.append(df.loc[i]['q2'])
                ldata.append(1)
                # line1.set_xdata(xdata)
                # line1.set_ydata(y1data)
                # line2.set_xdata(xdata)
                # line2.set_ydata(y2data)

        # line1.set_xdata(xdata)
        # line1.set_ydata(y1data)
        # line2.set_xdata(xdata)
        # line2.set_ydata(y2data)

        # plt.show()

        df2 = pd.DataFrame(np.array([xdata, y1data, y2data, ldata]))
        df2 = df2.T
        df2.columns = ['no', 'q1', 'q2', 'label']

        num_classes = 2
        # cmap = ListedColormap(['r', 'g', 'b', 'y'])
        cmap = ListedColormap(['b', 'r'])
        norm = BoundaryNorm(range(num_classes + 1), cmap.N)
        points = np.array([df2['no'], df2['q1']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(df2['label'])

        # fig1 = plt.figure()

        plt.gca().add_collection(lc)
        # plt.xlim(df.index.min(), df.index.max())
        # plt.ylim(-1.1, 1.1)
        plt.xlabel('Metrology Run No.(z)')
        plt.ylabel('e(z)')
        # plt.xticks(np.arange(0, 410, 50))
        ticks = np.arange(0, N + 1, 50)
        plt.xticks(ticks)
        if Noise == False:
            plt.yticks(np.arange(-1.2, 1.3, 0.2))
        else:
            plt.yticks(np.arange(-10.5, 7.5, 0.5))

        dic = {50: "50 \n (5 runs)", 100: "100 \n (10 runs)", 150: "150 \n (15 runs)", 200: "200 \n (20 runs)",
               250: "250 \n (30 runs)", 300: "300 \n (40 runs)", 350: "350 \n (50 runs)", 400: "400 \n (60 runs)"}
        labels = [ticks[i] if t not in dic.keys() else dic[t] for i, t in enumerate(ticks)]

        axes = plt.gca()
        # axes.set_xlim(0, N)
        # axes.set_ylim(-1.2, 1.2)
        # line1, = axes.plot(xdata, y1data, 'b', lw=2, ms=5, mew=2, linestyle='--')
        # line2, = axes.plot(xdata, y2data, 'g', lw=2, ms=5, mew=2, linestyle='--')

        i = 0
        for text in axes.get_xticklabels():
            if i >= 4:
                text.set_color("red")
            i = i + 1

        # ax = fig1.add_subplot(111)
        axes.set_xticklabels(labels)
        # axes.set_color_cycle(colors)
        #plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['lines.markersize'] = 5
        plt.rcParams['lines.markeredgewidth'] = 2
        #plt.rcParams['lines.linestyle'] = 'x--'
        plt.tight_layout()
        #plt.rcParams.update({'font.size': 20, 'lines.linewidth': 3, 'lines.markersize': 15})
        plt.show()

    def plt_show5_1(self, noise_ez_run, ds_ez_run, N, M, dM, S1, type):
        fig = plt.figure()
        if type == 1:
            plt.suptitle('Process-1 Abnomal vs Dynamic Sampling (60 Metrology Runs)', fontsize=15)
        else:
            plt.suptitle('Process-1 Abnomal vs Dynamic Sampling (100 Metrology Runs)', fontsize=15)

        df = pd.DataFrame(noise_ez_run, columns=['q1', 'q2'])
        xdata = []
        y1data = []
        y2data = []
        for i in np.arange(0, N + 1, 1):
            if i % M == 0:
                xdata.append(i)
                y1data.append(df.loc[i]['q1'])
                y2data.append(df.loc[i]['q2'])
        df2 = pd.DataFrame(np.array([xdata, y1data, y2data]))
        df2 = df2.T
        df2.columns = ['no', 'q1', 'q2']
        points = np.array([df2['no'], df2['q1']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, linestyle='--', linewidths=2, colors='b', label='Abnormal $y^{1}$')
        plt.gca().add_collection(lc)

        for i in np.arange(0, N + 1, 1):
            if i % M == 0:
                plt.plot(i, df.loc[i]['q1'], 'bx--')

        df = pd.DataFrame(ds_ez_run, columns=['q1', 'q2'])
        label = []
        for i in np.arange(0, N + 1, 1):
            if i <= S1 * M:
                label.append(0)
            else:
                label.append(1)
        df['label'] = pd.Series(label)
        # df.loc[251]['label']

        xdata = []
        y1data = []
        y2data = []
        ldata = []

        for i in np.arange(0, N + 1, 1):
            if i < S1 * M and i % M == 0:
                xdata.append(i)
                y1data.append(df.loc[i]['q1'])
                y2data.append(df.loc[i]['q2'])
                ldata.append(0)

            if i >= S1 * M and i % dM == 0:
                xdata.append(i)
                y1data.append(df.loc[i]['q1'])
                y2data.append(df.loc[i]['q2'])
                ldata.append(1)

        df2 = pd.DataFrame(np.array([xdata, y1data, y2data, ldata]))
        df2 = df2.T
        df2.columns = ['no', 'q1', 'q2', 'label']

        num_classes = 2
        # cmap = ListedColormap(['r', 'g', 'b', 'y'])
        cmap = ListedColormap(['b', 'r'])
        norm = BoundaryNorm(range(num_classes + 1), cmap.N)
        points = np.array([df2['no'], df2['q1']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linestyle='-', linewidths=2, label='Dynamic Sampling $y^{1}$', color='r')
        lc.set_array(df2['label'])
        plt.gca().add_collection(lc)

        for i in np.arange(0, N + 1, 1):
            if i >= S1 * M and i % dM == 0:
                plt.plot(i, df.loc[i]['q1'], 'ro--')

        # plt.xlim(df.index.min(), df.index.max())
        # plt.ylim(-1.1, 1.1)
        plt.xlabel('Metrology Run No.(z)')
        plt.ylabel(r'Prediction Error $(y_{z} - \hat y_{z})$')
        # plt.xticks(np.arange(0, 410, 50))
        ticks = np.arange(0, N + 1, 50)
        plt.xticks(ticks)
        plt.yticks(np.arange(-10.5, 7.5, 2))

        if type == 1:
            dic = {50: "50 \n (5 )", 100: "100 \n (10 runs)", 150: "150 \n (15 runs)", 200: "200 \n (20 runs)",
                   250: "250 \n (30 runs)", 300: "300 \n (40 runs)", 350: "350 \n (50 runs)", 400: "400 \n (60 runs)"}
        else:
            dic = {50: "50 \n (5 runs)", 100: "100 \n (10 runs)", 150: "150 \n (15 runs)", 200: "200 \n (20 runs)",
                   250: "250 \n (30 runs)", 300: "300 \n (40 runs)", 350: "350 \n (50 runs)", 400: "400 \n (60 runs)",
                   450: "450 \n (70 runs)", 500: "500 \n (80 runs)", 550: "550 \n (90 runs)", 600: "600 \n (100 runs)"}
        labels = [ticks[i] if t not in dic.keys() else dic[t] for i, t in enumerate(ticks)]

        axes = plt.gca()

        i = 0
        for text in axes.get_xticklabels():
            if i >= 4:
                text.set_color("red")
            i = i + 1

        axes.set_xticklabels(labels)
        plt.legend(loc='upper left', fontsize='large')
        plt.axvline(x=150, color='g', linestyle=':', linewidth=5)
        plt.axvline(x=200, color='r', linestyle=':', linewidth=5)
        plt.text(90, -8, 'Noise Start', fontsize=10, bbox=dict(facecolor='g', alpha=0.5))
        plt.text(210, -8, 'Noise Sensing', fontsize=10, bbox=dict(facecolor='r', alpha=0.5))

        #plt.tight_layout()
        plt.show()

    def mean_absolute_percentage_error(self, z, y_act, y_prd):
        #print('z: ', z, 'y_act : ', y_act, 'y_prd : ', y_prd)
        mape = np.mean(np.abs((y_act - y_prd) / y_act)) * 100
        #print('mape : ', mape)
        return mape

    def plt_show2_2(self, n, y1, y2, subtitle, q_param, color1='bx-', color2='rx--'):
        plt.figure()
        plt.suptitle(subtitle, fontsize=15)
        plt.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.91)
        plt.plot(np.arange(n), y1, color1, label='Process-1 Abnormal ($y^{' + q_param + '}$)', lw=2, ms=5, mew=2)
        plt.plot(np.arange(n), y2, color2, label='Process-1 Dynamic Sampling ($y^{' + q_param + '}$)', lw=2, ms=5, mew=2)
        plt.legend(loc='upper left', fontsize='large')
        plt.xticks(np.arange(0, n + 1, 50))
        plt.yticks(np.arange(-23, 15, 5))
        plt.axvline(x=200, color='y', linestyle=':', linewidth=5)
        plt.xlabel('Run No.')
        plt.ylabel(r'Prediction Error $(y - \hat y)$')

    def plt_show2_3(self, n, y1, y2, y3, subtitle, q_param, color1='bx-', color2='rx--', color3='gx--'):
        plt.figure()
        plt.suptitle(subtitle, fontsize=15)
        plt.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.91)
        plt.plot(np.arange(n), y3, color3, label='Process-1 Normal ($y^{' + q_param + '}$)', lw=2, ms=5,mew=2)
        plt.plot(np.arange(n), y1, color1, label='Process-1 Abnormal ($y^{' + q_param + '}$)', lw=2, ms=5, mew=2)
        plt.plot(np.arange(n), y2, color2, label='Process-1 Dynamic Sampling ($y^{' + q_param + '}$)', lw=2, ms=5, mew=2)
        plt.legend(loc='upper left', fontsize='large')
        plt.xticks(np.arange(0, n + 1, 50))
        plt.yticks(np.arange(-23, 15, 5))
        plt.axvline(x=200, color='y', linestyle=':', linewidth=5)
        plt.xlabel('Run No.')
        plt.ylabel(r'Prediction Error $(y - \hat y)$')

    def plt_show2_4(self, n, y1, y2, subtitle, q_param, color1='bx-', color2='rx--'):
        plt.figure()
        plt.suptitle(subtitle, fontsize=15)
        plt.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.91)
        plt.plot(np.arange(n), y1, color1, label='Process-1 UpStream Rule ($y^{' + q_param + '}$)', lw=2, ms=5, mew=2)
        plt.plot(np.arange(n), y2, color2, label='Process-1 Dynamic Sampling + UpStream Rule ($y^{' + q_param + '}$)', lw=2, ms=5, mew=2)
        plt.legend(loc='upper left', fontsize='large')
        plt.xticks(np.arange(0, n + 1, 50))
        plt.yticks(np.arange(-23, 15, 5))
        plt.axvline(x=200, color='y', linestyle=':', linewidth=5)
        plt.xlabel('Run No.')
        plt.ylabel(r'Prediction Error $(y - \hat y)$')

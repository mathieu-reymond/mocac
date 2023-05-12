import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import moviepy.editor as mpy
import torch


def plt_to_rgba(plot_maker, *maker_args):
    fig = plt.figure()
    fig.tight_layout(pad=0)
    # To remove the huge white borders
    plt.gca().margins(0)
    canvas = FigureCanvasAgg(fig)
    plot_maker(*maker_args)
    canvas.draw()
    buf = canvas.buffer_rgba()
    rgba = np.asarray(buf)
    plt.close(fig)
    return rgba

def moving_avg(x,n):
    mv =  np.convolve(x,np.ones(n)/n,mode='valid')
    return np.concatenate(([np.NaN for k in range(n-1)],mv))


class Plotter(object):

    def __init__(self, logdir):
        self.logdir = logdir

        self.make_plots()

    def make_avg_plots(self):
        for i, path in enumerate(Path(self.logdir).rglob('log.h5')):
            print('making plots for ' + str(path))
            plotdir = path.parents[0] / 'plots'
            plotdir.mkdir(parents=True, exist_ok=True)

    def make_plots(self):
        for i, path in enumerate(Path(self.logdir).rglob('log.h5')):
            print('making plots for ' + str(path))
            plotdir = path.parents[0] / 'plots'
            plotdir.mkdir(parents=True, exist_ok=True)
            try:
                with h5py.File(path, 'r') as logfile:
                    for p, item, type_ in h5py_dataset_iterator(logfile):
                        if type_ == 'scalar':
                            self.plot_scalar(item, plotdir)
                        # elif type_ == 'histogram':
                        #     self.plot_histogram(item, plotdir)
            except:
                print('could not process file')

    def plot_scalar(self, item, plotdir):

        plt.figure()
        plt.title(item.name.replace('/', '_'))
        plt.plot(item[:,0], moving_avg(item[:,1], np.minimum(100,item[:,1].shape[0])))
        name =  item.name.replace('/', '_') + '.png'
        plt.savefig(plotdir / name)

    def plot_histogram(self, item, plotdir):

        def hist1d_maker(item, i):
            plt.title(item.name.replace('/', '_') + f'_{i}')
            plt.bar(np.arange(item['ndarray'].shape[-1]), item['ndarray'][i])
            plt.ylim(0., 1.)

        def hist2d_maker(item, i):
            plt.title(item.name.replace('/', '_') + f'_{i}')
            hist = item['ndarray'][i]
            x, y = np.meshgrid(*[np.arange(i) for i in hist.shape])
            x, y = x.ravel(), y.ravel()

            width = depth = 1
            bottom = np.zeros_like(x)
            ax = plt.gcf().add_subplot(111, projection='3d')
            ax.bar3d(x, y, bottom, width, depth, hist.flatten(), shade=True, color='m')
            ax.set_zlim(0., 1.)

        nd = item['ndarray'].shape
        makers = {2: hist1d_maker, 3: hist2d_maker}
        hist_maker = makers[len(nd)]

        unique, indices, counts = np.unique(item['step'], return_index=True, return_counts=True)

        for vid in range(len(unique)):
            frames_i = np.arange(indices[vid], indices[vid]+counts[vid])
            frames = [plt_to_rgba(hist_maker, item, i) for i in frames_i]
            clip = mpy.ImageSequenceClip(frames, fps=24)
            name = item.name.replace('/', '_') + f'_{unique[vid]}.gif'
            clip.write_gif(plotdir / name, program='ffmpeg')
        
        # start-state over time
        frames = [plt_to_rgba(hist_maker, item, i) for i in indices]
        clip = mpy.ImageSequenceClip(frames, fps=24)
        name = item.name.replace('/', '_') + '_s_0.gif'
        clip.write_gif(plotdir / name, program='ffmpeg')


def h5py_dataset_iterator(g, prefix=''):
    for key in g.keys():
        print(key)
        item = g[key]
        path = '{}/{}'.format(prefix, key)
        if 'type' in item.attrs:
            yield (path, item, item.attrs['type'])
        else:
            yield from h5py_dataset_iterator(item, path)
        # if isinstance(item, h5py.Dataset): # test for dataset
        #     yield (path, item)
        # elif isinstance(item, h5py.Group): # test for group (go down)
        #     yield from h5py_dataset_iterator(item, path)


if __name__ == '__main__':
    import sys
    plotter = Plotter(sys.argv[1])

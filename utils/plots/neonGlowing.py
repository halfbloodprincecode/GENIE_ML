import numpy as np
import matplotlib.pyplot as plt

class Neon:
    def __init__(self, xlabel='x', ylabel='y', color='#D9D9D9', font='Purisa', title_fontdict=None, labels_fontdict=None, grid_args_dict=None):
        self.title_fontdict = title_fontdict if title_fontdict else dict(family=[font, 'Purisa', 'sans-serif', 'serif'], color=color, weight='ultralight')
        self.labels_fontdict = labels_fontdict if labels_fontdict else dict(family=[font, 'Purisa', 'sans-serif', 'serif'], color=color, weight='bold', fontsize='x-large')
        self.grid_args_dict = grid_args_dict if grid_args_dict else dict(zorder=0.5, alpha=.02, color=color)

        style = 'https://raw.githubusercontent.com/halfbloodprincecode/cache/master/styles/py/neon.mplstyle'
        plt.style.use(style)
        self.fig = plt.figure(figsize=(6, 4), facecolor=None)
        # self.fig.patch.set_alpha(.08)
        # plt.suptitle('suptitle', fontdict=self.title_fontdict)
        # plt.title('title', fontdict=self.title_fontdict)
        plt.xlabel(xlabel, fontdict=self.labels_fontdict)
        plt.ylabel(ylabel, fontdict=self.labels_fontdict)
        plt.grid(**self.grid_args_dict)
        plt.xticks(fontname=font)
        plt.yticks(fontname=font)
    
    def savefig(self, path, dpi=1200, bbox_inches='tight', **kwargs):
        try:
            from libs.basicIO import pathBIO
        except Exception as e:
            pathBIO = lambda x: x
        return self.fig.savefig(pathBIO(path), dpi=dpi, bbox_inches=bbox_inches, **kwargs)

    def plot(self, x, y, ax=None, label=None):
        if ax is None:
            ax = self.fig.gca() # ax = plt.gca()
        # ax.patch.set_facecolor('#3498db')
        # ax.patch.set_alpha(.08)
        # y_ticks=['y tick 1','y tick 2','y tick 3']
        # ax.set_yticklabels(y_ticks, rotation=0, fontsize=8)
        line, = ax.plot(x, y, lw=1, zorder=6, label=label)
        for cont in range(6, 1, -1):
            ax.plot(x, y, lw=cont, color=line.get_color(), zorder=5, alpha=0.05)
        ax.legend()
        return ax     
        

if __name__ == '__main__':
    neon = Neon(xlabel='x1', ylabel='y1')
    neon2 = Neon(xlabel='x2', ylabel='y2')
    x = np.linspace(0, 4, 100)
    y = np.sin(np.pi*x + 1e-6)/(np.pi*x + 1e-6)
    for cont in range(5):
        neon.plot(x, y/(cont + 1), label=f'f({cont})')
        neon2.plot(x, -y/(cont + 1))
    # neon.savefig('./neon_example1200.png')
    plt.show()
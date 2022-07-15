from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
import matplotlib.pyplot as plt
import fuzzytools.matplotlib.plots as cplots
import fuzzytools.matplotlib.colors as cc
from fuzzytools.datascience.statistics import dropout_extreme_percentiles
import pandas as pd

###################################################################################################################################################

def plot_class_distribution(lcdataset, lcset_names,
    figsize=None,
    uses_log_scale=True,
    caption=None,
    ):
    lcset = lcdataset[lcset_names[0]]
    lcobj_classes = lcset.get_lcobj_classes()
    pop_dict = {lcset_name:lcdataset[lcset_name].get_lcobj_classes() for lcset_name in lcset_names}
    title = ''
    title += 'SNe class distribution'+'\n'
    title += f'survey={lcset.survey}-{"".join(lcset.band_names)}'+'\n'
    fig, ax = cplots.plot_hist_labels(pop_dict, lcset.class_names,
        title=title[:-1],
        uses_log_scale=uses_log_scale,
        figsize=figsize,
        )

    fig.tight_layout()
    fig.text(.1,.1, caption)
    plt.plot()

def plot_sigma_distribution(lcdataset, set_name:str,
    figsize:tuple=(15,10),
    ):
    attr = 'obse'
    return plot_values_distribution(lcdataset, set_name, attr,
        figsize,
        )

def plot_values_distribution(lcdataset, set_name:str, attr:str,
    title='?',
    xlabel='?',
    p=0.5,
    figsize:tuple=(15,10),
    f=lambda x:x,
    ):
    lcset = lcdataset[set_name]
    fig, axes = plt.subplots(len(lcset.class_names), len(lcset.band_names), figsize=figsize)
    for kb,b in enumerate(lcset.band_names):
        for kc,c in enumerate(lcset.class_names):
            ax = axes[kc,kb]
            plot_dict = {c:dropout_extreme_percentiles(f(lcset.get_all_values_b(b, attr, c)), p, mode='upper')[0]}
            plot_df = pd.DataFrame.from_dict(plot_dict, orient='columns')
            fig, ax = cplots.plot_hist_bins(plot_df,
                fig=fig,
                ax=ax,
                xlabel=xlabel if kc==len(lcset.class_names)-1 else None,
                ylabel='' if kb==0 else None,
                title=f'band={b}' if kc==0 else '',
                uses_density=True,
                legend_loc='upper right',
                cmap=cc.colorlist2cmap(cc.get_default_colorlist()[kc:])
                )

            ### multi-band colors
            ax.grid(alpha=0)
            [ax.spines[border].set_color(_C.COLOR_DICT[b]) for border in ['bottom', 'top', 'right', 'left']]
            [ax.spines[border].set_linewidth(2) for border in ['bottom', 'top', 'right', 'left']]
    
    fig.suptitle(title, va='bottom', y=.99)#, fontsize=14)
    fig.tight_layout()
    plt.show()
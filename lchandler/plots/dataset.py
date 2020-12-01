from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import matplotlib.pyplot as plt
import flamingchoripan.cuteplots.plots as cplots
import flamingchoripan.cuteplots.colors as cc
import pandas as pd

###################################################################################################################################################

def plot_class_distribution(lcdataset, lcset_names,
	figsize=None,
	uses_log_scale=1,
	):
	#for ks,lcset_name in enumerate([lcset_name1, lcset_name2]):
	#ax = axs[ks]
	lcset = lcdataset[lcset_names[0]]
	lcobj_classes = lcset.get_lcobj_classes()
	pop_dict = {lcset_name:lcdataset[lcset_name].get_lcobj_classes() for lcset_name in lcset_names}
	title = 'class population distributions\n'
	title += f'survey: {lcset.survey}'
	plt_kwargs = {
		#'ylabel':'' if ks>0 else None,
		'title':title,
		#'cmap':cc.colorlist_to_cmap([cc.NICE_COLORS_DICT['nice_gray']]),
		'uses_log_scale':uses_log_scale,
		'figsize':figsize,
	}
	fig, ax = cplots.plot_hist_labels(pop_dict, lcset.class_names, **plt_kwargs)
		
	fig.tight_layout()
	plt.plot()

def plot_sigma_distribution(lcdataset, set_name:str,
	figsize:tuple=(15,10),
	):
	attr = 'obse'
	return plot_values_distribution(lcdataset, set_name, attr,
		figsize,
		)

def plot_values_distribution(lcdataset, set_name:str, attr:str,
	figsize:tuple=(15,10),
	):
	lcset = lcdataset[set_name]
	fig, axes = plt.subplots(len(lcset.class_names), len(lcset.band_names), figsize=figsize)
	for kb,b in enumerate(lcset.band_names):
		for kc,c in enumerate(lcset.class_names):
			ax = axes[kc,kb]
			plot_dict = {c:lcset.get_lcset_values_b(b, attr, c)*100}
			plot_df = pd.DataFrame.from_dict(plot_dict, orient='columns')
			title = f'{C_.LONG_NAME_DICT[attr]} distribution {C_.SYMBOLS_DICT[attr]}\n'
			title += f'survey: {lcset.survey} - set: {set_name.replace("_", "-")} - band: {b}'
			kwargs = {
				'fig':fig,
				'ax':ax,
				'xlabel':C_.XLABEL_DICT[attr] if kc==len(lcset.class_names)-1 else None,
				'ylabel':'' if kb==0 else None,
				'title':title if kc==0 else '',
				'xlim':(None if c=='SLSN' else (0, 150)) if attr=='obs' else None,
				'bins':100 if c=='SLSN' else 500,
				'uses_density':True,
				'legend_loc':'upper right',
				'cmap':cc.get_cmap(cc.get_default_colorlist()[kc:])
			}
			fig, ax = cplots.plot_hist_bins(plot_df, **kwargs)

			### multiband colors
			ax.grid(color=C_.COLOR_DICT[b])
			[ax.spines[border].set_color(C_.COLOR_DICT[b]) for border in ['bottom', 'top', 'right', 'left']]
			[ax.spines[border].set_linewidth(2) for border in ['bottom', 'top', 'right', 'left']]
		
	fig.tight_layout()
	plt.show()
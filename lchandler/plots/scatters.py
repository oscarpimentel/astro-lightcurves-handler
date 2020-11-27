from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import matplotlib.pyplot as plt
from flamingchoripan.cuteplots import colors as cc

###################################################################################################################################################

def plot_obs_obse_scatter(lcdataset, set_names,
	plot_ndict=None,
	figsize:tuple=(12,8),
	alpha=0.7,
	markersize=1.2,
	):
	fig, axs = plt.subplots(1, 2, figsize=figsize)
	band_names = lcdataset[set_names[0]].band_names
	#cmap = cc.get_default_cmap(len(set_names))
	cmap = cc.colorlist_to_cmap(['k']+cc.COLORS_DICT['cc_favs2'])
	plot_ndict = {pnk:None for pnk in plot_ndict.keys()} if plot_ndict is None else plot_ndict
	for kb,b in enumerate(band_names):
		ax = axs[kb]
		for k,set_name in enumerate(set_names):
			lcset = lcdataset[set_name]
			c = cmap.colors[k]

			obse = lcset.get_lcset_values_b(b, 'obse')
			obs = lcset.get_lcset_values_b(b, 'obs')
			if not plot_ndict[set_name] is None:
				idxs = np.random.permutation(np.arange(0, len(obse)))[:int(plot_ndict[set_name])]
				obse = obse[idxs]
				obs = obs[idxs]
			label = '$p(x_{ij},\sigma_{xij})$'+f' {set_name} samples'
			ax.plot(obse, obs, '.', c=c, markersize=markersize, alpha=alpha); ax.plot(np.nan, np.nan, '.', c=c, alpha=1, label=label)

			title = f'observations & observations-error joint distribution\n'
			title += f'survey: {lcset.survey} - band: {b}'
			ax.set_title(title)
			ax.set_xlabel('obs-error')
			ax.set_ylabel('obs' if kb==0 else None)
			ax.legend()
			#ax.set_xlim([0.0025, 0.02])
			ax.set_ylim([0, 0.4])

			### multiband colors
			#ax.grid(color=C_.COLOR_DICT[b])
			[ax.spines[border].set_color(C_.COLOR_DICT[b]) for border in ['bottom', 'top', 'right', 'left']]
			[ax.spines[border].set_linewidth(2) for border in ['bottom', 'top', 'right', 'left']]

	fig.tight_layout()
	plt.show()
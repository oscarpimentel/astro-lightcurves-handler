from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import matplotlib.pyplot as plt
from flamingchoripan.cuteplots import colors as cc

###################################################################################################################################################

def plot_obs_obse_scatter(lcdataset, lcset_names,
	plot_ndict=None,
	figsize:tuple=(12,8),
	alpha=0.7,
	markersize=1.2,
	):
	fig, axs = plt.subplots(1, 2, figsize=figsize)
	band_names = lcdataset[lcset_names[0]].band_names
	#cmap = cc.get_default_cmap(len(lcset_names))
	cmap = cc.colorlist_to_cmap(['k']+cc.COLORS_DICT['cc_favs2'])
	plot_ndict = {lcset_name:None for lcset_name in lcset_names} if plot_ndict is None else plot_ndict
	for kb,b in enumerate(band_names):
		ax = axs[kb]
		for k,lcset_name in enumerate(lcset_names):
			lcset = lcdataset[lcset_name]
			c = cmap.colors[k]

			obse = lcset.get_lcset_values_b(b, 'obse')
			obs = lcset.get_lcset_values_b(b, 'obs')
			if not plot_ndict[lcset_name] is None:
				idxs = np.random.permutation(np.arange(0, len(obse)))[:int(plot_ndict[lcset_name])]
				obse = obse[idxs]
				obs = obs[idxs]

			is_synthetic = '.' in lcset_name
			label = 'p(obs,obs-error) '+('[synth]' if is_synthetic else '[real]')
			ax.plot(obse, obs, '.', c=c, markersize=markersize, alpha=alpha); ax.plot(np.nan, np.nan, '.', c=c, alpha=1, label=label)

			ax.set_title(f'band={b}')
			ax.set_xlabel('observation-error [flux-error]')
			ax.set_ylabel('observation [flux]' if kb==0 else None)
			ax.legend()
			ax.set_xlim([0.0, 0.05])
			ax.set_ylim([0.0, 0.4])
			ax.grid(alpha=0.25)

			### multiband colors
			#ax.grid(color=C_.COLOR_DICT[b])
			[ax.spines[border].set_color(C_.COLOR_DICT[b]) for border in ['bottom', 'top', 'right', 'left']]
			[ax.spines[border].set_linewidth(2) for border in ['bottom', 'top', 'right', 'left']]

	title = ''
	title += f'observation-error v/s observation joint distribution'+'\n'
	title += f'survey={lcset.survey}-{"".join(band_names)} [{lcset_name}]'+'\n'
	fig.suptitle(title[:-1], va='bottom', y=.99)#, fontsize=14)
	fig.tight_layout()
	plt.show()
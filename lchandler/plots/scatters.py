from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import matplotlib.pyplot as plt

###################################################################################################################################################

def plot_obs_obse_scatter(lcdataset, set_name1, set_name2,
	figsize:tuple=(12,8),
	alpha=0.2,
	markersize=2,
	n=2e3,
	):
	fig, axs = plt.subplots(1, 2, figsize=figsize)
	lcset1 = lcdataset[set_name1]
	lcset2 = lcdataset[set_name2]
	band_names = lcset2.band_names
	for kb,b in enumerate(band_names):
		ax = axs[kb]
		ax.plot(lcset1.get_lcset_values_b(b, 'obse'), lcset1.get_lcset_values_b(b, 'obs'), 'k.', markersize=markersize, alpha=alpha)
		ax.plot(np.nan, np.nan, 'k.', alpha=1, label=f'original samples {set_name1}')

		obse = lcset2.get_lcset_values_b(b, 'obse')
		obs = lcset2.get_lcset_values_b(b, 'obs')
		idxs = np.random.permutation(np.arange(0, len(obse)))[:int(n)]
		obse = obse[idxs]
		obs = obs[idxs]
		ax.plot(obse, obs, 'r.', markersize=markersize, alpha=alpha)
		ax.plot(np.nan, np.nan, 'r.', alpha=1, label=f'original samples {set_name2}')

		title = f'survey:{lcset1.survey} - set: {set_name1}/{set_name2} - band: {b}'
		ax.set_title(title)
		ax.set_xlabel('obs error')
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
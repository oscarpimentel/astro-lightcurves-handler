from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import matplotlib.pyplot as plt
import flamingchoripan.cutePlots.colors as cc

###################################################################################################################################################

def get_synth_keys(keys, key,
	return_indexs=False,
	):
	skeys = []
	sindexs = []
	for k,skey in enumerate(keys):
		original_key_name, index = skey.split('.')
		if original_key_name==str(key) and int(index)>0:
			skeys.append(skey)
			sindexs.append(k)

	if return_indexs:
		return skeys, sindexs
	return skeys

###################################################################################################################################################

def plot_2Dprojections(lcdataset, set_name_train, pm_args_embd_results_train, set_name_test, pm_args_embd_results_test,
	target_class=None,
	figsize:tuple=(13,8),
	max_synth_samples:int=1,
	max_samples:int=200,
	x_mode='x_umap',
	alpha=0.4,
	):
	lcset_train = lcdataset.get(set_name_train)
	lcset_test = lcdataset.get(set_name_test)
	fig, axs = plt.subplots(1, len(lcset_train.band_names), figsize=figsize)
	has_label = {b:{c:False for c in lcset_train.class_names} for b in lcset_train.band_names}
	points_counter = {b:{c:0 for c in lcset_train.class_names} for b in lcset_train.band_names}
	for kb,b in enumerate(lcset_train.band_names):
		ax = axs[kb]
		tkeys = pm_args_embd_results_train[b]['keys']
		for k,tkey in enumerate(tkeys):
			y = lcset_train.data[tkey].y
			c = lcset_train.class_names[y]

			if not (max_samples is None or points_counter[b][c]<max_samples):
				continue

			if not (target_class is None or c==target_class):
				continue
			
			x_train = pm_args_embd_results_train[b][x_mode][k]
			skeys, sindexs = get_synth_keys(pm_args_embd_results_test[b]['keys'], tkey, return_indexs=True)
			sindexs = sindexs[:max_synth_samples]
			x = pm_args_embd_results_test[b][x_mode][sindexs]

			#print(tkey, skeys)
			#print(lcset_train.data[tkey].g.pm_args)
			#print(lcset_test.data[skeys[0]].g.pm_args)
			
			color = cc.get_default_colorlist()[y]

			if not target_class is None:
				for i in range(len(x)):
					ax.plot([x_train[0], x[i,0]], [x_train[1], x[i,1]], alpha=alpha, lw=1, c=color)
			
			ax.scatter(x[:,0], x[:,1],
				facecolors=[color],
				edgecolors='k',
				s=20,
				alpha=1,
				marker='o',
				lw=1.5,
				linewidth=0.5,
				label=f'{c} (synth)' if not has_label[b][c] else None,
			)
			ax.scatter(x_train[0], x_train[1],
				facecolors=[color],
				edgecolors='k',
				s=100,
				alpha=alpha,
				marker='o',
				linewidth=0.0,
				label=f'{c}' if not has_label[b][c] else None,
			)
			has_label[b][c] = True
			points_counter[b][c] += 1
			#assert 0
	
	dmode = {
		'x_umap':'$PCA_3+UMAP_2$',
		'x_tsne':'$PCA_3+TSNE_2$',
		'x_pca':'$PCA$',
	}
	for kb,b in enumerate(lcset_train.band_names):
		ax = axs[kb]
		title = f'{dmode[x_mode]} projection of parametric model args\n'
		title += f'survey: {lcset_train.survey} - set: {set_name_train} - band: {b}'
		ax.legend()
		ax.set_title(title)
		ax.grid(alpha=0.25)

		### multiband colors
		ax.grid(color=C_.COLOR_DICT[b])
		[ax.spines[border].set_color(C_.COLOR_DICT[b]) for border in ['bottom', 'top', 'right', 'left']]
		[ax.spines[border].set_linewidth(2) for border in ['bottom', 'top', 'right', 'left']]

	fig.tight_layout()
	plt.show()
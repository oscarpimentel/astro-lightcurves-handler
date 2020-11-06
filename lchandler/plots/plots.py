from __future__ import print_function
from __future__ import division
from . import C_

import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import flamingchoripan.cuteplots.plots as cplots
import flamingchoripan.cuteplots.colors as cc
import arviz as az
import pymc3 as pm

###################################################################################################################################################

def plot_obs_obse_scatter(lcdataset, set_name1, set_name2,
	figsize:tuple=(12,8),
	alpha=0.2,
	markersize=2,
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
		n = 2e3
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

###################################################################################################################################################

def plot_obse_samplers(lcdataset, set_name, obse_sampler_bdict,
	original_space:bool=1,
	pdf_scale:float=0.01,
	figsize:tuple=(12,8),
	add_samples=0,
	):
	fig, axs = plt.subplots(1, 2, figsize=figsize)
	band_names = obse_sampler_bdict.keys()
	lcset = lcdataset[set_name]
	for kb,b in enumerate(band_names):
		ax = axs[kb]
		obse_sampler = obse_sampler_bdict[b]
		if original_space:
			label='original samples $\sim p(\sigma_x,x)$'
			ax.plot(obse_sampler.raw_obse, obse_sampler.raw_obs, 'k.', markersize=2, alpha=0.2); ax.plot(np.nan, np.nan, 'k.', label=label)
			x = np.linspace(obse_sampler.raw_obse.min(), obse_sampler.raw_obse.max(), 100)
			ax.plot(x, x*obse_sampler.m+obse_sampler.n, 'b', alpha=0.75, label='rotation axis', lw=1)
			ax.plot(obse_sampler.lr_x, obse_sampler.lr_y, 'b.', alpha=1, markersize=4); ax.plot(np.nan, np.nan, 'b.', label='rotation axis support samples')

			### add samples
			if add_samples:
				n = int(2e3)
				to_sample = obse_sampler.raw_obs
				std = 1e-4
				new_obs = [np.random.normal(to_sample[np.random.randint(0, len(to_sample))], std) for _ in range(n)] # kde
				#x = 0.05; new_obs = np.linspace(x, x+0.001, 1000) # sanity check
				new_obse, new_obs = obse_sampler.conditional_sample(new_obs)
				ax.plot(new_obse, new_obs, 'r.', markersize=2, alpha=1); ax.plot(np.nan, np.nan, 'r.', label='synthetic samples $\sim \hat{p}(\sigma_x,x)$')

		else:
			label='original samples $\sim p(\sigma_x'+"'"+',x'+"'"+')$'
			ax.plot(obse_sampler.obse, obse_sampler.obs, 'k.', markersize=2, alpha=0.2); ax.plot(np.nan, np.nan, 'k.', label=label)
			min_obse = obse_sampler.obse.min()
			max_obse = obse_sampler.obse.max()
			pdfx = np.linspace(min_obse, max_obse, 200)
			colors = cm.viridis(np.linspace(0, 1, len(obse_sampler.distrs)))
			for p_idx in range(len(obse_sampler.distrs)):
				d = obse_sampler.distrs[p_idx]
				
				if p_idx%4==0:
					rank_ranges = obse_sampler.rank_ranges[p_idx]
					pdf_offset = rank_ranges[1] # upper of rank range
					pdfy = d['distr'].pdf(pdfx, *d['params'])
					pdfy = pdfy/pdfy.max()*pdf_scale+pdf_offset
					c = colors[p_idx]
					label = '$\hat{p}(\sigma_x'+"'"+'|x'+"'"+')$ Gamma fit'
					ax.plot(pdfx, pdfy, c=c, alpha=1, lw=1, label=label if p_idx==0 else None)
				
		if original_space:
			title = f'survey:{lcset.survey} - set: {set_name} - band: {b}'
			ax.set_xlabel('obs-error')
			ax.set_ylabel('obs' if kb==0 else None)
			ax.set_xlim([0.0, 0.02])
			ax.set_ylim([0.0, 0.25])
		else:
			title = f'survey:{lcset.survey} - set: {set_name} - band: {b}'
			ax.set_xlabel('rotated-flipped obs-error')
			ax.set_ylabel('rotated obs' if kb==0 else None)
			ax.set_xlim([0.0, 0.02])
			ax.set_ylim([0.0, 0.25])

		ax.set_title(title)
		ax.legend(loc='lower right')

		### multiband colors
		#ax.grid(color=C_.COLOR_DICT[b])
		[ax.spines[border].set_color(C_.COLOR_DICT[b]) for border in ['bottom', 'top', 'right', 'left']]
		[ax.spines[border].set_linewidth(2) for border in ['bottom', 'top', 'right', 'left']]

	fig.tight_layout()
	plt.show()

def plot_length_samplers(length_sampler_bdict, lcdataset, set_name:str,
	figsize:tuple=(12,5),
	):
	lcset = lcdataset[set_name]
	fig, axs = plt.subplots(1, len(lcset.band_names), figsize=figsize)
	for kb,b in enumerate(lcdataset[set_name].band_names):
		ax = axs[kb]
		length_sampler = length_sampler_bdict[b]
		#print(len_sampler.sample(10))
		to_plot = {
			'original':length_sampler.lengths,
			'sampler':length_sampler.sample(1e4),
				  }
		plot_kwargs = {
			'fig':fig,
			'ax':ax,
			'title':'$\{L_i\}_i^N$',
			'xlabel':'$L_i$ values',
			'bins':len(length_sampler.pdf),
			'uses_density':1,
			'return_legend_patches':1,
			'label_samples':0,
			'histtype':'stepfilled',
			'alpha':0.8,
		}
		cplots.plot_hist_bins(to_plot, **plot_kwargs)

		### multiband colors
		ax.grid(color=C_.COLOR_DICT[b])
		[ax.spines[border].set_color(C_.COLOR_DICT[b]) for border in ['bottom', 'top', 'right', 'left']]
		[ax.spines[border].set_linewidth(2) for border in ['bottom', 'top', 'right', 'left']]

	fig.tight_layout()
	plt.show()

###################################################################################################################################################

def plot_sigma_distribution(lcdataset, set_name:str,
	figsize:tuple=(15,10),
	):
	attr = 'obse'
	return plot_values_distribution(lcdataset, set_name, attr, figsize)

def plot_values_distribution(lcdataset, set_name:str, attr:str,
	figsize:tuple=(15,10),
	):
	lcset = lcdataset[set_name]
	fig, axes = plt.subplots(len(lcset.class_names), len(lcset.band_names), figsize=figsize)
	for kb,b in enumerate(lcset.band_names):
		for kc,c in enumerate(lcset.class_names):
			ax = axes[kc,kb]
			to_plot = {c:lcset.get_lcset_values_b(b, attr, c)*100}
			title = f'{C_.LONG_NAME_DICT[attr]} distribution {C_.SYMBOLS_DICT[attr]}\n'
			title += f'survey: {lcset.survey} - set: {set_name} - band: {b}'
			kwargs = {
				'fig':fig,
				'ax':ax,
				'xlabel':C_.XLABEL_DICT[attr] if kc==len(lcset.class_names)-1 else None,
				'ylabel':'' if kb==0 else None,
				'title':title if kc==0 else '',
				#'xlim':[0, 6],
				'bins':100,
				'uses_density':True,
				'legend_loc':'upper right',
				'cmap':cc.get_cmap(cc.get_default_colorlist()[kc:])
			}
			fig, ax = cplots.plot_hist_bins(to_plot, **kwargs)

			### multiband colors
			ax.grid(color=C_.COLOR_DICT[b])
			[ax.spines[border].set_color(C_.COLOR_DICT[b]) for border in ['bottom', 'top', 'right', 'left']]
			[ax.spines[border].set_linewidth(2) for border in ['bottom', 'top', 'right', 'left']]
		
	fig.tight_layout()
	plt.show()

###################################################################################################################################################

def plot_mcmc_trace(mcmc_trace_bdict, b):
	mcmc_trace = mcmc_trace_bdict[b]
	az.plot_trace(mcmc_trace)
	#pm.traceplot(mcmc_trace)
	#pm.autocorrplot(mcmc_trace)
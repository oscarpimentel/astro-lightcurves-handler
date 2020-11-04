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

		ax.plot(lcset2.get_lcset_values_b(b, 'obse'), lcset2.get_lcset_values_b(b, 'obs'), 'r.', markersize=markersize, alpha=alpha)
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
	):
	fig, axs = plt.subplots(1, 2, figsize=figsize)
	band_names = obse_sampler_bdict.keys()
	lcset = lcdataset[set_name]
	for kb,b in enumerate(band_names):
		ax = axs[kb]
		obse_sampler = obse_sampler_bdict[b]
		colors = cm.viridis(np.linspace(0, 1, len(obse_sampler.distrs)))
		txt_i = 0
		for p_idx in range(len(obse_sampler.distrs)):
			d = obse_sampler.distrs[p_idx]
			obse = obse_sampler.raw_obse[obse_sampler.obs_indexs_per_range[p_idx]] if original_space else d['scaler'].transform(-obse_sampler.raw_obse[obse_sampler.obs_indexs_per_range[p_idx]][:,None])[:,0]
			obs = obse_sampler.raw_obs[obse_sampler.obs_indexs_per_range[p_idx]]
			
			if p_idx%5==0:
				self_x =  obse_sampler.obse if original_space else d['scaler'].transform(-obse_sampler.obse[:,None])[:,0]
				ls_x_pdf = np.linspace(self_x.min(), self_x.max(), 100)
				tx_pdf = d['scaler'].transform(-ls_x_pdf[:,None])[:,0] if original_space else ls_x_pdf
				pdf = d['distr'].pdf(tx_pdf, *d['params'])
				rank_ranges = obse_sampler.rank_ranges[p_idx]
				#for k in range(len(ls_x_pdf)-1):
				#    alpha = pdf[k]/pdf.max()
				#    ax.fill_between([ls_x_pdf[k], ls_x_pdf[k+1]], rank_ranges[0], rank_ranges[1], fc='r', alpha=alpha)
				
				#pdf_offset = rank_ranges[0]+(rank_ranges[1]-rank_ranges[0])/2 # middle of rank range
				pdf_offset = rank_ranges[1] # upper of rank range
				pdf = pdf/pdf.max()*pdf_scale+pdf_offset
				alpha = 1-p_idx/len(obse_sampler.distrs)
				alpha = 1
				c = colors[p_idx]
				ax.plot(ls_x_pdf, pdf, c=c, alpha=alpha, lw=1, label='$P(\sigma_x|x)$ beta fit' if p_idx==0 else None)
				
				#txt = f'p[{obse_sampler.percentiles[p_idx]*1:.2f}-{obse_sampler.percentiles[p_idx+1]*1:.2f}]'
				#txt_x = ls_x_pdf[-1] if txt_i%2==0 else ls_x_pdf[0]
				#ax.text(txt_x, pdf[-1], txt, fontsize=8)
				#txt_i += 1
				
			ax.plot(obse, obs, 'k.', markersize=2, alpha=0.2)
		ax.plot(np.nan, np.nan, 'k.', alpha=1, label='original samples')

		#ax.plot(obse_sampler.lr_x, obse_sampler.lr_y, 'ro', alpha=1, label='s', markersize=4)
		x = np.linspace(obse_sampler.raw_obse.min(), obse_sampler.raw_obse.max(), 100)
		ax.plot(x, x*obse_sampler.m+obse_sampler.n, 'r', alpha=1, label='outliers threshold dropout', lw=1)
				
		title = f'survey:{lcset.survey} - set: {set_name} - band: {b}'
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
			'a':length_sampler.lengths,
			'b':length_sampler.sample(1e4),
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
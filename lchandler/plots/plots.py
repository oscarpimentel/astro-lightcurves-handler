from __future__ import print_function
from __future__ import division
from . import C_

import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import flamingchoripan.cuteplots.plots as cplots
import flamingchoripan.cuteplots.colors as cc

###################################################################################################################################################

def plot_obse_samplers(obse_sampler_bdict,
	original_space:bool=1,
	pdf_scale:float=0.01,
	figsize:tuple=(15,8),
	):
	fig, axs = plt.subplots(1, 2, figsize=figsize)
	band_names = obse_sampler_bdict.keys()
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
				
				txt = f'p[{obse_sampler.percentiles[p_idx]*1:.2f}-{obse_sampler.percentiles[p_idx+1]*1:.2f}]'
				txt_x = ls_x_pdf[-1] if txt_i%2==0 else ls_x_pdf[0]
				#ax.text(txt_x, pdf[-1], txt, fontsize=8)
				txt_i += 1
				
			ax.plot(obse, obs, 'k.', markersize=2, alpha=0.2)
		ax.plot(np.nan, np.nan, 'k.', alpha=1, label='original samples')

		#ax.plot(obse_sampler.lr_x, obse_sampler.lr_y, 'ro', alpha=1, label='s', markersize=4)
		x = np.linspace(obse_sampler.raw_obse.min(), obse_sampler.raw_obse.max(), 100)
		ax.plot(x, x*obse_sampler.m+obse_sampler.n, 'r', alpha=1, label='outliers threshold dropout', lw=1)
				
		ax.set_xlabel('obs error')
		ax.set_ylabel('obs')
		ax.legend()
		#ax.set_xlim([0.0025, 0.02])
		ax.set_ylim([0, 0.4])
		#ax.grid(alpha=0.5)
	plt.show()

def plot_values_fit_results(lcdataset, set_name:str, fit_results:dict,
	attr:str='obse',
	figsize:tuple=(12,5),
	):
	lcset = lcdataset[set_name]
	fig, axs = plt.subplots(1, len(lcset.band_names), figsize=figsize)
	for kb,b in enumerate(lcset.band_names):
		ax = axs[kb]
		values = lcset.get_lcset_values_b(b, attr)
		to_plot = {'observations error samples':values}
		title = f'observations error distribution & distribution fittings\n'
		title += f'survey: {lcset.survey} - set: {set_name} - band: {b} - samples: {len(values):,}'
		plot_kwargs = {
			'fig':fig,
			'ax':ax,
			'title':title,
			'xlabel':C_.XLABEL_DICT[attr],
			'bins':500,
			'uses_density':1,
			'return_legend_patches':1,
			'label_samples':0,
			'histtype':'stepfilled',
			'alpha':0.8,
		}
		_,_, legend_patches = cplots.plot_hist_bins(to_plot, **plot_kwargs)
		
		colors = cc.get_default_colorlist(len(fit_results.keys())+2)[2:]
		for kf,distr_name in enumerate(fit_results.keys()):
			params = fit_results[distr_name][b]
			distr = getattr(scipy.stats, distr_name)
			x = np.linspace(0, max(values), int(1e3))
			pdf_fitted = distr.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])
			ax.plot(x, pdf_fitted, '--' if distr_name=='chi2' else '-', label=f'{distr_name} fit', c=colors[kf])

		ax.set_xlim([0, max(values)])
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles=legend_patches+handles)
		if kb>0:
			ax.set_ylabel(None)

		### multiband colors
		ax.grid(color=C_.COLOR_DICT[b])
		[ax.spines[border].set_color(C_.COLOR_DICT[b]) for border in ['bottom', 'top', 'right', 'left']]
		[ax.spines[border].set_linewidth(2) for border in ['bottom', 'top', 'right', 'left']]

	fig.tight_layout()
	plt.show()

def plot_len_fit_results(lcdataset, set_name:str, fit_results:dict,
	attr:str='obse',
	figsize:tuple=(12,5),
	):
	lcset = lcdataset[set_name]
	fig, axs = plt.subplots(1, len(lcset.band_names), figsize=figsize)
	for kb,b in enumerate(lcset.band_names):
		ax = axs[kb]
		values = np.array([len(lcset.data[lcobj_name].get_b(b)) for lcobj_name in lcset.get_lcobj_names()])
		to_plot = {'lengths samples':values}
		title = f'lengths distribution & distribution fittings\n'
		title += f'survey: {lcset.survey} - set: {set_name} - band: {b} - samples: {len(values):,}'
		plot_kwargs = {
			'fig':fig,
			'ax':ax,
			'title':title,
			'xlabel':'lengths',
			'bins':50,
			'uses_density':1,
			'return_legend_patches':1,
			'label_samples':0,
			'histtype':'stepfilled',
			'alpha':0.8,
		}
		_,_, legend_patches = cplots.plot_hist_bins(to_plot, **plot_kwargs)
		
		colors = cc.get_default_colorlist(len(fit_results.keys())+2)[2:]
		for kf,distr_name in enumerate(fit_results.keys()):
			params = fit_results[distr_name][b]
			distr = getattr(scipy.stats, distr_name)
			x = np.linspace(0, max(values), int(1e3))
			pdf_fitted = distr.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])
			ax.plot(x, pdf_fitted, '--' if distr_name=='chi2' else '-', label=f'{distr_name} fit', c=colors[kf])

		ax.set_xlim([0, max(values)])
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles=legend_patches+handles)
		if kb>0:
			ax.set_ylabel(None)

		### multiband colors
		ax.grid(color=C_.COLOR_DICT[b])
		[ax.spines[border].set_color(C_.COLOR_DICT[b]) for border in ['bottom', 'top', 'right', 'left']]
		[ax.spines[border].set_linewidth(2) for border in ['bottom', 'top', 'right', 'left']]

	fig.tight_layout()
	plt.show()

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

def plot_class_distribution(
	figsize=(15,10),
	uses_log_scale:bool=False,
	):
	title = f'survey: {self.survey_name}\nclasses & curve points distributions'
	label_samples = self.labels_df[self.df_index_names['label']].values
	label_names = print(np.unique(label_samples))

	equiv = self.labels_df[self.df_index_names['label']].to_dict()
	#print(equiv)
	#assert 0
	curve_points_samples = self.detections_df.reset_index()[self.df_index_names['oid']].map(equiv).dropna()
	curve_points_samples = curve_points_samples.values
	to_plot = {
		'class samples':label_samples,
		'curve points samples':curve_points_samples,
	}
	fig, ax = plot_hist_labels(to_plot, self.classes_names, title=title, figsize=figsize, uses_log_scale=uses_log_scale)
	fig.tight_layout()
	plt.show()
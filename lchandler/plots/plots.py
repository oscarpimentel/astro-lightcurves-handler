from __future__ import print_function
from __future__ import division
from . import C_

import scipy
import numpy as np
import matplotlib.pyplot as plt
import flamingchoripan.cuteplots.plots as cplots
import flamingchoripan.cuteplots.colors as cc
from ..synthetic.synthetic_SNe_fun import SNE_fun_numpy as func

###################################################################################################################################################

def plot_values_fit_distr_sampler(lcdataset, set_name:str, obse_sampler,
	attr:str='obse',
	samples:int=None,
	figsize:tuple=(12,5),
	):
	lcset = lcdataset[set_name]
	fig, axs = plt.subplots(1, len(lcset.band_names), figsize=figsize)
	for kb,b in enumerate(lcset.band_names):
		ax = axs[kb]
		values = lcset.get_lcset_values_b(b, attr)
		obse_sampler_values = obse_sampler.sample(len(values) if samples is None else samples, b)
		to_plot = {
			'observations error samples':values,
			f'observations error samples ({obse_sampler.distr_name})':obse_sampler_values,
			}
		title = f'observations error distribution & {obse_sampler.distr_name} distribution fit\n'
		title += f'survey: {lcset.survey} - set: {set_name} - band: {b} - samples: {len(values):,}'
		plot_kwargs = {
			'fig':fig,
			'ax':ax,
			'title':title,
			'xlabel':C_.XLABEL_DICT[attr],
			'bins':500,
			'uses_density':1,
			'label_samples':0,
			#'histtype':'stepfilled',
			#'alpha':1,
		}
		fig, ax = cplots.plot_hist_bins(to_plot, **plot_kwargs)
		if kb>0:
			ax.set_ylabel(None)

		### multiband colors
		ax.grid(color=C_.COLOR_DICT[b])
		[ax.spines[border].set_color(C_.COLOR_DICT[b]) for border in ['bottom', 'top', 'right', 'left']]
		[ax.spines[border].set_linewidth(2) for border in ['bottom', 'top', 'right', 'left']]

	fig.tight_layout()
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
		values = np.array([len(lcset.data[key].get_b(b)) for key in lcset.data_keys()])
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
				'bins':80,
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
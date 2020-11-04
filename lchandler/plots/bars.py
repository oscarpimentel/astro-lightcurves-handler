from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import matplotlib.pyplot as plt
import flamingchoripan.cuteplots.plots as cplots
import flamingchoripan.cuteplots.colors as cc

###################################################################################################################################################

def plot_class_distribution_df(
	figsize=(15,10),
	uses_log_scale:bool=False,
	):
	title = f'survey: {self.survey_name}\nclasses & curve points distributions'
	label_samples = self.labels_df[self.df_index_names['label']].values

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

def plot_class_distribution(lcdataset, lcset_name1, lcset_name2,
	figsize=(12,4),
	):
	fig, axs = plt.subplots(1, 2, figsize=figsize)
	for ks,lcset_name in enumerate([lcset_name1, lcset_name2]):
		ax = axs[ks]
		lcset = lcdataset[lcset_name]
		lcobj_classes = lcset.get_lcobj_classes()
		to_plot = {'class samples':lcobj_classes}
		title = 'class population distribution\n'
		title += f'survey: {lcset.survey} - set: {lcset_name} - N: {len(lcobj_classes):,}'
		plt_kwargs = {
			'fig':fig,
			'ax':ax,
			'ylabel':'' if ks>0 else None,
			'title':title,
			'cmap':cc.colorlist_to_cmap([cc.NICE_COLORS_DICT['nice_gray']]),
			'uses_log_scale':0,
		}
		fig, ax = cplots.plot_hist_labels(to_plot, lcset.class_names, **plt_kwargs)
		
	fig.tight_layout()
	plt.plot()

def plot_mean_length_distribution(lcdataset, lcset_name1, lcset_name2,
	figsize=(12,4),
	):
	fig, axs = plt.subplots(1, 2, figsize=figsize)
	for ks,lcset_name in enumerate([lcset_name1, lcset_name2]):
		ax = axs[ks]
		lcset = lcdataset[lcset_name]
		lcobj_classes = lcset.get_lcobj_classes()
		to_plot = {c:{f'{b} band':lcset.get_lcobj_obsmean_b_cdict(b)[c] for b in lcset.band_names} for c in lcset.class_names}
		title = 'curve mean length distribution per band\n'
		title += f'survey: {lcset.survey} - set: {lcset_name} - N: {len(lcobj_classes):,}'
		plt_kwargs = {
			'fig':fig,
			'ax':ax,
			'ylabel':'' if ks>0 else 'curve mean length',
			#'legend_ncol':len(lcset.band_names),
			'uses_bottom_legend':0,
			'legend_loc':'upper right',
			'title':title,
			'cmap':cc.colorlist_to_cmap([C_.COLOR_DICT[b] for b in lcset.band_names]),
			'add_percent_annotations':True,
		}
		fig, ax = cplots.plot_bar(to_plot, [f'{b} band' for b in lcset.band_names], **plt_kwargs)
		
	fig.tight_layout()
	plt.plot()
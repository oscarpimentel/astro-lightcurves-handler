from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
import matplotlib.pyplot as plt
import fuzzytools.matplotlib.plots as cplots
import fuzzytools.matplotlib.colors as cc

###################################################################################################################################################

def plot_class_distribution_df(labels_df, detections_df, label_to_class_dict, df_index_names, class_names, band_dictionary, survey_name,
	figsize=None,
	uses_log_scale:bool=False,
	band_names:list=['g','r'],
	add_band_lengths:bool=False,
	rotate_xlabel:bool=False,
	caption=None,
	):
	label_samples = labels_df[df_index_names['label']].values
	to_plot = {'class samples':[label_to_class_dict[l] for l in label_samples]}

	if add_band_lengths: # slow process
		band_index = df_index_names['band']
		for b in band_names:
			b_key = band_dictionary[b]
			equiv = labels_df[df_index_names['label']].to_dict()
			detections_df = detections_df.reset_index()
			detections_df = detections_df.drop(detections_df[getattr(detections_df, band_index)!=b_key].index)
			curve_points_samples = detections_df[df_index_names['oid']].map(equiv).dropna()
			curve_points_samples = curve_points_samples.values
			to_plot[f'obs samples - band={b}'] = [label_to_class_dict[l] for l in curve_points_samples]

	#print(to_plot)
	cmap = cc.colorlist_to_cmap([cc.NICE_COLORS_DICT['nice_gray']]+[_C.COLOR_DICT[b] for b in band_names])
	title = ''
	# title += f'SNe class distribution'+'\n'
	title += f'set={survey_name}-{"".join(band_names)}; total samples={len(label_samples):,}#'+'\n'
	plt_kwargs = {
		'title':title[:-1],
		'uses_log_scale':uses_log_scale,
		'cmap':cmap,
		'legend_ncol':len(band_names),
		'rotate_xlabel':rotate_xlabel,
		'figsize':figsize,
		'xlabel':'class population',
	}
	fig, ax = cplots.plot_hist_labels(to_plot, class_names, **plt_kwargs)
	fig.text(.5,.0, caption, fontsize=12)
	fig.tight_layout()
	plt.show()
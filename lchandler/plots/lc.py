from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import matplotlib.pyplot as plt
import flamingchoripan.lists as lists
from flamingchoripan.cuteplots.utils import save_fig
import scipy.stats as stats

###################################################################################################################################################

def get_margin(x, x_per):
	if len(x)==0:
		return [0, 0]
	min_x = min(x)
	max_x = max(x)
	abs_x = abs(min_x - max_x)
	return [min_x-abs_x*x_per/100., max_x+abs_x*x_per/100.]

###################################################################################################################################################

def plot_synthetic_samples(lcdataset, set_name:str, method, lcobj_name, new_lcobjs, new_lcobjs_pm,
	fit_errors_bdict=None,
	figsize:tuple=(13,6),
	lw=1.5,
	save_rootdir=None,
	):
	lcset = lcdataset[set_name]
	fig, axs = plt.subplots(1, 2, figsize=figsize)
	band_names = lcset.band_names
	lcobj = lcset[lcobj_name]
	idx = 0

	###
	ax = axs[0]
	for b in band_names:
	    plot_lightcurve(ax, lcobj, b, label=f'{b} observation')
	    for k,new_lcobj_pm in enumerate(new_lcobjs_pm):
	        label = f'{b} posterior pm-sample' if k==0 else None
	        ax.plot(new_lcobj_pm.get_b(b).days, new_lcobj_pm.get_b(b).obs, alpha=0.15, lw=1, c=C_.COLOR_DICT[b]); ax.plot(np.nan, np.nan, lw=1, c=C_.COLOR_DICT[b], label=label)
	ax.grid(alpha=0.5)
	title = f'multiband light curve & parametric model samples\n'
	title += f'method: {method} - '+' - '.join([f'{b}-error: {np.mean(fit_errors_bdict[b]):.2f}$\pm${np.std(fit_errors_bdict[b]):.1f}' for b in band_names])+'\n'
	title += f'survey: {lcset.survey}/{set_name} - obj: {lcobj_name}- class: {lcset.class_names[lcobj.y]}'
	ax.set_title(title)
	ax.legend(loc='upper right')
	ax.set_ylabel('obs [flux]')
	ax.set_xlabel('days')

	###
	ax = axs[1]
	for b in band_names:
	    plot_lightcurve(ax, lcobj, b, label=f'{b} observation')
	    for k,new_lcobj in enumerate([new_lcobjs[idx]]):
	        plot_lightcurve(ax, new_lcobj, b, label=f'{b} observation' if k==0 else None, is_synthetic=1)
	        
	ax.grid(alpha=0.5)
	title = f'multiband light curve & synthetic curve example\n'
	title += f'method: {method} - '+' - '.join([f'{b}-error: {fit_errors_bdict[b][idx]:.2f}' for b in band_names])+'\n'
	title += f'survey: {lcset.survey}/{set_name} - obj: {lcobj_name}- class: {lcset.class_names[lcobj.y]}'
	ax.set_title(title)
	ax.legend(loc='upper right')
	#ax.set_ylabel('obs [flux]')
	ax.set_xlabel('days')

	fig.tight_layout()
	save_filedir = None if save_rootdir is None else f'{save_rootdir}/{lcset.survey}/{method}/{lcobj_name}.png'
	save_fig(fig, save_filedir)

def plot_lightcurve(ax, lcobj, b,
	max_day:float=np.infty,

	label:str=None,
	alpha:float=0.25,
	mode:str='bar', # shadow, bar, gauss
	capsize:int=0,
	x_margin_offset_percent:float=1,
	y_margin_offset_percent:float=10,
	is_synthetic:bool=False,
	std_factor:int=C_.OBSE_STD_SCALE, # asuming error as gaussian error
	percentile_bar:float=0.90, # show bars as percentile bound
	):
	lcobjb = lcobj.get_b(b)
	new_days = lcobjb.days
	valid_indexs = new_days<=max_day
	new_days = new_days[valid_indexs]
	obs = lcobjb.obs[valid_indexs]
	obse = lcobjb.obse[valid_indexs]

	bar = stats.norm(loc=0, scale=std_factor*obse).ppf(percentile_bar)
	color = C_.COLOR_DICT[b]
	if mode=='shadow':
		ax.fill_between(new_days, obs-bar, obs+bar, facecolor=color, alpha=alpha)
	elif mode=='bar':
		ax.errorbar(new_days, obs, yerr=bar, color=color, capsize=capsize, elinewidth=1, linewidth=0)
	else:
		raise Exception(f'not supported mode: {mode}')
	
	ax.plot(new_days, obs, '--', color=color, alpha=alpha)
	if is_synthetic and not label is None:
		label = label+' (synth)'
	ax.plot(new_days, obs, 'o', color=color, label=label, markeredgecolor='k' if is_synthetic else None)

	x_margins = get_margin(new_days, x_margin_offset_percent)
	y_margins = get_margin(obs, y_margin_offset_percent)
	return x_margins, y_margins

from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

###################################################################################################################################################

def get_margin(x, x_per):
	if len(x)==0:
		return [0, 0]
	min_x = min(x)
	max_x = max(x)
	abs_x = abs(min_x - max_x)
	return [min_x-abs_x*x_per/100., max_x+abs_x*x_per/100.]

def plot_lightcurve(ax, lcobj, b,
	max_day:float=np.infty,

	label:str=None,
	alpha:float=1,
	mode:str='bar', # shadow, bar, gauss
	capsize:int=0,
	x_margin_offset_percent:float=1,
	y_margin_offset_percent:float=10,
	std_factor:int=C_.PLOT_OBSE_STD_SCALE, # asuming error as gaussian error
	percentile_bar:float=C_.PLOT_PERCENTILE_BAR, # show bars as percentile bound
	show_obs_len=True,
	):
	'''
	plot a light curve!!
	bar errors asume that observation error is the scale of a gaussian distribution
	'''
	lcobjb = lcobj.get_b(b)
	is_synthetic = lcobjb.synthetic
	new_days = lcobjb.days
	valid_indexs = new_days<=max_day
	new_days = new_days[valid_indexs]
	obs = lcobjb.obs[valid_indexs]
	obse = lcobjb.obse[valid_indexs]

	bar = stats.norm(loc=0, scale=std_factor*obse).ppf(percentile_bar/100.) # I think It's correct, check with standar norm and p=84.1 > bar=1
	color = C_.COLOR_DICT[b]
	if mode=='shadow':
		ax.fill_between(new_days, obs-bar, obs+bar, facecolor=color, alpha=0.25)
	elif mode=='bar':
		ax.errorbar(new_days, obs, yerr=bar, color=color, capsize=capsize, elinewidth=1, linewidth=0, alpha=alpha)
	else:
		raise Exception(f'not supported mode: {mode}')
	
	ax.plot(new_days, obs, ':', color=color, alpha=0.25*alpha)
	label = f'{label} [synth]' if is_synthetic and not label is None else label
	label = f'{label} ({len(obs):,})' if show_obs_len else label
	ax.plot(new_days, obs, 'o', color=color, label=label, alpha=alpha, markeredgecolor='k' if is_synthetic else None)

	x_margins = get_margin(new_days, x_margin_offset_percent)
	y_margins = get_margin(obs, y_margin_offset_percent)
	return x_margins, y_margins

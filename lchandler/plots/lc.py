from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import matplotlib.pyplot as plt
import flamingchoripan.myUtils.lists as lists

###################################################################################################################################################

def get_margin(x, x_per):
	if len(x)==0:
		return [0, 0]
	min_x = min(x)
	max_x = max(x)
	abs_x = abs(min_x - max_x)
	return [min_x-abs_x*x_per/100., max_x+abs_x*x_per/100.]

###################################################################################################################################################

def plot_synthetic_samples(lcdataset, set_name:str,
	key:'str'=None,
	figsize:tuple=(12,5),
	lw=1.5,
	max_samples:int=2,
	):
	lcset = lcdataset.get(set_name)
	synth_lcset = lcdataset.get(f'synth_{set_name}')
	key = lists.get_random_item(synth_lcset.used_keys) if key is None else key
	synth_keys = get_synth_keys(synth_lcset.data_keys(), key)
	print(f'key: {key} - synth_keys: {synth_keys}')

	figsize = (14,6)
	fig, ax = plt.subplots(1, 1, figsize=figsize)

	y = lcdataset.get(set_name).data[key].y
	for sk,skey in enumerate(synth_keys):
		if sk>=max_samples:
			break

		synth_lcobj = synth_lcset.data[skey]
		for kb,b in enumerate(synth_lcset.band_names):
			synth_lcobjb = synth_lcobj.get_b(b)
			if synth_lcobjb.pm_times is None:
				continue

			days_fit = np.linspace(synth_lcobjb.pm_times['ti'], synth_lcobjb.pm_times['tf'], 100)
			if not synth_lcobjb.pm_guess is None:
				### guess
				func_args = synth_lcobjb.pm_guess
				obs_fit = func(days_fit, **func_args)
				label = f'{b} pm guess' if sk==0 else None
				ax.plot(days_fit, obs_fit, '-', c=cc.get_colorlist('cc_black')[kb], label=label, alpha=0.75, lw=lw)
				
				### fit
				func_args = synth_lcobjb.pm_args
				obs_fit = func(days_fit, **func_args)
				label = f'{b} pm fit' if sk==0 else None
				ax.plot(days_fit, obs_fit, '-', c=C_.COLOR_DICT[b], label=label, lw=lw)

				### tmax
				tmax = synth_lcobjb.pm_times['tmax']
				ax.axvline(tmax, ls='--', lw=lw, c=C_.COLOR_DICT[b], label=f'{b} tmax')

			plot_lightcurve(ax, synth_lcobj, b, label=f'{b} observation' if sk==0 else None, is_synthetic=True)

	for kb,b in enumerate(lcset.band_names):
		plot_lightcurve(ax, lcset.data[key], b, label=f'{b} observation')
		
	ax.legend(loc='upper right')
	ax.grid(alpha=0.5)
	title = 'multiband light curve example & parametric model fitting\n'
	title += f'survey: {synth_lcset.survey} - set: {set_name} - key: {key} - class: {synth_lcset.class_names[y]}'
	ax.set_title(title)
	ax.set_xlabel('days')
	ax.set_ylabel('flux')
	plt.show()

def plot_lightcurve(ax, lcobj, b,
	max_day:float=np.infty,

	label:str=None,
	alpha:float=0.25,
	mode:str='bar', # shadow, bar, gauss
	capsize:int=0,
	x_margin_offset_percent:float=1,
	y_margin_offset_percent:float=10,
	is_synthetic:bool=False,
	std_factor:int=1, # asuming error as error = std*2
	# 1:68.27%, 2:95.45%, 3:99.73%
	):
	lcobjb = lcobj.get_b(b)
	new_days = lcobjb.days
	valid_indexs = new_days<=max_day
	new_days = new_days[valid_indexs]
	obs = lcobjb.obs[valid_indexs]
	obse = lcobjb.obse[valid_indexs]

	color = C_.COLOR_DICT[b]
	if mode=='shadow':
		ax.fill_between(new_days, obs-obse*std_factor, obs+obse*std_factor, facecolor=color, alpha=alpha)
	elif mode=='bar':
		ax.errorbar(new_days, obs, yerr=obse*std_factor, color=color, capsize=capsize, elinewidth=1, linewidth=0)
	elif mode=='gauss':
		assert 0
		'''
		for k,_ in enumerate(new_days):
			new_x = new_days[k]+np.linspace(-obs_error[k]*std_factor, obs_error[k]*std_factor, 100)
			new_y = new_x*0+obs[k]
			ax.plot(new_x, new_y, '-')
		ax.errorbar(new_days, obs, yerr=obs_error*std_factor, color=color, capsize=capsize, elinewidth=1, linewidth=0)
		'''
	else:
		raise Exception(f'not supported mode: {mode}')
	
	ax.plot(new_days, obs, '-', color=color, alpha=alpha)
	if is_synthetic and not label is None:
		label = label+' (synth)'
	ax.plot(new_days, obs, 'o', color=color, label=label, markeredgecolor='k' if is_synthetic else None)

	x_margins = get_margin(new_days, x_margin_offset_percent)
	y_margins = get_margin(obs, y_margin_offset_percent)
	return x_margins, y_margins

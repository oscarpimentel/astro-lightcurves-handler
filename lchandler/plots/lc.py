from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import matplotlib.pyplot as plt
from fuzzytools.strings import get_string_from_dict
import fuzzytools.cuteplots.bars as bars

###################################################################################################################################################

def get_margin(x, x_per):
	if len(x)==0:
		return [0, 0]
	min_x = min(x)
	max_x = max(x)
	abs_x = abs(min_x - max_x)
	return [min_x-abs_x*x_per/100., max_x+abs_x*x_per/100.]

def plot_lightcurve(ax, lcobj, b, label,
	max_day:float=np.infty,
	alpha:float=1,
	capsize:int=0,
	x_margin_offset_percent:float=1,
	y_margin_offset_percent:float=10,
	label_snr=True,
	):
	lcobjb = lcobj.get_b(b) if not b is None else lcobj
	new_days = lcobjb.days
	valid_indexs = new_days<=max_day
	new_days = new_days[valid_indexs]
	obs = lcobjb.obs[valid_indexs]
	obse = lcobjb.obse[valid_indexs]
	color = C_.COLOR_DICT[b] if not b is None else 'k'
	bars.plot_norm_percentile_bar(ax, new_days, obs, obse, color=color)
	ax.plot(new_days, obs, ':', color=color, alpha=0.25*alpha)

	synth_label = ' [synth]' if lcobjb.is_synthetic() else ''
	extra_label = []
	extra_label += [f'snr={lcobjb.get_snr():.3f}'] if label_snr else []
	extra_label = f' {"; ".join(extra_label)}' if len(extra_label)>0 else ''
	ax.plot(new_days, obs, 'o',
		color=color,
		label=f'{label}{synth_label}{extra_label} ({len(obs):,}#)' if not label is None else None,
		alpha=alpha,
		markeredgecolor='k' if lcobjb.is_synthetic() else None,
		)
	x_margins = get_margin(new_days, x_margin_offset_percent)
	y_margins = get_margin(obs, y_margin_offset_percent)
	return x_margins, y_margins
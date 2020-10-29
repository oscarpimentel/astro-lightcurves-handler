from __future__ import print_function
from __future__ import division
from . import C_


import numpy as np
import copy
from flamingchoripan.dataScience.statistics import get_sigma_clipping_indexing
from flamingchoripan.myUtils.prints import HiddenPrints, ShowPrints

def search_over_sigma_samples(lcset, b:str, dist_mean, dist_sigma, sigma_m,
	apply_lower_bound:bool=True,
	):
	total_deleted_points = 0
	for key in lcset.data_keys():
		sigmas = lcset.data[key].get_b(b).obse
		valid_indexs = get_sigma_clipping_indexing(sigmas, dist_mean, dist_sigma, sigma_m, apply_lower_bound)
		deleted_points = (~valid_indexs).astype(int).sum()
		total_deleted_points += deleted_points
		lcset.data[key].get_b(b).apply_valid_indexs_to_attrs(valid_indexs)

	return total_deleted_points

def sigma_clipping(lcdataset, set_name,
	sigma_n:int=1,
	sigma_m:float=3.,
	apply_lower_bound:bool=True,
	verbose:int=1,
	):
	new_set_name = set_name.split('_')[-1]
	lcset = lcdataset.set_custom(new_set_name, lcdataset.get(set_name).copy())
	print(f'survey: {lcset.survey} - after processing: {set_name} (>{new_set_name})')
	printClass = ShowPrints if verbose else HiddenPrints
	total_deleted_points = {b:0 for b in lcset.band_names}
	with printClass():
		for k in range(sigma_n):
			print(f'k: {k}')
			for b in lcset.band_names:
				sigma_values = lcset.get_lcset_values_b(b, 'obse')
				sigma_samples = len(sigma_values)
				mean = np.mean(sigma_values)
				sigma = np.std(sigma_values)
				deleted_points = search_over_sigma_samples(lcset, b, mean, sigma, sigma_m, apply_lower_bound)
				print(f'\tband: {b} - sigma_samples: {sigma_samples:,} - mean: {mean} - std: {sigma}')
				print(f'\tdeleted_points: {deleted_points:,}')
				total_deleted_points[b] += deleted_points
	
	lcset.clean_empty_obs_keys()
	lcset.reset_day_offset_serial()
	sigma_samples = len(lcset.get_lcset_values_b(b, 'obse'))
	print(f'sigma_samples: {sigma_samples:,} - total_deleted_points: {total_deleted_points}')
	return total_deleted_points
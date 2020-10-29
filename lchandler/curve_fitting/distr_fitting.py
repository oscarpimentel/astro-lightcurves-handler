from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import scipy.stats as stats

class ValuesSampler():
	def __init__(self, params_dict:dict, distr_name:str,
		offset:float=0,
		default_band:str=None,
		int_output:bool=False,
		):
		self.params_dict = params_dict[distr_name]
		self.distr_name = distr_name
		self.offset = offset
		self.default_band = default_band
		self.int_output = int_output

	def sample(self, size:int,
		b:str=None,
		):
		params = self.params_dict[b if self.default_band is None else self.default_band]
		distr = getattr(stats, self.distr_name)
		samples = distr.rvs(*params, size=size) + self.offset
		if self.int_output:
			return samples.astype(np.int)
		return samples

def get_values_distribution_fits(lcdataset, set_name:str, distr_names:list,
	attr:str='obse',
	):
	lcset = lcdataset.get(set_name)
	results = {d:{} for d in distr_names}
	for k,distr_name in enumerate(distr_names):
		for kb,b in enumerate(lcset.band_names):
			values = lcset.get_lcset_values_b(b, attr)
			floc = values.min()-1e-5
			distr = getattr(stats, distr_name)
			params = distr.fit(values, floc=floc)
			print(f'distr_name: {distr_name} - band: {b} - samples: {len(values):,} - params: {params}')
			results[distr_name][b] = params

	return results

def get_len_distribution_fits(lcdataset, set_name:str, distr_names:list,
	):
	lcset = lcdataset.get(set_name)
	results = {d:{} for d in distr_names}
	for k,distr_name in enumerate(distr_names):
		for kb,b in enumerate(lcset.band_names):
			values = np.array([len(lcset.data[key].get_b(b)) for key in lcset.data_keys()])
			floc = values.min()-1e-5
			distr = getattr(stats, distr_name)
			params = distr.fit(values, floc=floc)
			print(f'distr_name: {distr_name} - band: {b} - samples: {len(values):,} - params: {params}')
			results[distr_name][b] = params

	return results
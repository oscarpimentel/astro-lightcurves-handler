from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import random
import copy
from flamingchoripan.datascience.statistics import get_sigma_clipping_indexing
from flamingchoripan.prints import HiddenPrints, ShowPrints

###################################################################################################################################################

def search_over_sigma_samples(lcset, b:str, dist_mean, dist_sigma, sigma_m,
	apply_lower_bound:bool=True,
	):
	total_deleted_points = 0
	for lcobj_name in lcset.get_lcobj_names():
		sigmas = lcset[lcobj_name].get_b(b).obse # get
		valid_indexs = get_sigma_clipping_indexing(sigmas, dist_mean, dist_sigma, sigma_m, apply_lower_bound)
		deleted_points = (~valid_indexs).astype(int).sum()
		total_deleted_points += deleted_points
		lcset.data[lcobj_name].get_b(b).apply_valid_indexs_to_attrs(valid_indexs) # set

	return total_deleted_points

###################################################################################################################################################

class LCDataset():
	def __init__(self):
		self.lcsets = {}

	def set_lcset(self, lcset_name:str, lcset):
		self.lcsets[lcset_name] = lcset
		return lcset

	def get_lcset_names(self):
		return list(self.lcsets.keys())

	def exists(self, lcset_name):
		return lcset_name in self.get_lcset_names()

	def del_lcset(self, lcset_name):
		self.lcsets.pop(lcset_name, None)

	def __getitem__(self, lcset_name):
		return self.lcsets[lcset_name]

	def __repr__(self):
		txt = 'LCDataset(\n'
		for lcset_name in self.get_lcset_names():
			txt += f'({lcset_name} - samples {len(self[lcset_name]):,})\n - {self[lcset_name]}\n'
		txt += ')'
		return txt

	def clean_empty_obs_keys(self):
		'''
		Along all lcsets
		Use this to delete empty light curves: 0 obs in all bands
		'''
		for lcset_name in self.get_lcset_names():
			deleted_keys = self[lcset_name].clean_empty_obs_keys(verbose=0)
			print(f'({lcset_name}) deleted keys: {deleted_keys}')

	def split(self, to_split_lcset_name, new_lcsets,
		random_state=42,
		):
		'''stratified'''
		assert sum([new_lcsets[k] for k in new_lcsets.keys()])==1.
		assert len(new_lcsets.keys())>=2

		to_split_lcset = self[to_split_lcset_name]
		lcobj_names = self[to_split_lcset_name].get_lcobj_names()
		random.seed(random_state)
		random.shuffle(lcobj_names)
		to_split_lcset_data = {k:self[to_split_lcset_name].data[k] for k in lcobj_names}
		populations_cdict = to_split_lcset.get_populations_cdict()
		class_names = to_split_lcset.class_names

		for k,new_lcset_name in enumerate(new_lcsets.keys()):
			self.set_lcset(new_lcset_name, to_split_lcset.copy({}))
			for c in class_names:
				to_fill_pop = int(populations_cdict[c]*new_lcsets[new_lcset_name])
				lcobj_names = np.array(list(to_split_lcset_data.keys()))
				lcobj_classes = np.array([class_names[to_split_lcset_data[lcobj_name].y] for lcobj_name in lcobj_names])
				valid_indexs = np.where(lcobj_classes==c)[0][:to_fill_pop] if k<=len(new_lcsets.keys())-2 else np.where(lcobj_classes==c)[0]
				lcobj_names = lcobj_names[valid_indexs].tolist()

				for lcobj_name in lcobj_names:
					lcobj = to_split_lcset_data.pop(lcobj_name)
					self[new_lcset_name].data.update({lcobj_name:lcobj})

		return

	### sigma clipping

	def sigma_clipping(self, lcset_name, new_lcset_name,
		sigma_n:int=1,
		sigma_m:float=3.,
		apply_lower_bound:bool=True,
		verbose:int=1,
		):
		printClass = ShowPrints if verbose else HiddenPrints
		with printClass():
			lcset = self.set_lcset(new_lcset_name, self[lcset_name].copy())
			print(f'survey: {lcset.survey} - after processing: {lcset_name} (>{new_lcset_name})')
			total_deleted_points = {b:0 for b in lcset.band_names}
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

###################################################################################################################################################

class LCSet():
	def __init__(self,
		data:dict,
		survey:str,
		description:str,
		band_names:list,
		class_names:list,
		obs_is_flux:bool,
		):
		self.data = data
		self.survey = survey
		self.description = description
		self.band_names = band_names.copy()
		self.class_names = class_names.copy()
		self.obs_is_flux = obs_is_flux

	def __getitem__(self, lcobj_name):
		return self.data[lcobj_name]

	def get_lcobj_names(self):
		return list(self.data.keys())

	def get_lcobjs(self):
		return [self[lcobj_name] for lcobj_name in self.get_lcobj_names()]

	def clean_empty_obs_keys(self,
		verbose:int=1,
		):
		to_delete_lcobj_names = [lcobj_name for lcobj_name in self.get_lcobj_names() if not np.any([len(self[lcobj_name].get_b(b))>=C_.MIN_POINTS_LIGHTCURVE_DEFINITION for b in self.band_names])]
		deleted_lcobjs = len(to_delete_lcobj_names)

		if verbose:
			print(f'deleted lcobjs: {deleted_lcobjs}')

		for lcobj_name in to_delete_lcobj_names:
			self.data.pop(lcobj_name, None)

		return deleted_lcobjs

	def get_random_lcobj_name(self):
		lcobj_names = self.get_lcobj_names()
		return lcobj_names[random.randint(0, len(lcobj_names)-1)]

	def get_random_lcobj(self,
		return_key:bool=True,
		):
		lcobj_name = self.get_random_lcobj_name()
		if return_key:
			return self[lcobj_name], lcobj_name
		return self[lcobj_name]

	def set_lcobj(self, lcobj_name, lcobj):
		self.data[lcobj_name] = lcobj

	def set_diff_parallel(self, attr:str):
		'''
		Along all keys
		'''
		for lcobj_name in self.get_lcobj_names():
			self[lcobj_name].set_diff_parallel(attr)

	def set_log_parallel(self, attr:str):
		'''
		Along all keys
		'''
		for lcobj_name in self.get_lcobj_names():
			self[lcobj_name].set_log_parallel(attr)

	def keys(self):
		return self.__dict__.keys()

	def get_lcobj_labels(self):
		return [lcobj.y for lcobj in self.get_lcobjs()]

	def get_lcobj_classes(self):
		'''
		Used for classes histogram
		'''
		return [self.class_names[y] for y in self.get_lcobj_labels()]

	def get_populations_cdict(self):
		lcobj_classes = self.get_lcobj_classes()
		uniques, counts = np.unique(lcobj_classes, return_counts=True)
		return {c:counts[list(uniques).index(c)] for c in self.class_names}

	def __repr__(self):
		obs_len = sum([len(lcobj) for lcobj in self.get_lcobjs()])
		obs_len_dict = {b:sum([len(lcobj.get_b(b)) for lcobj in self.get_lcobjs()]) for b in self.band_names}
		obs_len_txt = ' - '.join([f'{b}: {obs_len_dict[b]:,}' for b in self.band_names])
		max_duration = max([lcobj.get_days_serial_duration() for lcobj in self.get_lcobjs()])

		txt = f'samples: {len(self):,} - obs samples: {obs_len:,} ({obs_len_txt})\n'
		txt += f' - max_length_serial: {self.get_max_length_serial()} - max_duration: {max_duration:.2f}[days]\n'
		populations_cdict = self.get_populations_cdict()
		total_population = sum([populations_cdict[k] for k in populations_cdict.keys()])
		txt += f' - population: '+' - '.join([f'{k}: {populations_cdict[k]:,}({populations_cdict[k]/total_population*100:.2f}%)' for k in populations_cdict.keys()])
		return txt

	def __len__(self):
		return len(self.get_lcobj_names())

	def get_lcobj_obs_classes_b_cdict(self, b:str):
		'''
		Used for obs histogram
		'''
		classes = [[self.class_names[self.data[key].y]]*len(self.data[key].get_b(b)) for key in self.data.keys()]
		classes = sum(classes, []) # flat lists
		return classes

	def get_lcobj_obsmean_b_cdict(self, b:str):
		population_dict = self.get_populations_cdict()
		uniques, counts = np.unique(self.get_lcobj_obs_classes_b_cdict(b), return_counts=True)
		return {c:counts[list(uniques).index(c)]/population_dict[c] for c in self.class_names}

	def get_max_length_serial(self):
		return max([len(self.data[key]) for key in self.data.keys()])

	def copy(self,
		data:dict=None,
		):
		new_set = LCSet(
			{key:self.data[key].copy() for key in self.data.keys()} if data is None else data,
			self.survey,
			self.description,
			self.band_names,
			self.class_names,
			self.obs_is_flux,
			)
		return new_set

	def get_min_population(self):
		pop_cdict = self.get_populations()
		min_index = np.argmin([pop_cdict[c] for c in self.class_names])
		min_populated_class = self.class_names[min_index]
		min_population = pop_cdict[min_populated_class]
		obs_len_txt = ' - '.join([f'{b}: {obs_len_dict[b]:,}' for b in self.band_names])
		max_duration = max([lcobj.get_days_serial_duration() for lcobj in self.get_lcobjs()])
		return min_populated_class, min_population

	def get_random_keys(self, nc):
		'''stratified'''
		d = {c:[] for c in self.class_names}
		keys = random.sample(self.data.keys(), len(self.data.keys()))
		index = 0
		while any([len(d[c])<nc for c in self.class_names]):
			key = keys[index]
			obj = self.data[key]
			c = self.class_names[obj.y]
			if len(d[c])<nc:
				d[c].append(key)
			index +=1
		return d

	def get_lcset_values_b(self, b:str, attr:str,
		target_class:str=None,
		):
		values = [getattr(lcobj.get_b(b), attr) for lcobj in self.get_lcobjs() if (target_class is None or target_class==self.class_names[lcobj.y])]
		values = np.concatenate(values, axis=0)
		return values

	def get_lcset_max_value_b(self, b:str, attr,
		target_class=None,
		):
		values = self.get_lcset_values_b(b, attr, target_class)
		return max(values)

	def get_lcset_values_serial(self, attr:str,
		target_class=None,
		):
		'''
		Get values of attr along all bands
		'''
		values = [self.get_lcset_values_b(b, attr, target_class) for b in self.band_names]
		return np.concatenate(values, axis=0)

	def reset_day_offset_serial(self,
		store_day_offset:bool=False,
		):
		'''
		Along all keys
		'''
		for lcobj in self.get_lcobjs():
			lcobj.reset_day_offset_serial(store_day_offset)
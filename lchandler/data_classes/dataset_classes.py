from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import random
import copy

class LCDataset():
	def __init__(self):
		self.raw = None
		self.train = None
		self.val = None
		self.test = None
		self.raw_train = None
		self.raw_val = None
		self.raw_test = None

	def set_custom(self, name:str, lcset):
		setattr(self, name, lcset)
		return lcset

	def set_raw(self, lcset):
		return self.set_custom('raw', lcset)

	def set_raw_train(self, lcset):
		return self.set_custom('raw_train', lcset)

	def set_raw_val(self, lcset):
		return self.set_custom('raw_val', lcset)

	def set_raw_test(self, lcset):
		return self.set_custom('raw_test', lcset)

	def set_train(self, lcset):
		return self.set_custom('train', lcset)

	def set_val(self, lcset):
		return self.set_custom('val', lcset)

	def set_test(self, lcset):
		return self.set_custom('test', lcset)

	def del_set(self, name):
		setattr(self, name, None)

	def get(self, set_name):
		return getattr(self, set_name)

	def __repr__(self):
		txt = 'LCDataset():\n'
		for key in self.__dict__.keys():
			lcset = self.__dict__[key]
			set_text = lcset if not lcset is None else '-'
			txt += f'({key}) - {set_text}\n'
		return txt

	def clean_empty_obs_keys(self):
		'''
		Along all lcsets
		Use this to delete empty light curves: 0 obs in all bands
		'''
		for key in self.__dict__.keys():
			lcset = self.__dict__[key]
			deleted_keys = lcset.clean_empty_obs_keys(verbose=0)
			print(f'({key}) deleted keys: {deleted_keys}')

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

	def clean_empty_obs_keys(self,
		verbose:int=1,
		):
		to_delete_keys = [key for key in self.data.keys() if not np.any([len(self.data[key].get_b(b))>=C_.MIN_POINTS_LIGHTCURVE_DEFINITION for b in self.band_names])]
		deleted_keys = len(to_delete_keys)
		if verbose:
			print('deleted keys:', deleted_keys)
		for key in to_delete_keys:
			self.data.pop(key, None)
		return deleted_keys

	def get_random_key(self):
		keys = list(self.data.keys())
		return keys[random.randint(0, len(keys)-1)]

	def get_random_lcobj(self,
		return_key:bool=True,
		):
		key = self.get_random_key()
		if return_key:
			return self.data[key], key
		return self.data[key]

	def set_diff_parallel(self, attr:str):
		'''
		Along all keys
		'''
		for key in self.data_keys():
			self.data[key].set_diff_parallel(attr)

	def set_log_parallel(self, attr:str):
		'''
		Along all keys
		'''
		for key in self.data_keys():
			self.data[key].set_log_parallel(attr)

	def keys(self):
		return self.__dict__.keys()

	def data_keys(self):
		return list(self.data.keys())

	def get_lcobj_labels(self):
		return [self.data[key].y for key in self.data.keys()]

	def get_lcobj_classes(self):
		'''
		Used for classes histogram
		'''
		return [self.class_names[self.data[key].y] for key in self.data.keys()]

	def get_lcobj_obsmean_classes_b(self, b:str):
		population_dict = self.get_populations()
		uniques, counts = np.unique(self.get_lcobj_obs_classes_b(b), return_counts=True)
		return {c:counts[list(uniques).index(c)]/population_dict[c] for c in self.class_names}

	def get_lcobj_obs_classes_b(self, b:str):
		'''
		Used for obs histogram
		'''
		classes = [[self.class_names[self.data[key].y]]*len(self.data[key].get_b(b)) for key in self.data.keys()]
		classes = sum(classes, []) # flat lists
		return classes

	def __len__(self):
		return len(self.data.keys())

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

	def get_populations(self):
		lcobj_classes = self.get_lcobj_classes()
		uniques, counts = np.unique(lcobj_classes, return_counts=True)
		return {c:counts[list(uniques).index(c)] for c in self.class_names}

	def get_min_population(self):
		pop_cdict = self.get_populations()
		min_index = np.argmin([pop_cdict[c] for c in self.class_names])
		min_populated_class = self.class_names[min_index]
		min_population = pop_cdict[min_populated_class]
		return min_populated_class, min_population

	def __repr__(self):
		obs_len = sum([len(self.data[key]) for key in self.data_keys()])
		obs_len_dict = {b:sum([len(self.data[key].get_b(b)) for key in self.data_keys()]) for b in self.band_names}
		obs_len_txt = ' - '.join([f'{b}: {obs_len_dict[b]:,}' for b in self.band_names])
		max_duration = max([self.data[key].get_days_serial_duration() for key in self.data_keys()])

		txt = f'samples: {len(self):,} - obs samples: {obs_len:,} ({obs_len_txt})'
		txt += f' - max_length_serial: {self.get_max_length_serial()} - max_duration: {max_duration:.2f}'
		#txt += f' - bands: {self.band_names} - classes: {self.class_names} '
		return txt

	def get_random_keys(self, nc): # stratified
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
		keys = self.data_keys()
		values = [getattr(self.data[key].get_b(b), attr) for key in keys if (target_class is None or target_class==self.class_names[self.data[key].y])]
		values = np.concatenate(values, axis=0)#.astype(values[0].dtype)
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
		return np.concatenate(values, axis=0)#.astype(values[0].dtype)

	def reset_day_offset_serial(self,
		store_day_offset:bool=False,
		):
		'''
		Along all keys
		'''
		for key in self.data_keys():
			self.data[key].reset_day_offset_serial(store_day_offset)
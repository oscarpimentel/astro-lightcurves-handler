from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import random
import copy
import flamingchoripan.datascience.statistics as fstats
from flamingchoripan.prints import HiddenPrints, ShowPrints
from flamingchoripan.strings import get_bar
from flamingchoripan.level_bars import LevelBar
from .lc_classes import diff_vector
import pandas as pd

###################################################################################################################################################

def search_over_sigma_samples(lcset, b:str, dist_mean, dist_sigma, sigma_m,
	apply_lower_bound:bool=True,
	):
	total_deleted_points = 0
	for lcobj_name in lcset.get_lcobj_names():
		sigmas = lcset[lcobj_name].get_b(b).obse # get
		valid_indexs = fstats.get_sigma_clipping_indexing(sigmas, dist_mean, dist_sigma, sigma_m, apply_lower_bound)
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
		txt = 'LCDataset:\n'
		for lcset_name in self.get_lcset_names():
			txt += f'[{lcset_name} - samples {len(self[lcset_name]):,}]\n{self[lcset_name]}\n'
			txt += get_bar()+'\n'
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
		sum_ = sum([new_lcsets[k] for k in new_lcsets.keys()])
		assert abs(1-sum_)<=C_.EPS
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

	def get_serial_stats_idf(self,
		lcset_names=None,
		):
		dfs = []
		lcset_names = self.get_lcset_names() if lcset_names is None else lcset_names
		for lcset_name in lcset_names:
			df = self[lcset_name].get_serial_stats_idf(lcset_name)
			dfs.append(df)
		return pd.concat(dfs)

	def get_bstats_idf(self, b,
		lcset_names=None,
		):
		dfs = []
		lcset_names = self.get_lcset_names() if lcset_names is None else lcset_names
		for lcset_name in lcset_names:
			df = self[lcset_name].get_bstats_idf(b, lcset_name)
			dfs.append(df)
		return pd.concat(dfs)

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

	def get_lcobj_names(self,
		c=None,
		):
		return [k for k in self.data.keys() if self.class_names[self.data[k].y]==c or c is None]

	def get_lcobjs(self,
		c=None,
		):
		return [self[lcobj_name] for lcobj_name in self.get_lcobj_names(c)]

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
		return fstats.get_populations_cdict(self.get_lcobj_classes(), self.class_names)

	def get_class_freq_weights_cdict(self):
		pop_cdict = self.get_populations_cdict()
		total_pop = sum([pop_cdict[c] for c in self.class_names])
		return {c:total_pop/pop_cdict[c] for c in self.class_names}

	def get_class_brfc_weights_cdict(self):
		# used for BalancedRandomForestClassifier
		pop_cdict = self.get_populations_cdict()
		total_pop = sum([pop_cdict[c] for c in self.class_names])
		return {c:total_pop/(len(self.class_names)*pop_cdict[c]) for c in self.class_names}

	def get_class_efective_weigths_cdict(self, beta):
		assert beta>0 and beta<1
		return {c:(1-beta**pop_cdict[c])/(1-beta) for c in self.class_names}

	def get_mean_length_df_bdict(self,
		index=None,
		):
		df_bdict = {}
		for kb,b in enumerate(self.band_names):
			info_dict = {}
			for kc,c in enumerate(self.class_names):
				lcobjs = self.get_lcobjs(c)
				info_dict[f'{c}{b}-$N_c$'] = sum([len(lcobj.get_b(b)) for lcobj in lcobjs])/len(lcobjs)
			df = pd.DataFrame.from_dict({id(self) if index is None else index:info_dict}, orient='index')
			df.index.rename(C_.SET_NAME_STR, inplace=True)
			df_bdict[b] = df

		return df_bdict

	def get_class_stats_idf(self,
		index=None,
		):
		info_dict = {}
		for kc,c in enumerate(self.class_names):
			lcobjs = self.get_lcobjs(c)
			info_dict[f'{c}-$N_c$'] = len(lcobjs)
		df = pd.DataFrame.from_dict({id(self) if index is None else index:info_dict}, orient='index')
		df.index.rename(C_.SET_NAME_STR, inplace=True)
		return df, self.get_mean_length_df_bdict()

	def get_serial_stats_idf(self,
		index=None,
		):
		info_dict = {}
		for kc,c in enumerate(self.class_names):
			lcobjs = self.get_lcobjs(c)
			xs = [lcobj.get_x_serial() for lcobj in lcobjs]
			info_dict[f'{c}-$x$'] = fstats.XError(np.concatenate([x[:,C_.OBS_INDEX] for x in xs]))
			info_dict[f'{c}-$L$'] = fstats.XError([len(lcobj) for lcobj in lcobjs])
			info_dict[f'{c}-$\Delta T$'] = fstats.XError([lcobj.get_days_serial_duration() for lcobj in lcobjs])
			info_dict[f'{c}-$\Delta t$'] = fstats.XError(np.concatenate([diff_vector(x[:,C_.DAYS_INDEX]) for x in xs]))
		df = pd.DataFrame.from_dict({id(self) if index is None else index:info_dict}, orient='index')
		df.index.rename(C_.SET_NAME_STR, inplace=True)
		return df

	def get_bstats_idf_c(self, c, b,
		index=None,
		):
		info_dict = {}
		lcobjs = self.get_lcobjs(c)
		info_dict[f'{c}-$x$'] = fstats.XError(np.concatenate([x.get_b(b).obs for x in lcobjs]))
		info_dict[f'{c}-$L$'] = fstats.XError([len(x.get_b(b)) for x in lcobjs])
		info_dict[f'{c}-$\Delta T$'] = fstats.XError([x.get_b(b).get_days_duration() for x in lcobjs if len(x.get_b(b))>=1])
		info_dict[f'{c}-$\Delta t$'] = fstats.XError(np.concatenate([x.get_b(b).get_diff('days') for x in lcobjs]))
		df = pd.DataFrame.from_dict({id(self) if index is None else index:info_dict}, orient='index')
		df.index.rename(C_.SET_NAME_STR, inplace=True)
		return df

	def get_bstats_idf(self, b,
		index=None,
		):
		dfs = []
		for kc,c in enumerate(self.class_names):
			df = self.get_bstats_idf_c(c, b, index)
			dfs.append(df)
		df = pd.concat(dfs, axis=1)
		return df

	### repr
	def __repr__b(self, b):
		df = self.get_bstats_idf(b)
		lengths = sum([df[f'{c}-$L$'].values[0] for c in self.class_names])
		durations = sum([df[f'{c}-$\Delta T$'].values[0] for c in self.class_names])
		cadences = sum([df[f'{c}-$\Delta t$'].values[0] for c in self.class_names])
		txt = f'({b}) obs_samples: {lengths.sum():,} - min_len: {lengths.min()} - max_dur: {durations.max():.1f}[days] - dur(p50): {durations.p50:.1f}[days] - cadence(p50): {cadences.p50:.1f}[days]\n'
		return txt

	def __repr_serial(self):
		df = self.get_serial_stats_idf()
		lengths = sum([df[f'{c}-$L$'].values[0] for c in self.class_names])
		durations = sum([df[f'{c}-$\Delta T$'].values[0] for c in self.class_names])
		cadences = sum([df[f'{c}-$\Delta t$'].values[0] for c in self.class_names])
		txt = f'(*) obs_samples: {lengths.sum():,} - min_len: {lengths.min()} - max_dur: {durations.max():.1f}[days] - dur(p50): {durations.p50:.1f}[days] - cadence(p50): {cadences.p50:.1f}[days]\n'
		return txt

	def __repr__(self):
		if len(self)>0:
			txt = self.__repr_serial()
			for b in self.band_names:
				txt += self.__repr__b(b)
			populations_cdict = self.get_populations_cdict()
			txt += str(LevelBar(populations_cdict, ' '*3))
		else:
			txt = 'empty lcset\n'
		return txt[:-1]

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
		pop_cdict = self.get_populations_cdict()
		min_index = np.argmin([pop_cdict[c] for c in self.class_names])
		min_populated_class = self.class_names[min_index]
		min_population = pop_cdict[min_populated_class]
		return min_populated_class, min_population

	def get_random_stratified_keys(self, nc):
		return fstats.get_random_stratified_keys(self.get_lcobj_names(), self.get_lcobj_classes(), self.class_names, nc)

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
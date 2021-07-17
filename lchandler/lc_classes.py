from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import random
from scipy.stats import t
from copy import copy, deepcopy
from fuzzytools import numba as ftnumba

DF = 2 # 1 2 5 np.inf
OBSE_STD_SCALE = 1/2
OBS_NOISE_RANGE = 1

###################################################################################################################################################

def diff_vector(x,
	uses_prepend=True,
	):
	if len(x)==0:
		return x
	if uses_prepend:
		new_x = np.concatenate([x[0][None,...], x], axis=0)
	else:
		new_x = x
	dx = new_x[1:]-new_x[:-1]
	return dx

def get_new_noisy_obs(obs, obse, obs_min_lim,
	std_scale=OBSE_STD_SCALE,
	df=DF,
	obs_noise_range=OBS_NOISE_RANGE,
	):
	assert df>=0
	dtype = obs.dtype
	std = obse*std_scale
	if df==np.inf:
		new_obs = np.random.standard_normal(size=len(obs)).astype(dtype)*std+obs
	else:
		new_obs = np.random.standard_t(df, size=len(obs)).astype(dtype)*std+obs

	bar_size = (1.645*2)*obse # for .95 percentile used in plot
	min_lim = obs-bar_size*obs_noise_range/2
	max_lim = obs+bar_size*obs_noise_range/2
	new_obs = np.clip(new_obs, min_lim, max_lim)
	new_obs = np.clip(new_obs, obs_min_lim, None)
	return new_obs

###################################################################################################################################################

class SubLCO():
	'''
	Dataclass object used to store an astronomical light curve
	'''
	def __init__(self, days, obs, obse,
		y:int=None,
		dtype=np.float32,
		):
		self.days = days
		self.obs = obs
		self.obse = obse
		self.y = y
		self.dtype = dtype
		self.reset()

	def reset(self):
		self.set_values(self.days, self.obs, self.obse)
		self.set_synthetic_mode(None)

	def get_synthetic_mode(self):
		return self.synthetic_mode

	def set_synthetic_mode(self, synthetic_mode):
		self.synthetic_mode = synthetic_mode

	def is_synthetic(self):
		return not self.synthetic_mode is None

	def set_values(self, days, obs, obse):
		'''
		Always use this method to set new values!
		'''
		assert len(days)==len(obs)
		assert len(days)==len(obse)
		if isinstance(days, np.ndarray):
			tdays = copy(days).astype(self.dtype)
			tobs = copy(obs).astype(self.dtype)
			tobse = copy(obse).astype(self.dtype)
		else:
			tdays = np.array(days, dtype=self.dtype)
			tobs = np.array(obs, dtype=self.dtype)
			tobse = np.array(obse, dtype=self.dtype)

		self._set_days(tdays)
		self._set_obs(tobs)
		self._set_obse(tobse)

	def _set_days(self, days):
		assert len(days.shape)==1
		if C_.CHECK:
			assert np.all((diff_vector(days, uses_prepend=False)>0)) # C_.check if days are in order
		self.days = days

	def _set_obs(self, obs):
		assert len(obs.shape)==1
		if C_.CHECK:
			assert np.all(obs>=0)
		self.obs = obs

	def _set_obse(self, obse):
		assert len(obse.shape)==1
		if C_.CHECK:
			assert np.all(obse>=0)
		self.obse = obse

	def add_day_values(self, values,
		recalculate_order:bool=True,
		):
		'''
		This method overrides information!
		Always use this method to add values
		calcule d_days again
		'''
		assert len(self)==len(values)
		new_days = self.days+values
		valid_indexs = np.argsort(new_days) # must sort before the values to mantain sequenciality
		self.days = new_days # bypass _set_days() because non-sorted asumption
		self.apply_valid_indexs_to_attrs(valid_indexs, recalculate_order) # apply valid indexs to all

	def add_day_noise_uniform(self, hours_noise:float,
		recalculate_order:bool=True,
		):
		'''
		This method overrides information!
		'''
		if hours_noise==0:
			return

		hours_noise = np.Floatndarray(len(self)).uniform_(-hours_noise, hours_noise)
		self.add_day_values(hours_noise/24.,
			recalculate_order,
			)

	def add_obs_values(self, values):
		'''
		This method overrides information!
		Always use this method to add values
		calcule d_obs again
		'''
		assert len(self)==len(values)
		new_obs = self.obs+values
		self._set_obs(new_obs)

	def add_obs_noise_gaussian(self, obs_min_lim:float,
		std_scale=OBSE_STD_SCALE,
		df=DF,
		obs_noise_range=OBS_NOISE_RANGE,
		):
		'''
		This method overrides information!
		'''
		if std_scale==0:
			return
		obs_values = get_new_noisy_obs(self.obs, self.obse, obs_min_lim,
			std_scale,
			df,
			obs_noise_range,
			)
		self.add_obs_values(obs_values-self.obs)
		return

	def apply_downsampling_window(self, mode_d, ds_prob,
		min_valid_length:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		recalculate_order:bool=True,
		min_frac=1/3,
		):
		if len(self)<=min_valid_length:
			return
		if mode_d is None or len(mode_d)==0:
			mode_d = {'none':1}
		keys = list(mode_d.keys())
		mode = np.random.choice(keys, p=[mode_d[k] for k in keys])

		### mask
		valid_mask = np.zeros((len(self)), dtype=np.bool)
		min_length = max(min_valid_length, int(min_frac*len(self)))
		if mode=='none':
			valid_mask[:] = True

		elif mode=='left':
			new_length = random.randint(min_length, len(self)) # [a,b]
			valid_mask[:new_length] = True

		elif mode=='random':
			new_length = random.randint(min_length, len(self)) # [a,b]
			index = random.randint(0, len(self)-new_length) # [a,b]
			valid_mask[index:index+new_length] = True
		else:
			raise Exception(f'no mode {mode}')

		assert ds_prob>=0 and ds_prob<=1
		if ds_prob>0:
			#p = np.full((len(self),), fill_value=1-ds_prob)
			ber_valid_mask = ftnumba.bernoulli(1-ds_prob, len(self))
			valid_mask = valid_mask & ber_valid_mask 

		if valid_mask.sum()<min_length: # extra case. If by change the mask implies a very short curve
			valid_mask = np.zeros((len(self)), dtype=np.bool)
			valid_mask[:min_length] = True
			valid_mask = valid_mask[np.random.permutation(len(valid_mask))]

		### calcule again as the original values changed
		self.apply_valid_indexs_to_attrs(valid_mask, recalculate_order)
		return

	def get_diff(self, attr:str):
		return diff_vector(getattr(self, attr))

	def set_diff(self, attr:str):
		'''
		Calculate a diff version from an attr and create a new attr with new name
		'''
		diffv = self.get_diff(attr)
		setattr(self, f'd_{attr}', diffv)

	def apply_valid_indexs_to_attrs(self, valid_indexs,
		recalculate_order:bool=True,
		):
		'''
		Be careful, this method can remove info
		calcule d_days again
		calcule d_obs again
		fixme: this function is not opimized... specially due the d_days and that kind of variables
		'''
		original_len = len(self)
		for key in self.__dict__.keys():
			x = self.__dict__[key]
			if isinstance(x, np.ndarray): # apply same mask to all in the object
				assert len(x.shape)==1 # 1D
				assert original_len==len(x), f'{key} {original_len}=={len(x)}'
				new_x = x[valid_indexs]
				setattr(self, key, new_x)

		### calcule again as the original values changed
		if recalculate_order:
			if hasattr(self, 'd_days'):
				self.set_diff('days')
			if hasattr(self, 'd_obs'):
				self.set_diff('obs')

	def get_valid_indexs_max_day(self, max_day:float,
		remove_offset=False,
		):
		offset = self.days[0] if remove_offset else 0
		return self.days-offset<=max_day

	def clip_attrs_given_max_day(self, max_day:float,
		remove_offset=False,
		):
		'''
		Be careful, this method remove info!
		'''
		valid_indexs = self.get_valid_indexs_max_day(max_day, remove_offset)
		self.apply_valid_indexs_to_attrs(valid_indexs)

	def get_valid_indexs_max_duration(self, max_duration:float):
		return self.get_valid_indexs_max_day(max_duration, True)

	def clip_attrs_given_max_duration(self, max_duration:float):
		'''
		Be careful, this method remove info!
		'''
		self.clip_attrs_given_max_day(max_duration, True)

	def get_x(self):
		attrs = ['days', 'obs', 'obse']
		return self.get_custom_x(attrs)

	def get_attr(self, attr:str):
		return getattr(self, attr)

	def get_custom_x(self, attrs:list):
		values = [self.get_attr(attr)[...,None] for attr in attrs]
		x = np.concatenate(values, axis=-1)
		return x

	def get_first_day(self):
		return self.days[0]

	def get_last_day(self):
		return self.days[-1]

	def get_days_duration(self):
		return self.get_last_day()-self.get_first_day() if len(self)>0 else None

	def keys(self):
		return self.__dict__.keys()

	def copy(self):
		return copy(self)

	def __copy__(self):
		new_sublco = SubLCO(
			copy(self.days),
			copy(self.obs),
			copy(self.obse),
			self.y,
			self.dtype,
			)
		new_sublco.set_synthetic_mode(self.get_synthetic_mode())

		for key in self.__dict__.keys():
			if key in ['days', 'obs', 'obse']:
				continue
			v = self.__dict__[key]
			if isinstance(v, np.ndarray):
				setattr(new_sublco, key, copy(v))
		return new_sublco

	def __len__(self):
		l = len(self.days)
		assert l==len(self.obs)
		assert l==len(self.obse)
		return l

	def __repr__(self):
		txt = f'[d:{self.days}{self.days.dtype}'
		txt += f', o:{self.obs}{self.obs.dtype}'
		txt += f', oe:{self.obse}{self.obse.dtype}]'
		return txt

	def clean_small_cadence(self,
		dt=C_.CADENCE_THRESHOLD,
		mode='expectation',
		):
		ddict = {}
		i = 0
		while i<len(self.days):
			day = self.days[i]
			valid_indexs = np.where((self.days>=day) & (self.days<day+dt))[0]
			ddict[day] = valid_indexs
			i += len(valid_indexs)

		new_days = []
		new_obs = []
		new_obse = []
		for k in ddict.keys():
			if mode=='mean':
				new_days.append(np.mean(self.days[ddict[k]]))
				new_obs.append(np.mean(self.obs[ddict[k]]))
				new_obse.append(np.mean(self.obse[ddict[k]]))
			elif mode=='min_obse':
				i = np.argmin(self.obse[ddict[k]])
				new_days.append(self.days[ddict[k]][i])
				new_obs.append(self.obs[ddict[k]][i])
				new_obse.append(self.obse[ddict[k]][i])
			elif mode=='expectation':
				obse_exp = np.exp(-np.log(self.obse[ddict[k]]+C_.EPS))
				assert len(np.where(obse_exp==np.inf)[0])==0
				#print(obse_exp, obse_exp.shape)
				dist = obse_exp/obse_exp.sum()
				new_days.append(np.sum(self.days[ddict[k]]*dist))
				new_obs.append(np.sum(self.obs[ddict[k]]*dist))
				new_obse.append(np.sum(self.obse[ddict[k]]*dist))
			else:
				raise Exception(f'no mode {mode}')

		self.set_values(new_days, new_obs, new_obse)

	def get_snr(self,
		eps=1e-10,
		max_len=None,
		):
		if len(self)==0:
			return np.nan
		else:
			max_len = len(self) if max_len is None else max_len
			snr = (self.obs[:max_len]**2)/(self.obse[:max_len]**2+eps)
			return np.mean(snr)

	def get_max(self):
		if len(self)==0:
			return np.nan
		else:
			return np.max(self.obs)

	def get_tmax(self):
		if len(self)==0:
			return np.nan
		else:
			tmax = self.days[np.argmax(self.obs)]
			return tmax

	def __add__(self, other):
		if other==0 or other is None:
			return copy(self)
		elif self==0 or self is None:
			return copy(other)
		else:
			new_days = np.concatenate([self.days, other.days], axis=0)
			new_obs = np.concatenate([self.obs, other.obs], axis=0)
			new_obse = np.concatenate([self.obse, other.obse], axis=0)
			valid_indexs = np.argsort(new_days)
			new_lco = SubLCO(
				new_days[valid_indexs],
				new_obs[valid_indexs],
				new_obse[valid_indexs],
				self.y,
				self.dtype,
				)
			return new_lco

	def __radd__(self, other):
		return self+other

	def astype(self, dtype):
		self.dtype = dtype
		for key in self.__dict__.keys():
			x = self.__dict__[key]
			if isinstance(x, np.ndarray): # apply same mask to all in the object
				new_x = x.astype(self.dtype)
				setattr(self, key, new_x)
		return self

###################################################################################################################################################

class LCO():
	'''
	Dataclass object used to store a multiband astronomical light curve
	'''
	def __init__(self,
		is_flux:bool=True,
		y:int=None,
		global_first_day:int=0,
		ra:float=None,
		dec:float=None,
		z:float=None,
		):
		self.is_flux = is_flux
		self.set_y(y)
		self.global_first_day = global_first_day
		self.ra = ra
		self.dec = dec
		self.z = z
		self.reset()

	def reset(self):
		self.bands = []

	def add_b(self, b:str, days, obs, obse):
		'''
		Always use this method
		'''
		sublcobj = SubLCO(days, obs, obse, self.y)
		self.add_sublcobj_b(b, sublcobj)

	def add_sublcobj_b(self, b:str, sublcobj):
		setattr(self, b, sublcobj)
		if not b in self.bands:
			self.bands += [b]

	def copy_only_data(self):
		new_lco = LCO(
			is_flux=self.is_flux,
			y=self.y,
			global_first_day=self.global_first_day,
			ra=self.ra,
			dec=self.dec,
			z=self.z,
		)
		return new_lco

	def copy(self):
		return copy(self)

	def __copy__(self):
		new_lco = LCO(
			is_flux=self.is_flux,
			y=self.y,
			global_first_day=self.global_first_day,
			ra=self.ra,
			dec=self.dec,
			z=self.z,
		)
		for b in self.bands:
			new_sublcobj = copy(self.get_b(b))
			new_lco.add_sublcobj_b(b, new_sublcobj)
		return new_lco

	def set_y(self, y:int):
		'''
		Always use this method
		'''
		self.y = None if y is None else int(y)

	def keys(self):
		return self.get_b(self.bands[0]).keys()

	def __repr__(self):
		txt = ''
		for b in self.bands:
			obj = self.get_b(b)
			txt += f'({b}:{len(obj)}) - {obj.__repr__()}\n'
		return txt

	def __len__(self):
		return sum([len(self.get_b(b)) for b in self.bands])

	#########  serial/multiband important methods
	def reset_day_offset_serial(self,
		store_day_offset:bool=False,
		return_day_offset:bool=False,
		bands=None,
		):
		'''
		delete day offset acording to the first day along any day!
		'''
		bands = self.bands if bands is None else bands
		first_days = [self.get_b(b).get_first_day() for b in bands if len(self.get_b(b))>0]
		assert len(first_days)>0
		day_offset = min(first_days) # select the min along all bands
		for b in bands:
			self.get_b(b).days = self.get_b(b).days-day_offset
		if store_day_offset:
			self.global_first_day = day_offset
		if return_day_offset:
			return self, day_offset
		return self

	def get_sorted_days_indexs_serial(self,
		bands=None,
		):
		bands = self.bands if bands is None else bands
		values = [self.get_b(b).days for b in bands]
		all_days = np.concatenate(values, axis=0)
		sorted_days_indexs = np.argsort(all_days)
		return sorted_days_indexs

	def get_onehot_serial(self,
		bands=None,
		):
		bands = self.bands if bands is None else bands
		onehot = np.zeros((len(self), len(bands)), dtype=np.bool)
		index = 0
		for kb,b in enumerate(bands):
			l = len(getattr(self, b))
			onehot[index:index+l,kb] = True
			index += l
		sorted_days_indexs = self.get_sorted_days_indexs_serial(bands)
		onehot = onehot[sorted_days_indexs]
		return onehot

	def get_custom_x_serial(self, attrs:list,
		bands=None,
		):
		bands = self.bands if bands is None else bands
		values = [self.get_b(b).get_custom_x(attrs) for b in bands]
		x = np.concatenate(values, axis=0)
		sorted_days_indexs = self.get_sorted_days_indexs_serial(bands)
		x = x[sorted_days_indexs]
		return x

	def get_x_serial(self,
		bands=None,
		):
		bands = self.bands if bands is None else bands
		return self.get_custom_x_serial(['days', 'obs', 'obse'],
			bands,
			)

	def get_days_serial(self,
		bands=None,
		):
		bands = self.bands if bands is None else bands
		return self.get_custom_x_serial(['days'],
			bands,
			)[:,0]

	def get_days_serial_duration(self,
		bands=None,
		):
		'''
		Duration in days of complete light curve
		'''
		bands = self.bands if bands is None else bands
		days = np.concatenate([self.get_b(b).days for b in bands], axis=0)
		duration = np.max(days)-np.min(days)
		return duration

	def set_diff_b(self, b:str, attr:str):
		self.get_b(b).set_diff(attr) 

	def set_diff_parallel(self, attr:str,
		bands=None,
		):
		'''
		Along all bands
		'''
		bands = self.bands if bands is None else bands
		for b in bands:
			self.set_diff_b(b, attr)

	def get_b(self, b:str):
		return getattr(self, b)

	def get_bands(self):
		return self.bands

	def get_length_b(self, b:str):
		return len(self.get_b(b))

	def get_length_bdict(self):
		return {b:self.get_length_b(b) for b in self.bands}

	def any_synthetic(self):
		return any([self.get_b(b).is_synthetic() for b in self.bands])

	def all_synthetic(self):
		return all([self.get_b(b).is_synthetic() for b in self.bands])

	def any_real(self):
		return any([not self.get_b(b).is_synthetic() for b in self.bands])

	def all_real(self):
		return all([not self.get_b(b).is_synthetic() for b in self.bands])

	def any_band_eqover_length(self,
		th_length=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		):
		return any([len(self.get_b(b))>=th_length for b in self.bands])

	def clean_small_cadence(self,
		dt=C_.CADENCE_THRESHOLD,
		mode='expectation',
		):
		for b in self.bands:
			self.get_b(b).clean_small_cadence(dt, mode)

	def get_snr(self):
		snr_d = {b:self.get_b(b).get_snr() for b in self.bands}
		snr_max = np.nanmax([snr_d[b] for b in self.bands])
		return snr_max

	def get_tmax(self):
		tmax_d = {b:self.get_b(b).get_tmax() for b in self.bands}
		return tmax_d

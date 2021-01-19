from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import random
import copy
from scipy.stats import t

###################################################################################################################################################

def diff_vector(x:np.ndarray,
	uses_prepend=True,
	):
	if len(x)==0:
		return x
	x = x[...,None]
	if uses_prepend:
		to_append = np.expand_dims(x[0,...], axis=1)
		dx = np.diff(x, axis=0, prepend=to_append.T)
		return dx[:,0]
	else:
		dx = np.diff(x, axis=0)
		return dx[:,0]

def log_vector(x:np.ndarray):
	assert np.all(x>=0)
	return np.log(x+1)

def inv_log_vector(x:np.ndarray):
	return np.exp(x)-1

def get_obs_noise_gaussian(obs, obse, obs_min_lim,
	std_scale:float=C_.OBSE_STD_SCALE,
	mode='norm',
	):
	if mode=='norm':
		obs_values = np.clip(np.random.normal(obs, obse*std_scale), obs_min_lim, None)
	elif mode=='t':
		t_rvs = t.rvs(5, obs, obse*std_scale)
		obs_values = np.clip(t_rvs, obs_min_lim, None)
	else:
		raise Exception(f'no mode {mode}')
	return obs_values

###################################################################################################################################################

class SubLCO():
	'''
	Dataclass object used to store an astronomical light curve
	'''
	def __init__(self, days:np.ndarray, obs:np.ndarray, obs_errors:np.ndarray,
		y:int=None,
		):
		self.set_values(days, obs, obs_errors)
		self.y = y
		self.synthetic = False

	def set_values(self, days:np.ndarray, obs:np.ndarray, obs_errors:np.ndarray):
		'''
		Always use this method to set new values!
		'''
		assert len(days)==len(obs)
		assert len(obs)==len(obs_errors)
		self.set_days(days)
		self.set_obs(obs)
		self.set_obse(obs_errors)

	def set_days(self, days):
		assert len(days.shape)==1
		assert np.all((diff_vector(days, uses_prepend=False)>0)) # check if days are in order
		self.days = days

	def set_obs(self, obs):
		assert len(obs.shape)==1
		assert np.all(obs>=0)
		self.obs = obs

	def set_obse(self, obs_errors):
		assert len(obs_errors.shape)==1
		if not np.all(obs_errors>0):
			raise Exception(f'wrong obs_errors: {obs_errors}')
		self.obse = obs_errors

	def add_day_values(self, values,
		recalculate:bool=True,
		):
		'''
		This method overrides information!
		Always use this method to add values
		calcule d_days again
		calcule log_days again
		'''
		assert len(self)==len(values)
		new_days = self.days+values
		valid_indexs = np.argsort(new_days) # must sort before the values to mantain sequenciality
		self.days = new_days # bypass set_days() because non-sorted asumption
		self.apply_valid_indexs_to_attrs(valid_indexs) # apply valid indexs to all

		### calcule again as the original values changed
		if recalculate:
			if hasattr(self, 'd_days'): 
				self.set_diff('days')
			if hasattr(self, 'log_days'):
				self.set_log('days')

	def add_day_noise_uniform(self, hours_noise:float,
		recalculate:bool=True,
		):
		'''
		This method overrides information!
		'''
		hours_noise = np.random.uniform(-hours_noise, hours_noise, size=len(self))
		self.add_day_values(hours_noise/24., recalculate)

	def add_obs_values(self, values,
		recalculate:bool=True,
		):
		'''
		This method overrides information!
		Always use this method to add values
		calcule d_obs again
		calcule log_obs again
		'''
		assert len(self)==len(values)
		new_obs = self.obs+values
		self.set_obs(new_obs)

		###  calcule again as the original values changed
		if recalculate:
			if hasattr(self, 'd_obs'):
				self.set_diff('obs')
			if hasattr(self, 'log_obs'):
				self.set_log('obs')

	def add_obs_noise_gaussian(self, obs_min_lim:float,
		std_scale:float=C_.OBSE_STD_SCALE,
		mode='norm',
		recalculate:bool=True,
		):
		'''
		This method overrides information!
		'''
		assert np.all(self.obs>=obs_min_lim)
		obs_values = get_obs_noise_gaussian(self.obs, self.obse, obs_min_lim, std_scale, mode)
		self.add_obs_values(obs_values-self.obs, recalculate)

	def apply_downsampling(self, ds_prob,
		min_valid_length:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		recalculate:bool=True,
		):
		assert ds_prob>=0 and ds_prob<=1
		if len(self)<=min_valid_length or ds_prob==0:
			return

		valid_mask = np.random.uniform(0, 1, len(self))>ds_prob
		if valid_mask.sum()<min_valid_length:
			valid_mask = valid_mask*0
			valid_mask[:min_valid_length] = 1
			valid_mask = np.random.permutation(valid_mask.astype(bool))

		### calcule again as the original values changed
		self.apply_valid_indexs_to_attrs(valid_mask, recalculate)

	def get_diff(self, attr:str):
		return diff_vector(getattr(self, attr))

	def set_diff(self, attr:str):
		'''
		Calculate a diff version from an attr and create a new attr with new name
		'''
		diffv = self.get_diff(attr)
		setattr(self, f'd_{attr}', diffv)

	def get_log(self, attr:str):
		return log_vector(getattr(self, attr))

	def set_log(self, attr:str):
		'''
		Calculate a lof version from an attr and create a new attr with new name
		'''
		logv = self.get_log(attr)
		setattr(self, f'log_{attr}', logv)

	def apply_valid_indexs_to_attrs(self, valid_indexs,
		recalculate:bool=True,
		):
		'''
		Be careful, this method can remove info
		calcule d_days again
		calcule d_obs again
		'''
		original_len = len(self)
		for key in self.__dict__.keys():
			x = self.__dict__[key]
			if isinstance(x, np.ndarray): # apply same mask to all np.ndarray in the object
				assert original_len==len(x)
				assert len(x.shape)==1 # 1D tensor
				setattr(self, key, x[valid_indexs])

		### calcule again as the original values changed
		if recalculate:
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
		new_sublco = SubLCO(
			self.days.copy(),
			self.obs.copy(),
			self.obse.copy(),
			self.y,
			)
		for key in self.__dict__.keys():
			if key in ['days', 'obs', 'obse']:
				continue
			v = self.__dict__[key]
			if isinstance(v, np.ndarray):
				setattr(new_sublco, key, v.copy())
		return new_sublco

	def synthetic_copy(self):
		new_obj = self.copy()
		new_obj.synthetic = True
		return new_obj

	def __len__(self):
		return len(self.days)

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
				assert len(np.where(obse_exp==np.infty)[0])==0
				#print(obse_exp, obse_exp.shape)
				dist = obse_exp/obse_exp.sum()
				new_days.append(np.sum(self.days[ddict[k]]*dist))
				new_obs.append(np.sum(self.obs[ddict[k]]*dist))
				new_obse.append(np.sum(self.obse[ddict[k]]*dist))
			else:
				raise Exception(f'no mode {mode}')

		self.set_values(np.array(new_days), np.array(new_obs), np.array(new_obse))

	def get_snr(self):
		if len(self)==0:
			return -np.infty
		else:
			snr = (self.obs**2)/(self.obse**2)
			return np.mean(snr)

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
		self.bands = []

	def add_b(self, b:str, days:np.ndarray, obs:np.ndarray, obs_errors:np.ndarray):
		'''
		Always use this method
		'''
		sublcobj = SubLCO(days, obs, obs_errors, self.y)
		self.add_sublcobj_b(b, sublcobj)

	def add_sublcobj_b(self, b:str, sublcobj):
		setattr(self, b, sublcobj)
		self.bands.append(b)

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
		new_lco = LCO(
			is_flux=self.is_flux,
			y=self.y,
			global_first_day=self.global_first_day,
			ra=self.ra,
			dec=self.dec,
			z=self.z,
		)
		for b in self.bands:
			new_sublcobj = self.get_b(b).copy()
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
		):
		'''
		delete day offset acording to the first day along any day!
		'''
		first_days = [self.get_b(b).get_first_day() for b in self.bands if len(self.get_b(b))>0]
		if len(first_days)==0:
			return # do nothing
		day_offset = min(first_days) # select the min along all bands
		for b in self.bands:
			self.get_b(b).days -= day_offset
		if store_day_offset:
			self.global_first_day = day_offset
			
		if return_day_offset:
			return self, day_offset
		return self

	def get_sorted_days_indexs_serial(self):
		values = [self.get_b(b).days for b in self.bands]
		all_days = np.concatenate(values, axis=0)
		sorted_days_indexs = np.argsort(all_days)
		return sorted_days_indexs

	def get_onehot_serial(self,
		sorted_days_indexs=None,
		max_day:float=np.infty,
		):
		onehot = np.zeros((len(self), len(self.get_bands())), dtype=np.bool)
		index = 0
		for kb,b in enumerate(self.bands):
			l = len(getattr(self, b))
			onehot[index:index+l,kb] = 1.
			index += l
		sorted_days_indexs = self.get_sorted_days_indexs_serial() if sorted_days_indexs is None else sorted_days_indexs
		onehot = onehot[sorted_days_indexs]
		
		### clip by max day
		if not (max_day==np.infty or max_day is None):
			valid_indexs = self.get_days_serial() <= max_day
			onehot = onehot[valid_indexs]
		return onehot

	def get_x_serial(self,
		sorted_days_indexs=None,
		max_day:float=np.infty,
		):
		attrs = ['days', 'obs', 'obse']
		return self.get_custom_x_serial(attrs, sorted_days_indexs, max_day)

	def get_custom_x_serial(self, attrs:list,
		sorted_days_indexs=None,
		max_day:float=np.infty,
		):
		values = [self.get_b(b).get_custom_x(attrs) for b in self.bands]
		x = np.concatenate(values, axis=0)
		sorted_days_indexs = self.get_sorted_days_indexs_serial() if sorted_days_indexs is None else sorted_days_indexs
		x = x[sorted_days_indexs]

		### clip by max day
		if not (max_day==np.infty or max_day is None):
			valid_indexs = self.get_days_serial() <= max_day
			x = x[valid_indexs]
		return x

	def get_days_serial(self,
		sorted_days_indexs=None,
		):
		return self.get_custom_x_serial(['days'], sorted_days_indexs)[:,0]

	def get_days_serial_duration(self,
		sorted_days_indexs=None,
		):
		'''
		Duration in days of complete light curve
		'''
		days = np.concatenate([self.get_b(b).days for b in self.bands])
		return max(days)-min(days)

	def set_diff_b(self, b:str, attr:str):
		self.get_b(b).set_diff(attr) 

	def set_diff_parallel(self, attr:str):
		'''
		Along all bands
		'''
		for b in self.bands:
			self.set_diff_b(b, attr)

	def set_log_b(self, b:str, attr:str):
		self.get_b(b).set_log(attr) 

	def set_log_parallel(self, attr:str):
		'''
		Along all bands
		'''
		for b in self.bands:
			self.set_log_b(b, attr)

	def get_b(self, b:str):
		return getattr(self, b)

	def get_bands(self):
		return self.bands

	def get_length_b(self, b:str):
		return len(self.get_b(b))

	def get_length_bdict(self):
		return {b:self.get_length_b(b) for b in self.bands}

	def any_synthetic(self):
		return any([self.get_b(b).synthetic for b in self.bands])

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
		return max([self.get_b(b).get_snr() for b in self.bands])
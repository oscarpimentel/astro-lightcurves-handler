from __future__ import print_function
from __future__ import division
from . import C_

from scipy.optimize import fmin
from scipy.optimize import curve_fit
from .synth.synthetic_SNe_fun import SNE_fun_numpy, inverse_SNE_fun_numpy
import numpy as np
from flamingchoripan.myUtils.progress_bars import ProgressBar

###################################################################################################################################################

def diff_vector(x:np.ndarray):
	if len(x)==0:
		return x
	x = x[...,None]
	to_append = np.expand_dims(x[0,...], axis=1)
	dx = np.diff(x, axis=0, prepend=to_append.T)
	return dx[:,0]

def calculate_parametric_model(lcdataset, set_name):
	lcset = lcdataset.get(set_name)
	keys = lcset.data_keys()
	bar = ProgressBar(len(keys))
	for key in keys:
		lcobj = lcset.data[key]
		for b in lcobj.bands:
			### fit parametric model of original object
			lcobjb = lcobj.get_b(b)
			success, pm_guess, pm_args, pm_features, times_dict = light_curve_fit(lcobjb, uses_random_guess=False)
			setattr(lcobjb, 'pm_guess', pm_guess)
			setattr(lcobjb, 'pm_args', pm_args)
			setattr(lcobjb, 'pm_times', times_dict)
			setattr(lcset, 'pm_features', pm_features)
			
		bar(f'set_name: {set_name} - key: {key} - pm_args: {pm_args}')
	bar.done()

def get_synth_dataset(lcdataset, set_name, desired_class_samples:int, obse_sampler, len_sampler,
	hours_noise_amp:float=5,
	cpds_p:float=0.015,
	std_scale:float=0.5,
	min_cadence_days:float=3.,
	min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
	get_from_synthetic:bool=False
	):
	assert cpds_p>=0 and cpds_p<=1
	desired_class_samples = int(desired_class_samples)
	original_lcset = lcdataset.get(set_name)
	max_obs = {b:original_lcset.get_lcset_max_value_b(b, 'obs') for b in original_lcset.band_names}
	print(f'generating synthetic samples - set_name: {set_name} - desired_class_samples: {desired_class_samples:,} - max_obs: {max_obs}')
	lcset = lcdataset.set_custom(f'synth_{set_name}', original_lcset.copy({}))
	keys = lcdataset.get(set_name).data_keys()
	bar = ProgressBar(desired_class_samples*len(lcset.class_names))
	class_counter = {c:0 for c in lcset.class_names}
	keys_counter = {key:0 for key in keys}
	used_keys = []
	index = 0
	while any([class_counter[c]<desired_class_samples for c in lcset.class_names]):
		if index>=len(keys):
			index = 0
			get_from_synthetic = True
		
		key = keys[index]
		original_lcobj = original_lcset.data[key].copy() # important to copy!!!
		
		### check counter
		if class_counter[lcset.class_names[original_lcobj.y]]>=desired_class_samples:
			index += 1
			continue

		### generate new object
		to_fit_lcobj = original_lcobj.copy() # important to copy!!!
		if get_from_synthetic:
			success_list = []
			for b in to_fit_lcobj.bands:
				to_fit_lcobjb = to_fit_lcobj.get_b(b)
				to_fit_lcobjb.add_day_noise_uniform(hours_noise_amp) # add day noise
				to_fit_lcobjb.add_obs_noise_gaussian(std_scale) # add obs noise
				to_fit_lcobjb.apply_downsampling(cpds_p) # curve points downsampling
				
				### fit parametric model per band using random variations
				success, pm_guess, pm_args, pm_features, times_dict = light_curve_fit(to_fit_lcobjb, uses_random_guess=False)
				setattr(to_fit_lcobjb, 'pm_guess', pm_guess)
				setattr(to_fit_lcobjb, 'pm_args', pm_args)
				setattr(to_fit_lcobjb, 'pm_times', times_dict)
				setattr(lcset, 'pm_features', pm_features)
				success_list.append(success)

				### generate new synth curve!
				if not success:
					continue

				### generate curve length
				curve_len = len_sampler.sample(1)[0]
				new_days = np.random.uniform(times_dict['ti'], times_dict['tf'], size=curve_len)
				new_days = np.sort(new_days)
				valid_new_days = diff_vector(new_days)>min_cadence_days
				new_days = new_days[valid_new_days]
				new_len_b = len(new_days)
				if new_len_b<=min_synthetic_len_b: # need to be long enough
					success = False
					continue

				### generate new observations
				pm_obs = SNE_fun_numpy(new_days, **pm_args)
				if pm_obs.min()<0: # can't have negative observations
					success = False
					continue

				new_obse = obse_sampler.sample(new_len_b, b)
				new_obs = np.clip(np.random.normal(pm_obs, new_obse*std_scale), 0, None)
				if new_obs.max()>=max_obs[b]: # can't be higher than max in original set
					success = False
					continue

				to_fit_lcobjb.set_values(new_days, new_obs, new_obse)

			sucesss_condition = any(success_list)

		else:
			for b in to_fit_lcobj.bands:
				to_fit_lcobjb = to_fit_lcobj.get_b(b)
				
				### fit parametric model per band using random variations
				success, pm_guess, pm_args, pm_features, times_dict = light_curve_fit(to_fit_lcobjb, uses_random_guess=False)
				setattr(to_fit_lcobjb, 'pm_guess', pm_guess)
				setattr(to_fit_lcobjb, 'pm_args', pm_args)
				setattr(to_fit_lcobjb, 'pm_times', times_dict)
				setattr(lcset, 'pm_features', pm_features)
				sucesss_condition = True

		if sucesss_condition:
			class_counter[lcset.class_names[to_fit_lcobj.y]] += 1
			new_key = f'{key}.{keys_counter[key]}'
			lcset.data[new_key] = to_fit_lcobj
			keys_counter[key] += 1
			if not key in used_keys:
				used_keys.append(key)
			bar(f'get_from_synthetic: {get_from_synthetic} - set_name: {set_name} - class_counter: {class_counter} - key: {key} - new_key: {new_key} - pm_args: {pm_args}')

		index += 1

	bar.done()
	setattr(lcset, 'used_keys', used_keys)
	return

def light_curve_fit(lcobjb,
	replace_nan_inf:bool=True,
	max_obs_error:float=1e10,
	p0:list=None,
	uses_random_guess:bool=False,
	pow_obs_error:bool=True,
	):
	success, pm_args, popt, pcov, pm_features, pm_guess = light_curve_fit_(lcobjb.days, lcobjb.obs, lcobjb.obse,
		replace_nan_inf,
		max_obs_error,
		p0,
		uses_random_guess,
		pow_obs_error,
		)
	times_dict = None
	if success:
		t0 = pm_args['t0']
		fmin_args = tuple([pm_args[key] for key in pm_args.keys()])
		tmax = fmin(inverse_SNE_fun_numpy, t0, fmin_args, disp=False)[0]
		ti = np.clip(tmax - (pm_args['trise']*5+pm_args['gamma']/10.), None, lcobjb.days[0])
		tf = np.clip(tmax + (pm_args['tfall']*5+pm_args['gamma']/2.), lcobjb.days[-1], None)
		assert tmax>=ti
		assert tf>=tmax
		times_dict = {
			'ti':ti,
			'tmax':tmax,
			'tf':tf,
		}
		
	return success, pm_guess, pm_args, pm_features, times_dict

def light_curve_fit_(days:np.ndarray, obs:np.ndarray, obs_error:np.ndarray,
	replace_nan_inf:bool=True,
	max_obs_error:float=1e10,
	p0:list=None,
	uses_random_guess:bool=False,
	pow_obs_error:bool=True,
	):
	'''
	-> pm_args:dict, popt:np.ndarray, pcov:np.ndarray
	'''
	### prepare
	pm_features = ['A', 't0', 'gamma', 'f', 'trise', 'tfall']
	obs_error = obs_error**2 if pow_obs_error else obs_error

	### checks
	if len(days)<C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT: # min points to even try a curve fit
		return False, None, None, None, pm_features, None
	assert np.all(obs_error>=0)
	assert len(days)==len(obs)
	assert len(days)==len(obs_error)

	### solve nans
	if replace_nan_inf:
		invalid_indexs = (obs == np.infty) | (obs == -np.infty) | np.isnan(obs)
		obs[invalid_indexs] = 0 # as a patch, use 0
		obs_error[invalid_indexs] = max_obs_error # as a patch, use a big obs error to null obs

	### bounds
	pm_bounds = get_pm_bounds(days, obs, obs_error)
	p0 = get_p0(days, obs, obs_error, pm_bounds, uses_random_guess) if p0 is None else p0
	
	fit_kwargs = {
		#'method':'lm',
		#'method':'trf',
		#'method':'dogbox',
		#'absolute_sigma':True,
		#'maxfev':1e6,
		'check_finite':True,
		'bounds':([pm_bounds[bkey][0] for bkey in pm_bounds.keys()], [pm_bounds[bkey][1] for bkey in pm_bounds.keys()]),
		'ftol':p0[0]/20., # A_guess
		'sigma':(obs_error+1e-20),
	}
	try:
		### A, t0, gamma, f, trise, tfall
		popt, pcov = curve_fit(SNE_fun_numpy, days, obs, p0=p0, **fit_kwargs)
	
	except ValueError:
		print('\n>>> wrong curve fitting')
		print('days', days)
		print('obs', obs)
		print('obs_error', obs_error)
		return False, None, None, None, pm_features, None
		#raise ValueError()

	except RuntimeError:
		print('\n>>> wrong curve fitting')
		print('days', days)
		print('obs', obs)
		print('obs_error', obs_error)
		return False, None, None, None, pm_features, None
		#raise RuntimeError()

	pm_args = {bound:popt[kbound] for kbound,bound in enumerate(pm_bounds.keys())}
	pm_guess = {bound:p0[kbound] for kbound,bound in enumerate(pm_bounds.keys())}
	return True, pm_args, popt, pcov, pm_features, pm_guess

def get_pm_bounds(days:np.ndarray, obs:np.ndarray, obs_error:np.ndarray):
	if len(days)<C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT: # min points to even try a curve fit
		return None

	### utils
	min_flux = np.min(obs)
	max_flux = np.max(obs)
	mean_flux = np.mean(obs)
	first_flux = obs[0]
	day_max_flux = days[np.argmax(obs)]
	first_day = days.min()
	last_day = days.max()

	t0_bound_value = 50
	pm_bounds = {
		#'A':(min_flux, max_flux),
		'A':(max_flux / 3., max_flux * 3.),
		#'A':(max_flux / 2., max_flux * 2.),
		#'A':(mean_flux / 3., max_flux * 3.),
		#'A':(mean_flux, max_flux*1.5),
		't0':(-t0_bound_value, +t0_bound_value),
		'gamma':(1., 100.),
		'f':(0., 1.),
		'trise':(1., 100.),
		'tfall':(1., 100.),
	}
	return pm_bounds

def get_random_mean(a, b, r):
	assert a<=b
	assert r>=0 and r<=1
	mid = a+(b-a)/2
	return np.random.uniform(mid*(1-r), mid*(1+r))

def get_p0(days:np.ndarray, obs:np.ndarray, obs_error:np.ndarray, pm_bounds,
	uses_random_guess:bool=False,
	):
	if len(days)<C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT: # min points to even try a curve fit
		return None

	### utils
	new_days = days-days[0]
	min_flux = np.min(obs)
	max_flux = np.max(obs)
	mean_flux = np.mean(obs)
	first_flux = obs[0]
	max_flux_day = new_days[np.argmax(obs)]
	first_day = new_days.min()
	last_day = new_days.max()
	frac_r = 0.2

	### A
	#A_guess = 1.2*max_flux if not uses_random_guess else get_random_mean(pm_bounds['A'][0], pm_bounds['A'][1], frac_r)
	A_guess = 1.2*max_flux if not uses_random_guess else get_random_mean(1.2*max_flux, 1.2*max_flux, frac_r)
	A_guess = np.clip(A_guess, pm_bounds['A'][0], pm_bounds['A'][1])

	### t0
	t0_guess = max_flux_day
	t0_guess = np.clip(t0_guess, pm_bounds['t0'][0], pm_bounds['t0'][1])
	
	### gamma
	mask = obs >= max_flux / 3. #np.percentile(obs, 33)
	gamma_guess = new_days[mask].max() - new_days[mask].min() if mask.sum() > 0 else 2.
	gamma_guess = np.clip(gamma_guess, pm_bounds['gamma'][0], pm_bounds['gamma'][1])

	### f
	f_guess = 0.5 if not uses_random_guess else get_random_mean(pm_bounds['f'][0], pm_bounds['f'][1], frac_r)
	
	### trise
	trise_guess = (max_flux_day - first_day) / 2.
	trise_guess = np.clip(trise_guess, pm_bounds['trise'][0], pm_bounds['trise'][1])
	
	### tfall
	tfall_guess = 40.
	tfall_guess = np.clip(tfall_guess, pm_bounds['tfall'][0], pm_bounds['tfall'][1])
	return [A_guess, t0_guess, gamma_guess, f_guess, trise_guess, tfall_guess]
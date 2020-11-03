from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from flamingchoripan.progress_bars import ProgressBar

###################################################################################################################################################

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
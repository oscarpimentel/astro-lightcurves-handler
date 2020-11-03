from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from scipy.optimize import fmin
from scipy.optimize import curve_fit
from . import exceptions as ex
from ..lc_classes import diff_vector
import pymc3 as pm
import logging

###################################################################################################################################################

def sgm(x:float):
	return 1/(1 + np.exp(-x))

def syn_sne_fnumpy(t, A, t0, gamma, f, trise, tfall,
	*args,
	**kwargs):
	assert np.all(~np.isnan(t))
	assert np.all(~np.isnan(A))
	nf = np.clip(f, 0, 1)
	early = 1.0*(A*(1 - (nf*(t-t0)/gamma))   /   (1 + np.exp(-(t-t0)/trise)))   *   (1 - sgm((t-(gamma+t0))/3))
	late = 1.0*(A*(1-nf)*np.exp(-(t-(gamma+t0))/tfall)   /   (1 + np.exp(-(t-t0)/trise)))   *   sgm((t-(gamma+t0))/3)
	flux = early + late
	return flux

def syn_sne_fpymc3(t, A, t0, gamma, f, trise, tfall,
	*args,
	**kwargs):
	early = 1.0*(A*(1 - (f*(t-t0)/gamma))   /   (1 + np.exp(-(t-t0)/trise)))   *   (1 - sgm((t-(gamma+t0))/3))
	late = 1.0*(A*(1-f)*np.exp(-(t-(gamma+t0))/tfall)   /   (1 + np.exp(-(t-t0)/trise)))   *   sgm((t-(gamma+t0))/3)
	flux = early + late
	return flux

def inverse_syn_sne_fnumpy(t, A, t0, gamma, f, trise, tfall):
	return -syn_sne_fnumpy(t, A, t0, gamma, f, trise, tfall)

def get_random_mean(a, b, r):
	assert a<=b
	assert r>=0 and r<=1
	mid = a+(b-a)/2
	return np.random.uniform(mid*(1-r), mid*(1+r))

def get_tmax(pm_args, pm_features, lcobjb):
		t0 = pm_args['t0']
		fmin_args = tuple([pm_args[pmf] for pmf in pm_features])
		tmax = fmin(inverse_syn_sne_fnumpy, t0, fmin_args, disp=False)[0]
		ti = np.clip(tmax - (pm_args['trise']*5+pm_args['gamma']/10.), None, lcobjb.days[0])
		tf = np.clip(tmax + (pm_args['tfall']*5+pm_args['gamma']/2.), lcobjb.days[-1], None)
		assert tmax>=ti
		assert tf>=tmax
		pm_times = {
			'ti':ti,
			'tmax':tmax,
			'tf':tf,
		}
		return pm_times

def extract_arrays(lcobjb):
	return lcobjb.days, lcobjb.obs, lcobjb.obse

###################################################################################################################################################

class SynSNeGeneratorCF():
	def __init__(self, lcobj, band_names, obse_sampler_bdict, length_sampler_bdict,
		pow_obs_error:bool=True,
		replace_nan_inf:bool=True,
		max_obs_error:float=1e10,
		uses_random_guess:bool=False,

		hours_noise_amp:float=5,
		cpds_p:float=0.015,
		std_scale:float=0.5,
		min_cadence_days:float=3.,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		):
		self.pm_features = ['A', 't0', 'gamma', 'f', 'trise', 'tfall']
		self.lcobj = lcobj.copy()
		self.band_names = band_names.copy()
		self.obse_sampler_bdict = obse_sampler_bdict
		self.length_sampler_bdict = length_sampler_bdict
		
		self.pow_obs_error = pow_obs_error
		self.replace_nan_inf = replace_nan_inf,
		self.max_obs_error = max_obs_error,
		self.uses_random_guess = uses_random_guess

		self.hours_noise_amp = hours_noise_amp
		self.cpds_p = cpds_p
		self.std_scale = std_scale
		self.min_cadence_days = min_cadence_days
		self.min_synthetic_len_b = min_synthetic_len_b

		self.reset()

	def reset(self):
		pass

	def get_pm_bounds(self, lcobjb):
		days, obs, obs_error = extract_arrays(lcobjb)

		### checks
		if len(days)<C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT: # min points to even try a curve fit
			raise ex.TooShortCurveError()

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
			#'gamma':(1., 100.),
			'gamma':(1., 25.),
			'f':(0., 1.),
			#'trise':(1., 100.),
			'trise':(1., 30.),
			#'tfall':(1., 100.),
			'tfall':(1., 70.),
		}
		return pm_bounds

	def get_p0(self, lcobjb, pm_bounds):
		days, obs, obs_error = extract_arrays(lcobjb)

		### checks
		if len(days)<C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT: # min points to even try a curve fit
			raise ex.TooShortCurveError()

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
		#A_guess = 1.2*max_flux if not self.uses_random_guess else get_random_mean(pm_bounds['A'][0], pm_bounds['A'][1], frac_r)
		A_guess = 1.2*max_flux if not self.uses_random_guess else get_random_mean(1.2*max_flux, 1.2*max_flux, frac_r)

		### t0
		t0_guess = max_flux_day
		
		### gamma
		mask = obs >= max_flux / 3. #np.percentile(obs, 33)
		gamma_guess = new_days[mask].max() - new_days[mask].min() if mask.sum() > 0 else 2.

		### f
		f_guess = 0.5 if not self.uses_random_guess else get_random_mean(pm_bounds['f'][0], pm_bounds['f'][-1], frac_r)
		
		### trise
		trise_guess = (max_flux_day - first_day) / 2.
		
		### tfall
		tfall_guess = 40.

		### set
		p0 = {
			'A':np.clip(A_guess, pm_bounds['A'][0], pm_bounds['A'][-1]),
			't0':np.clip(t0_guess, pm_bounds['t0'][0], pm_bounds['t0'][-1]),
			'gamma':np.clip(gamma_guess, pm_bounds['gamma'][0], pm_bounds['gamma'][-1]),
			'f':np.clip(gamma_guess, pm_bounds['f'][0], pm_bounds['f'][-1]),
			'trise':np.clip(trise_guess, pm_bounds['trise'][0], pm_bounds['trise'][-1]),
			'tfall':np.clip(tfall_guess, pm_bounds['tfall'][0], pm_bounds['tfall'][-1])
		}
		return p0

	def get_fitting_data_b(self, b):
		lcobjb = self.lcobj.get_b(b).copy() # copy
		lcobjb.add_day_noise_uniform(self.hours_noise_amp) # add day noise
		lcobjb.add_obs_noise_gaussian(self.std_scale) # add obs noise
		lcobjb.apply_downsampling(self.cpds_p) # curve points downsampling

		days, obs, obs_error = extract_arrays(lcobjb)
		pm_bounds = self.get_pm_bounds(lcobjb)
		p0 = self.get_p0(lcobjb, pm_bounds)
		obs_error = obs_error**2 if self.pow_obs_error else obs_error

		### checks
		if len(days)<C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT: # min points to even try a curve fit
			raise ex.TooShortCurveError()

		assert np.all(obs_error>=0)
		assert len(days)==len(obs)
		assert len(days)==len(obs_error)

		### solve nans
		if self.replace_nan_inf:
			invalid_indexs = (obs == np.infty) | (obs == -np.infty) | np.isnan(obs)
			obs[invalid_indexs] = 0 # as a patch, use 0
			obs_error[invalid_indexs] = self.max_obs_error # as a patch, use a big obs error to null obs

		### bounds
		fit_kwargs = {
			#'method':'lm',
			#'method':'trf',
			#'method':'dogbox',
			#'absolute_sigma':True,
			#'maxfev':1e6,
			'check_finite':True,
			'bounds':([pm_bounds[pmf][0] for pmf in self.pm_features], [pm_bounds[pmf][-1] for pmf in self.pm_features]),
			'ftol':p0['A']/20., # A_guess
			'sigma':(obs_error+1e-20),
		}

		### fitting
		try:
			popt, pcov = curve_fit(SNE_fun_numpy, days, obs, p0=[p0[pmf] for pmf in self.pm_features], **fit_kwargs)
		
		except ValueError:
			raise ex.CurveFitError()

		except RuntimeError:
			raise ex.CurveFitError()

		pm_args = {pmf:popt[kpmf] for kpmf,pmf in enumerate(self.pm_features)}
		pm_guess = {pmf:p0[pmf] for kpmf,pmf in enumerate(self.pm_features)}
		return pm_args, pm_guess, pcov, lcobjb

	def sample_curves(self, n,
		uses_pm_obs:bool=False,
		):
		new_lcobjs = [self.lcobj.copy() for _ in range(n)]
		for b in self.band_names:
			new_lcobjbs = self.sample_curve_b(b, n, uses_pm_obs)
			for new_lcobj,new_lcobjb in zip(new_lcobjs, new_lcobjbs):
				new_lcobj.add_sublcobj_b(b, new_lcobjb)

		return new_lcobjs

	def sample_curve_b(self, b, n,
		uses_pm_obs:bool=False,
		):
		curve_lengths = self.length_sampler_bdict[b].sample(n)
		new_lcobjbs = []
		for k in range(n):
			try:
				pm_args, pm_guess, pcov, lcobjb = self.get_fitting_data_b(b)
				pm_times = get_tmax(pm_args, self.pm_features, lcobjb)
				new_lcobjb = self.__sample_curve__(pm_times, pm_args, curve_lengths[k], lcobjb, b, uses_pm_obs)
			except ex.TooShortCurveError:
				new_lcobjb = self.lcobj.get_b(b).copy()
			except ex.SyntheticCurveTimeoutError:
				new_lcobjb = self.lcobj.get_b(b).copy()

			new_lcobjbs.append(new_lcobjb)

		return new_lcobjbs

	def __sample_curve__(self, pm_times, pm_args, size, lcobjb, b,
		uses_pm_obs:bool=False,
		):
		timeout_counter = 1000
		max_obs_threshold_scale = 2
		new_lcobjb = lcobjb.copy()
		i = 0
		while True:
			i += 1
			if i>=timeout_counter:
				raise ex.SyntheticCurveTimeoutError()

			### generate times to evaluate
			if uses_pm_obs:
				new_days = np.linspace(pm_times['ti'], pm_times['tf'], 100)
			else:
				new_days = np.random.uniform(pm_times['ti'], pm_times['tf'], size=size)
				new_days = np.sort(new_days) # sort
				valid_new_days = diff_vector(new_days)>=self.min_cadence_days
				new_days = new_days[valid_new_days]
				new_len_b = len(new_days)
				if new_len_b<=self.min_synthetic_len_b: # need to be long enough
					continue

			### generate parametric observations
			pm_obs = syn_sne_fnumpy(new_days, **pm_args)
			if pm_obs.min()<0: # can't have negative observations
				continue

			### resampling obs using obs error
			if uses_pm_obs:
				new_obse = pm_obs*0
				new_obs = pm_obs
			else:
				new_obse = self.obse_sampler_bdict[b].conditional_sample(pm_obs)
				new_obs = np.clip(np.random.normal(pm_obs, new_obse*self.std_scale), 0, None)
			
			if new_obs.max()>=lcobjb.obs.max()*max_obs_threshold_scale: # can't be too high
				continue

			new_lcobjb.set_values(new_days, new_obs, new_obse)
			return new_lcobjb

###################################################################################################################################################

class SynSNeGeneratorMCMC(SynSNeGeneratorCF):
	def __init__(self, lcobj, band_names, obse_sampler_bdict, length_sampler_bdict):
		super().__init__(lcobj, band_names, obse_sampler_bdict, length_sampler_bdict)

	def reset(self):
		return

	def get_mcmc_traces(self, b, n):
		lcobjb = self.lcobj.get_b(b).copy() # copy
		days, obs, obs_error = extract_arrays(lcobjb)

		### checks
		if len(days)<C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT: # min points to even try a curve fit
			raise ex.TooShortCurveError()

		### utils
		min_flux = np.min(obs)
		max_flux = np.max(obs)
		mean_flux = np.mean(obs)
		first_flux = obs[0]
		day_max_flux = days[np.argmax(obs)]
		first_day = days.min()
		last_day = days.max()
		t0_bound_value = 50
		
		basic_model = pm.Model()
		with basic_model:
			A = pm.Uniform('A', 0, 5*max_flux)#,testval = A_est)
			t0 = pm.Uniform('t0', day_max_flux - 40, day_max_flux+60)# ,testval = t0_est)
			gamma = pm.Normal('gamma', 35, 10)#, testval = gamma_est)
			f = pm.Uniform('f', 0, 1)
			trise = pm.Uniform('trise', 1, 30)#, testval = taur_est)
			tfall = pm.Uniform('tfall',1, 70)#, testval = tauf_est)

			pm_obs = syn_sne_fpymc3(days, A, t0, gamma, f, trise, tfall)
			pm_obs = pm.Normal('pm_obs', mu=pm_obs, sigma=obs_error, observed=obs)
			#prior_checks = pm.sample_prior_predictive(samples=50, random_seed=0)

		trace_kwargs = {
			'tune':1000, # burn-in steps
			'discard_tuned_samples':True,
			'draws':n, # posterior draws per chain
			'cores':2,
			'progressbar':0,
		}
		#logger = logging.getLogger('pymc3'); logger.setLevel(logging.ERROR)
		with basic_model:
			mcmc_trace = pm.sample(step=pm.Metropolis(), **trace_kwargs)
			#ppc = pm.sample_posterior_predictive(trace, var_names=trace.varnames)

		return mcmc_trace, lcobjb

	def sample_curve_b(self, b, n,
		uses_pm_obs:bool=False,
		):
		mcmc_trace, lcobjb = self.get_mcmc_traces(b, n)
		self.mcmc_trace = mcmc_trace # debug
		curve_lengths = self.length_sampler_bdict[b].sample(n)
		new_lcobjbs = []
		for k in range(n):
			try:
				pm_args = {pmf:mcmc_trace[pmf][-k] for pmf in self.pm_features}
				pm_times = get_tmax(pm_args, self.pm_features, lcobjb)
				new_lcobjb = self.__sample_curve__(pm_times, pm_args, curve_lengths[k], lcobjb, b, uses_pm_obs)
			except ex.TooShortCurveError:
				new_lcobjb = self.lcobj.get_b(b).copy()
			except ex.SyntheticCurveTimeoutError:
				new_lcobjb = self.lcobj.get_b(b).copy()

			new_lcobjbs.append(new_lcobjb)

		return new_lcobjbs
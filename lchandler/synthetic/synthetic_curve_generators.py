from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from scipy.optimize import fmin
from scipy.optimize import curve_fit
from . import exceptions as ex
from ..lc_classes import diff_vector
import pymc3 as pm

###################################################################################################################################################

def override(func): return func

def sgm(x:float):
	return 1/(1 + np.exp(-x))

def syn_sne_fnumpy(t, A, t0, gamma, f, trise, tfall):
	assert np.all(~np.isnan(t))
	assert np.all(~np.isnan(A))
	nf = np.clip(f, 0, 1)

	#s = sgm((t-(gamma+t0))*0.33333333)
	s = sgm((t-(gamma+t0))*1e0)
	#s = sgm((t-t0)*1e2)

	early = 1.0*(A*(1 - (nf*(t-t0)/gamma))   /   (1 + np.exp(-(t-t0)/trise)))   *   (1 - s)
	late = 1.0*(A*(1-nf)*np.exp(-(t-(gamma+t0))/tfall)   /   (1 + np.exp(-(t-t0)/trise)))   *   s
	flux = early+late
	return flux

def syn_sne_fpymc3(t, A, t0, gamma, f, trise, tfall):
	#s = sgm((t-(gamma+t0))*0.33333333)
	s = sgm((t-(gamma+t0))*1e0)
	#s = sgm((t-t0)*1e2)

	early = 1.0*(A*(1 - (f*(t-t0)/gamma))   /   (1 + np.exp(-(t-t0)/trise)))   *   (1 - s)
	late = 1.0*(A*(1-f)*np.exp(-(t-(gamma+t0))/tfall)   /   (1 + np.exp(-(t-t0)/trise)))   *   s
	flux = early+late
	return flux

def inverse_syn_sne_fnumpy(t, A, t0, gamma, f, trise, tfall):
	return -syn_sne_fnumpy(t, A, t0, gamma, f, trise, tfall)

def get_random_mean(a, b, r):
	assert a<=b
	assert r>=0 and r<=1
	mid = a+(b-a)/2
	return np.random.uniform(mid*(1-r), mid*(1+r))

def get_tmax(pm_args, pm_features, lcobjb, threshold):
		t0 = pm_args['t0']
		fmin_args = tuple([pm_args[pmf] for pmf in pm_features])
		tmax = fmin(inverse_syn_sne_fnumpy, t0, fmin_args, disp=False)[0]
		first_day = lcobjb.days[0]
		last_day = lcobjb.days[-1]

		### ti
		dti = 25
		lin_times = np.linspace(tmax-dti, tmax, int(dti)*24)
		indexs = np.where(syn_sne_fnumpy(lin_times, *fmin_args)>threshold)[0]
		ti = np.clip(lin_times[indexs[0]], tmax-dti, tmax)
		
		### tf
		dtf = 200
		lin_times = np.linspace(tmax, tmax+dtf, int(dtf)*24)
		indexs = np.where(syn_sne_fnumpy(lin_times, *fmin_args)>threshold)[0]
		tf = np.clip(lin_times[indexs[-1]], last_day, tmax+dtf)
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
		new_bounds=True,
		pow_obs_error:bool=False,
		replace_nan_inf:bool=True,
		max_obs_error:float=1e10,
		uses_random_guess:bool=False,

		hours_noise_amp:float=5,
		cpds_p:float=0.015,
		std_scale:float=1,
		min_cadence_days:float=3.,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		):
		self.pm_features = ['A', 't0', 'gamma', 'f', 'trise', 'tfall']
		self.lcobj = lcobj.copy()
		self.band_names = band_names.copy()
		self.obse_sampler_bdict = obse_sampler_bdict
		self.length_sampler_bdict = length_sampler_bdict
		
		self.new_bounds = new_bounds
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
		self.min_obs_bdict = {b:self.obse_sampler_bdict[b].min_obs for b in self.band_names}

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

		if not self.new_bounds:
			pm_bounds = {
				'A':(max_flux / 3., max_flux * 3.),
				't0':(-80, +80),
				'gamma':(1., 100.),
				'f':(0., 1.),
				'trise':(1., 100.),
				'tfall':(1., 100.),
			}
		else:
			pm_bounds = {
				'A':(max_flux / 8., max_flux * 8.),
				't0':(day_max_flux-100, day_max_flux+100),
				'gamma':(1., 70.),
				'f':(0., 1.),
				'trise':(1., 20.),
				'tfall':(1., 100.),
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
		lcobjb.add_obs_noise_gaussian(self.std_scale, self.min_obs_bdict[b]) # add obs noise
		lcobjb.apply_downsampling(self.cpds_p) # curve points downsampling

		days, obs, obs_error = extract_arrays(lcobjb)
		obs_error = (obs_error*self.std_scale)**2 if self.pow_obs_error else obs_error*self.std_scale
		pm_bounds = self.get_pm_bounds(lcobjb)
		p0 = self.get_p0(lcobjb, pm_bounds)

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
			p0_ = [p0[pmf] for pmf in self.pm_features]
			popt, pcov = curve_fit(syn_sne_fnumpy, days, obs, p0=p0_, **fit_kwargs)
			#popt, pcov = curve_fit(lambda t, A, t0, gamma, f, trise, tfall:syn_sne_fnumpy(t, A, t0, gamma, f, trise, tfall, self.min_obs_bdict[b]), days, obs, p0=p0_, **fit_kwargs)
		
		except ValueError:
			raise ex.CurveFitError()

		except RuntimeError:
			raise ex.CurveFitError()

		pm_args = {pmf:popt[kpmf] for kpmf,pmf in enumerate(self.pm_features)}
		pm_guess = {pmf:p0[pmf] for kpmf,pmf in enumerate(self.pm_features)}
		return pm_args, pm_guess, pcov, lcobjb

	def sample_curves(self, n):
		new_lcobjs = [self.lcobj.copy() for _ in range(n)]
		new_lcobjs_pm = [self.lcobj.copy() for _ in range(n)]
		for b in self.band_names:
			new_lcobjbs, new_lcobjbs_pm = self.sample_curve_b(b, n)

			for new_lcobj,new_lcobjb in zip(new_lcobjs, new_lcobjbs):
				new_lcobj.add_sublcobj_b(b, new_lcobjb)

			for new_lcobj_pm,new_lcobjb_pm in zip(new_lcobjs_pm, new_lcobjbs_pm):
				new_lcobj_pm.add_sublcobj_b(b, new_lcobjb_pm)

		return new_lcobjs, new_lcobjs_pm

	def sample_curve_b(self, b, n):
		curve_lengths = self.length_sampler_bdict[b].sample(n)
		new_lcobjbs = []
		new_lcobjbs_pm = []
		for kn in range(n):
			try:
				pm_args, pm_guess, pcov, lcobjb = self.get_fitting_data_b(b)
				pm_times = get_tmax(pm_args, self.pm_features, lcobjb, self.min_obs_bdict[b])
				new_lcobjb = self.__sample_curve__(pm_times, pm_args, curve_lengths[kn], lcobjb, b)
				new_lcobjb_pm = self.__sample_curve__(pm_times, pm_args, curve_lengths[kn], lcobjb, b, True)
			except ex.TooShortCurveError:
				new_lcobjb = self.lcobj.get_b(b).copy() # just use the original
				new_lcobjb_pm = self.lcobj.get_b(b).copy() # just use the original
			except ex.SyntheticCurveTimeoutError:
				new_lcobjb = self.lcobj.get_b(b).copy() # just use the original
				new_lcobjb_pm = self.lcobj.get_b(b).copy() # just use the original

			new_lcobjbs.append(new_lcobjb)
			new_lcobjbs_pm.append(new_lcobjb_pm)

		return new_lcobjbs, new_lcobjbs_pm

	def __sample_curve__(self, pm_times, pm_args, size, lcobjb, b,
		uses_pm_obs:bool=False,
		timeout_counter=10000,
		max_obs_threshold_scale=10.,
		pm_obs_n=100,
		):
		new_lcobjb = lcobjb.copy()
		i = 0
		while True:
			i += 1
			if i>=timeout_counter:
				raise ex.SyntheticCurveTimeoutError()

			### generate times to evaluate
			if uses_pm_obs:
				new_days = np.linspace(pm_times['ti'], pm_times['tf'], pm_obs_n)
			else:
				### generate days grid according to cadence
				new_day = pm_times['ti']
				new_days = []
				while new_day<pm_times['tf']:
					new_days.append(new_day)
					new_day += self.min_cadence_days
				new_days = np.array(new_days)

				### generate actual observation times
				idxs = np.random.permutation(np.arange(0, len(new_days)))
				new_days = new_days[idxs][:size] # random select
				new_days = np.sort(new_days) # sort

				if len(new_days)<=self.min_synthetic_len_b: # need to be long enough
					continue

			### generate parametric observations
			pm_obs = syn_sne_fnumpy(new_days, *[pm_args[pmf] for pmf in self.pm_features])
			if pm_obs.min()<self.min_obs_bdict[b]: # can't have observation above the threshold
				continue

			### resampling obs using obs error
			if uses_pm_obs:
				new_obse = pm_obs*0
				new_obs = pm_obs
			else:
				new_obse, new_obs = self.obse_sampler_bdict[b].conditional_sample(pm_obs)
				new_obs = np.clip(np.random.normal(pm_obs, new_obse*self.std_scale), self.min_obs_bdict[b], None)
			
			if new_obs.max()>lcobjb.obs.max()*max_obs_threshold_scale: # flux can't be too high
				continue

			new_lcobjb.set_values(new_days, new_obs, new_obse)
			return new_lcobjb

###################################################################################################################################################

class SynSNeGeneratorMCMC(SynSNeGeneratorCF):
	def __init__(self, lcobj, band_names, obse_sampler_bdict, length_sampler_bdict,
		new_bounds=True,
		pow_obs_error:bool=False,
		replace_nan_inf:bool=True,
		max_obs_error:float=1e10,
		uses_random_guess:bool=False,

		hours_noise_amp:float=5,
		cpds_p:float=0.015, # used only in curve_fit mode
		std_scale:float=0.5,
		min_cadence_days:float=3.,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		):
		super().__init__(lcobj, band_names, obse_sampler_bdict, length_sampler_bdict,
			new_bounds,
			pow_obs_error,
			replace_nan_inf,
			max_obs_error,
			uses_random_guess,

			hours_noise_amp,
			cpds_p,
			std_scale,
			min_cadence_days,
			min_synthetic_len_b,
			)
		self.mcmc_trace_bdict = {}

	def get_mcmc_traces(self, b, n,
		cores=2,
		n_tune=1000,
		n_samples=2000,
		):
		lcobjb = self.lcobj.get_b(b).copy() # copy
		days, obs, obs_error = extract_arrays(lcobjb)
		obs_error = (obs_error*self.std_scale)**2 if self.pow_obs_error else obs_error*self.std_scale

		### checks
		assert n%cores==0
		if len(days)<C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT: # min points to even try a curve fit
			raise ex.TooShortCurveError()
		
		### pymc3
		trace_kwargs = {
			'tune':n_tune, # burn-in steps
			'cores':cores,
			'progressbar':False,
			#'target_accept':.95,
			'target_accept':1.,
		}
		pm_bounds = self.get_pm_bounds(lcobjb)
		import logging; logger = logging.getLogger('pymc3'); logger.setLevel(logging.ERROR) # remove logger
		basic_model = pm.Model()
		with basic_model:
			try:
				A = pm.Uniform('A', *pm_bounds['A'])
				t0 = pm.Uniform('t0', *pm_bounds['t0'])
				#gamma = pm.Normal('gamma', 35, 10)
				gamma = pm.Uniform('gamma', *pm_bounds['gamma'])
				#gamma = pm.Gamma('gamma', alpha=pm_bounds['gamma'][0]+(pm_bounds['gamma'][-1]-pm_bounds['gamma'][0])/2., beta=1.)
				f = pm.Uniform('f', 0., 1.)
				#f = pm.Beta('f', alpha=2.5, beta=1)
				trise = pm.Uniform('trise', *pm_bounds['trise'])
				tfall = pm.Uniform('tfall', *pm_bounds['tfall'])

				pm_obs = syn_sne_fpymc3(days, A, t0, gamma, f, trise, tfall)
				pm_obs = pm.Normal('pm_obs', mu=pm_obs, sigma=obs_error, observed=obs)

				# trace
				#step = pm.Metropolis()
				#step = pm.NUTS()
				mcmc_trace = pm.sample(n_samples, **trace_kwargs)
			except ValueError:
				raise ex.MCMCError()

		return mcmc_trace, lcobjb, n_samples

	@override
	def sample_curve_b(self, b, n):
		try:
			mcmc_trace, lcobjb, n_samples = self.get_mcmc_traces(b, n)
			self.mcmc_trace_bdict[b] = mcmc_trace
		except ex.TooShortCurveError:
			return [self.lcobj.get_b(b).copy() for kn in range(n)], [self.lcobj.get_b(b).copy() for kn in range(n)]
		except ex.MCMCError:
			return [self.lcobj.get_b(b).copy() for kn in range(n)], [self.lcobj.get_b(b).copy() for kn in range(n)]

		curve_lengths = self.length_sampler_bdict[b].sample(n)
		new_lcobjbs = []
		new_lcobjbs_pm = []
		#rindexs = np.random.permutation(np.arange(0, n_samples))
		for kn in range(n):
			try:
				#pm_args = {pmf:self.mcmc_trace_bdict[b][pmf][rindexs[kn]] for pmf in self.pm_features}
				pm_args = {pmf:self.mcmc_trace_bdict[b][pmf][-kn] for pmf in self.pm_features}
				pm_times = get_tmax(pm_args, self.pm_features, lcobjb, self.min_obs_bdict[b])
				new_lcobjb = self.__sample_curve__(pm_times, pm_args, curve_lengths[kn], lcobjb, b)
				new_lcobjb_pm = self.__sample_curve__(pm_times, pm_args, curve_lengths[kn], lcobjb, b, True)
			except ex.TooShortCurveError:
				new_lcobjb = self.lcobj.get_b(b).copy() # just use the original
				new_lcobjb_pm = self.lcobj.get_b(b).copy() # just use the original
			except ex.SyntheticCurveTimeoutError:
				new_lcobjb = self.lcobj.get_b(b).copy() # just use the original
				new_lcobjb_pm = self.lcobj.get_b(b).copy() # just use the original

			new_lcobjbs.append(new_lcobjb)
			new_lcobjbs_pm.append(new_lcobjb_pm)

		return new_lcobjbs, new_lcobjbs_pm
import numpy as np
from .synthetic_utils import *
from .synthetic_SNe_fun import SNE_fun_numpy
import copy

def generate_FSNe(SNe_type:str, t:np.ndarray, dropout:float, sigma_error_arguments:dict,
	SNe_kwargs=None):
	t0 = -60
	tsigm = 0.66
	Asigm = 0.02
	tmax = 100
	SHORT_PARAMS = {
		'A':np.clip(np.random.normal(0.12, Asigm), 0, None),
		't0':np.random.normal(t0, tsigm),
		'gamma':np.clip(np.random.normal(40, 1), 0, None),
		'f':np.random.beta(100, 30),
		'trise':np.clip(np.random.normal(7, tsigm), 0, tmax),
		'tfall':np.clip(np.random.normal(10, tsigm), 0, tmax),
	}
	LONG_PARAMS = {
		'A':np.clip(np.random.normal(0.1, Asigm), 0, None),
		't0':np.random.normal(t0, tsigm),
		'gamma':np.clip(np.random.normal(40, 1), 0, None),
		'f':np.random.beta(200, 200),
		'trise':np.clip(np.random.normal(9, tsigm), 0, tmax),
		'tfall':np.clip(np.random.normal(50, tsigm), 0, tmax),
	}
	SMALL_PARAMS = {
		'A':np.clip(np.random.normal(0.045, Asigm), 0, None),
		't0':np.random.normal(t0, tsigm),
		'gamma':np.clip(np.random.normal(30, 1), 0, None),
		'f':np.random.beta(50, 1), # -> 1
		'trise':np.clip(np.random.normal(7, tsigm), 0, tmax),
		'tfall':np.clip(np.random.normal(50, tsigm), 0, tmax),
	}
	PARAMS_DIC = {
		'short':SHORT_PARAMS,
		'long':LONG_PARAMS,
		'small':SMALL_PARAMS,
	}

	assert SNe_type in PARAMS_DIC.keys()
	theorical_model = (PARAMS_DIC[SNe_type] if SNe_kwargs is None else copy.deepcopy(SNe_kwargs)) # generate new theorical model	
	theorical_model.update({'it':t[0], 'ft':t[-1]})
	flux = SNE_fun_numpy(t, theorical_model['A'], theorical_model['t0'], theorical_model['gamma'], theorical_model['f'], theorical_model['trise'], theorical_model['tfall'])

	t, flux = subsampling(t, flux, dropout)
	t, flux, error_amp = error_resampling(t, flux, sigma_error_arguments)
	lc = np.concatenate([t[:,None], flux[:,None], error_amp[:,None]], axis=-1).astype(np.float32)

	return lc, theorical_model


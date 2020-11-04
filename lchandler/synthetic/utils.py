from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from scipy.stats import bernoulli

###################################################################################################################################################

def subsampling(t:np.ndarray, x:np.ndarray, dropout:float):
	assert dropout>=0 and dropout<=1
	N = len(t)
	m = bernoulli.rvs((1-dropout), size=N).astype(bool)
	t = t[m]
	x = x[m]
	return t, x 
	
def error_resampling(t:np.ndarray, x:np.ndarray, sigma_error_arguments:dict):
	N = len(t)
	error_amp = np.random.gamma(sigma_error_arguments['shape'], sigma_error_arguments['scale'], size=N)
	error_amp += sigma_error_arguments['loc']
	x = np.clip(np.random.normal(x, error_amp, N), 0, None)
	return t, x, error_amp
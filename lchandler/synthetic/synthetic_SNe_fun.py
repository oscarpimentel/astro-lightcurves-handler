from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np

###################################################################################################################################################

def sgm(x:float):
	return 1/(1 + np.exp(-x))

def inverse_SNE_fun_numpy(t, A, t0, gamma, f, trise, tfall):
	return -SNE_fun_numpy(t, A, t0, gamma, f, trise, tfall)

def SNE_fun_numpy(t, A, t0, gamma, f, trise, tfall,
	*args,
	**kwargs):
	assert np.all(~np.isnan(t))
	assert np.all(~np.isnan(A))
	nf = np.clip(f, 0, 1)
	early = 1.0*(A*(1 - (nf*(t-t0)/gamma))   /   (1 + np.exp(-(t-t0)/trise)))   *   (1 - sgm((t-(gamma+t0))/3))
	late = 1.0*(A*(1-nf)*np.exp(-(t-(gamma+t0))/tfall)   /   (1 + np.exp(-(t-t0)/trise)))   *   sgm((t-(gamma+t0))/3)
	flux = early + late
	return flux

def parametricSNe(t,
	**SNe_kwargs):
	return SNE_fun_numpy(t, SNe_kwargs['A'], SNe_kwargs['t0'], SNe_kwargs['gamma'], SNe_kwargs['f'], SNe_kwargs['trise'], SNe_kwargs['tfall'])
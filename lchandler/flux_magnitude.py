from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np

DEFAULT_ZP = _C.DEFAULT_ZP
DEFAULT_FLUX_SCALE = _C.DEFAULT_FLUX_SCALE
CLIP_EPS = 1e-16

# http://slittlefair.staff.shef.ac.uk/teaching/phy217/lectures/stats/L18/index.html
# https://www.nde-ed.org/GeneralResources/ErrorAnalysis/UncertaintyTerms.htm
###################################################################################################################################################

def get_flux_from_magnitude(mag:np.ndarray,
	zero_point:float=DEFAULT_ZP,
	scale:float=DEFAULT_FLUX_SCALE,
	):
	assert np.all(mag>=0)
	flux = 10**(-(mag+zero_point)/2.5) # return 10 ** (-(mag + 48.6) / 2.5 + 26.0)
	flux = flux*scale
	assert np.all(flux>=0)
	return flux

def get_flux_error_from_magnitude(mag:np.ndarray, mag_error:np.ndarray,
	zero_point:float=DEFAULT_ZP,
	scale:float=DEFAULT_FLUX_SCALE,
	):
	assert np.all(mag>=0)
	assert np.all(mag_error>=0)
	flux1 = get_flux_from_magnitude(mag,
		zero_point,
		scale,
		)
	flux2 = get_flux_from_magnitude(mag+mag_error,
		zero_point,
		scale,
		)
	flux_error = np.abs(flux1-flux2)
	return flux_error

###################################################################################################################################################

def get_magnitude_from_flux(flux:np.ndarray,
	zero_point:float=DEFAULT_ZP,
	scale:float=DEFAULT_FLUX_SCALE,
	clip_eps=CLIP_EPS,
	):
	assert np.all(flux>=0)
	new_flux = flux if clip_eps is None else np.clip(flux, clip_eps, None) 
	mag = -2.5*np.log10(new_flux/scale)-zero_point # inverse
	assert np.all(mag>=0)
	return mag

def get_magnitude_error_from_flux(flux:np.ndarray, flux_error:np.ndarray,
	zero_point:float=DEFAULT_ZP,
	scale:float=DEFAULT_FLUX_SCALE,
	clip_eps=CLIP_EPS,
	):
	assert np.all(flux>=0)
	assert np.all(flux_error>=0)
	mag1 = get_magnitude_from_flux(flux,
		zero_point,
		scale,
		clip_eps,
		)
	mag2 = get_magnitude_from_flux(flux+flux_error,
		zero_point,
		scale,
		clip_eps,
		)
	mag_error = np.abs(mag1-mag2)
	return mag_error
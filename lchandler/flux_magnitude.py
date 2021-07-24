from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np

DEFAULT_ZP = _C.DEFAULT_ZP
DEFAULT_FLUX_SCALE = _C.DEFAULT_FLUX_SCALE
DEFAULT_MAG_SCALE = _C.DEFAULT_MAG_SCALE
EPS = 1e-10

# http://slittlefair.staff.shef.ac.uk/teaching/phy217/lectures/stats/L18/index.html
# https://www.nde-ed.org/GeneralResources/ErrorAnalysis/UncertaintyTerms.htm
###################################################################################################################################################

def get_flux_from_magnitude(mag:np.ndarray,
	zero_point:float=DEFAULT_ZP,
	scale:float=DEFAULT_FLUX_SCALE,
	):
	assert np.all(mag>=0)
	flux = 10**(-(mag+zero_point)/2.5)*scale
	return flux

def get_flux_error_from_magnitude(mag:np.ndarray, mag_error:np.ndarray,
	zero_point:float=DEFAULT_ZP,
	scale:float=DEFAULT_FLUX_SCALE,
	):
	assert np.all(mag>=0)
	assert np.all(mag_error>=0)
	flux1 = get_flux_from_magnitude(mag, zero_point, scale)
	flux2 = get_flux_from_magnitude(mag+mag_error, zero_point, scale)
	flux_error = np.abs(flux1-flux2)
	return flux_error

def get_magnitude_from_flux(flux:np.ndarray,
	zero_point:float=DEFAULT_ZP,
	scale:float=DEFAULT_MAG_SCALE,
	clip_flux:bool=False,
	eps=EPS,
	):
	assert np.all(flux>=0)
	new_flux = np.clip(flux, eps, None) if clip_flux else flux
	mag = (-2.5*np.log10(new_flux)+zero_point)*scale
	return mag

def get_magnitude_error_from_flux(flux:np.ndarray, flux_error:np.ndarray,
	zero_point:float=DEFAULT_ZP,
	scale:float=DEFAULT_MAG_SCALE,
	clip_flux:bool=False,
	):
	assert np.all(flux>=0)
	assert np.all(flux_error>=0)
	mag1 = get_magnitude_from_flux(flux, zero_point, scale, clip_flux)
	mag2 = get_magnitude_from_flux(flux+flux_error, zero_point, scale, clip_flux)
	mag_error = np.abs(mag1-mag2)
	return mag_error
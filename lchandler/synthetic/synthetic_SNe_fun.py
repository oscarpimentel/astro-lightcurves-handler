from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np

###################################################################################################################################################



def parametricSNe(t,
	**SNe_kwargs):
	return SNE_fun_numpy(t, SNe_kwargs['A'], SNe_kwargs['t0'], SNe_kwargs['gamma'], SNe_kwargs['f'], SNe_kwargs['trise'], SNe_kwargs['tfall'])
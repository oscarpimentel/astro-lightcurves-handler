import numpy as np
import torch

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

def SNE_fun_torch(t, A, t0, gamma, f, trise, tfall,
	*args,
	**kwargs):
	assert torch.all(~torch.isnan(t))
	assert torch.all(~torch.isnan(A))
	nf = torch.clamp(f, 0, 1)
	early = 1.0*(A*(1 - (nf*(t-t0)/gamma))   /   (1 + torch.exp(-(t-t0)/trise)))   *   (1 - torch.sigmoid((t-(gamma+t0))/3))
	late = 1.0*(A*(1-nf)*torch.exp(-(t-(gamma+t0))/tfall)   /   (1 + torch.exp(-(t-t0)/trise)))   *   torch.sigmoid((t-(gamma+t0))/3)
	flux = early + late
	return flux

def parametricSNe(t,
	**SNe_kwargs):
	return SNE_fun_numpy(t, SNe_kwargs['A'], SNe_kwargs['t0'], SNe_kwargs['gamma'], SNe_kwargs['f'], SNe_kwargs['trise'], SNe_kwargs['tfall'])
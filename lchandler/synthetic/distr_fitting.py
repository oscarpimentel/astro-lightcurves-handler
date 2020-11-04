from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import scipy.stats as stats
from sklearn import preprocessing as prep
from flamingchoripan.datascience.statistics import dropout_extreme_percentiles, get_linspace_ranks

###################################################################################################################################################

class ObsErrorConditionalSampler():
	def __init__(self, lcdataset:dict, set_name:str, b:str,
		samples_per_range:int=15,
		):
		self.lcdataset = lcdataset
		self.lcset = lcdataset[set_name]
		self.b = b
		self.samples_per_range = samples_per_range
		self.raw_obse = np.concatenate([lcobj.get_b(b).obse for lcobj in self.lcset.get_lcobjs()])
		self.raw_obs = np.concatenate([lcobj.get_b(b).obs for lcobj in self.lcset.get_lcobjs()])
		self.min_obs = self.raw_obs.min()
		assert self.min_obs>=0
		self.reset()
		
	def get_m_n(self):
		obse = self.raw_obse
		obs = self.raw_obs
		n = 6
		grid = np.linspace(0, 0.15, n+1)
		index_per_range = [np.where((obs>grid[k]) & (obs<=grid[k+1])) for k in range(len(grid)-1)]

		self.lr_x = []
		self.lr_y = []
		for k,indexs in enumerate(index_per_range):
			if len(indexs[0])<=1:
				continue
			sub_obse = obse[indexs]
			sub_obs = obs[indexs]
			sub_obse, ind = dropout_extreme_percentiles(sub_obse, p=np.exp(-k*2)*15, mode='upper')
			sub_obs = sub_obs[ind]
			i = sub_obse.argmax()
			self.lr_x.append(sub_obse[i])
			self.lr_y.append(sub_obs[i])

		#print(self.lr_x, self.lr_y)
		slope, intercept, r_value, p_value, std_err = stats.linregress(self.lr_x, self.lr_y)
		self.m = slope
		self.n = intercept

	def reset(self):
		### fit diagonal line
		self.get_m_n()
		valid_indexs = np.where(self.raw_obs>=self.raw_obse*self.m+self.n)
		self.obse = self.raw_obse[valid_indexs]
		self.obs = self.raw_obs[valid_indexs]

		### generate obs grid
		self.rank_ranges, self.obs_indexs_per_range, self.ranks = get_linspace_ranks(self.obs, self.samples_per_range)
		self.distrs = [self.get_fitted_distr(obs_indexs, k) for k,obs_indexs in enumerate(self.obs_indexs_per_range)]
		
	def get_fitted_distr(self, obs_indexs, k,
		eps=1e-3,
		):
		#distr_name = 'norm'
		#distr_name = 'truncnorm'
		distr_name = 'skewnorm'
		#distr_name = 'beta'
		#distr_name = 't'
		distr = getattr(stats, distr_name)

		## clean by percentile
		p = np.exp(-k*2)*6
		obse_values,_ = dropout_extreme_percentiles(self.obse[obs_indexs], p, mode='upper')
		
		scaler_kwargs = {
			'n_quantiles':min(1000, len(obse_values)),
			'output_distribution':'normal',
		}
		#scaler = prep.QuantileTransformer(**scaler_kwargs)
		scaler = prep.MinMaxScaler(feature_range=(0+eps, 1-eps))
		#scaler = prep.StandardScaler()
		
		obse_values = scaler.fit_transform(-obse_values[:,None])[:,0]
		if distr_name=='beta':
			params = distr.fit(obse_values, floc=0, fscale=1)
		else:
			params = distr.fit(obse_values)
		return {'distr':distr, 'params':params, 'scaler':scaler}
	
	def get_percentile_range(self, obs):
		return np.where(np.clip(obs, None, self.obs.max())<=self.rank_ranges[:,1])[0][0]
		
	def conditional_sample_i(self, obs_):
		d = self.distrs[self.get_percentile_range(obs_)]
		samples = d['distr'].rvs(*d['params'], size=1)
		samples = -d['scaler'].inverse_transform(samples[:,None])[:,0]
		return samples
		
	def conditional_sample(self, obs):
		return np.array([self.conditional_sample_i(obs_) for obs_ in obs])[:,0]

###################################################################################################################################################

class CurveLengthSampler():
	def __init__(self, lcdataset:dict, set_name:str, b:str,
		offset:float=0,
		):
		self.lcdataset = lcdataset
		self.lcset = lcdataset[set_name]
		self.b = b
		self.offset = offset
		self.reset()

	def reset(self):
		self.lengths = np.array([len(lcobj.get_b(self.b)) for lcobj in self.lcset.get_lcobjs()])
		uniques, count = np.unique(self.lengths, return_counts=1)
		d = {u:count[ku] for ku,u in enumerate(uniques)}
		x_pdf = np.arange(self.lengths.min(), self.lengths.max())
		self.pdf = np.array([d.get(x,0) for x in x_pdf]) + self.offset
		self.pdf = self.pdf/self.pdf.sum()
		self.cdf = np.cumsum(self.pdf)

	def sample(self, size:int):
		pdf_indexs = [np.where(self.cdf>r)[0][0] for r in np.random.uniform(size=int(size))]
		samples = pdf_indexs
		return samples
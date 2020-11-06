from __future__ import print_function
from __future__ import division
from . import C_

import math
import numpy as np
import scipy.stats as stats
from sklearn import preprocessing as prep
from flamingchoripan.datascience.statistics import dropout_extreme_percentiles, get_linspace_ranks

###################################################################################################################################################

class CustomRotor():
	def __init__(self, m, n):
		self.m = m
		assert math.atan(m)>0 and math.atan(m)<np.pi/2
		self.a = np.pi/2-math.atan(m)
		self.n = n
		self.rot = np.linalg.inv(np.array([[np.cos(self.a), np.sin(self.a)], [np.cos(self.a+np.pi/2), np.sin(self.a+np.pi/2)]]))
		self.inv_rot = np.linalg.inv(self.rot)

	def transform(self, obse, obs):
		x = np.concatenate([obse[:,None], obs[:,None]], axis=-1)
		assert x.shape[-1]==2
		x[:,1] = x[:,1]-self.n
		x = (self.rot@x.T).T
		x[:,0] = -1*x[:,0]
		return x[:,0], x[:,1]

	def inverse_transform(self, obse, obs):
		x = np.concatenate([obse[:,None], obs[:,None]], axis=-1)
		assert x.shape[-1]==2
		x[:,0] = -1*x[:,0]
		x = (self.inv_rot@x.T).T
		x[:,1] = x[:,1]+self.n
		return x[:,0], x[:,1]

###################################################################################################################################################

class ObsErrorConditionalSampler():
	def __init__(self, lcdataset:dict, set_name:str, b:str,
		samples_per_range:int=50,
		rank_threshold=0.04,
		dist_threshold=5e-4,
		neighborhood_n=10,
		):
		self.lcdataset = lcdataset
		self.lcset = lcdataset[set_name]
		self.b = b
		self.samples_per_range = samples_per_range
		self.rank_threshold = rank_threshold
		self.dist_threshold = dist_threshold
		self.neighborhood_n = neighborhood_n
		self.raw_obse = np.concatenate([lcobj.get_b(b).obse for lcobj in self.lcset.get_lcobjs()])
		self.raw_obs = np.concatenate([lcobj.get_b(b).obs for lcobj in self.lcset.get_lcobjs()])
		self.min_obs = self.raw_obs.min()
		self.min_obse = self.raw_obse.min()
		self.max_obse = self.raw_obse.max()
		assert self.min_obs>=0
		assert self.min_obse>=0
		self.reset()
		
	def get_m_n(self):
		obse = []
		obs = []
		for k1 in range(len(self.raw_obse)):
			obse1 = self.raw_obse[k1]
			obs1 = self.raw_obs[k1]
			if obs1>self.rank_threshold:
				continue
			neighborhood = 0
			for k2 in range(len(self.raw_obse)):
				if k1==k2:
					continue
				obse2 = self.raw_obse[k2]
				obs2 = self.raw_obs[k2]
				dist = np.linalg.norm(np.array([obse1-obse2, obs1-obs2]))
				if dist<=self.dist_threshold:
					neighborhood += 1
					if neighborhood>=self.neighborhood_n:
						obse.append(obse1)
						obs.append(obs1)
						break

		assert len(obse)>0
		obse = np.array(obse)
		obs = np.array(obs)
		rank_ranges, index_per_range, ranks = get_linspace_ranks(obs, self.samples_per_range)

		self.lr_x = []
		self.lr_y = []
		for k,indexs in enumerate(index_per_range):
			if len(indexs[0])==0:
				continue
			sub_obse = obse[indexs]
			sub_obs = obs[indexs]
			i = sub_obse.argmax()
			self.lr_x.append(sub_obse[i])
			self.lr_y.append(sub_obs[i])

		#print(self.lr_x, self.lr_y)
		slope, intercept, r_value, p_value, std_err = stats.linregress(self.lr_x, self.lr_y)
		self.m = slope
		self.n = intercept
		self.rotor = CustomRotor(self.m, self.n, )

	def reset(self):
		### fit diagonal line
		self.get_m_n()

		### rotate space & clip
		p = 1
		self.obse, self.obs = self.rotor.transform(self.raw_obse, self.raw_obs)
		valid_indexs = np.where(
			(self.obse>=0) &
			(self.obs>np.percentile(self.obs, p)) & (self.obs<np.percentile(self.obs, 100-p))
		)
		self.obse = self.obse[valid_indexs]
		self.obs = self.obs[valid_indexs]

		### generate obs grid
		self.rank_ranges, self.obs_indexs_per_range, self.ranks = get_linspace_ranks(self.obs, self.samples_per_range)
		self.distrs = [self.get_fitted_distr(obs_indexs, k) for k,obs_indexs in enumerate(self.obs_indexs_per_range)]
		
	def get_fitted_distr(self, obs_indexs, k):
		### clean by percentile
		obse_values = self.obse[obs_indexs]
		p = (1-np.exp(-k*0.5))*5
		obse_values,_ = dropout_extreme_percentiles(obse_values, p, mode='both')
		distr = getattr(stats, 'gamma')
		params = distr.fit(obse_values, floc=0)
		return {'distr':distr, 'params':params}
	
	def get_percentile_range(self, obs):
		return np.where(np.clip(obs, None, self.obs.max())<=self.rank_ranges[:,1])[0][0]
		
	def conditional_sample_i(self, obs):
		new_obse = np.array([0])
		new_obs = np.array([obs])
		new_obse, _ = self.rotor.inverse_transform(new_obse, new_obs)
		d = self.distrs[self.get_percentile_range(new_obs)]
		new_obse = d['distr'].rvs(*d['params'], size=1)
		new_obse, _ = self.rotor.inverse_transform(new_obse, new_obs)
		new_obse = np.clip(new_obse, self.min_obse, self.max_obse)
		return new_obse[0], new_obs[0]
		
	def conditional_sample(self, obs):
		x = np.concatenate([np.array(self.conditional_sample_i(obs_))[None] for obs_ in obs], axis=0)
		return x[:,0], x[:,1]

###################################################################################################################################################

class CurveLengthSampler():
	def __init__(self, lcdataset:dict, set_name:str, b:str,
		offset:int=0,
		):
		self.lcdataset = lcdataset
		self.lcset = lcdataset[set_name]
		self.b = b
		self.offset = offset
		self.reset()

	def reset(self):
		self.lengths = np.array([len(lcobj.get_b(self.b)) for lcobj in self.lcset.get_lcobjs()])
		uniques, count = np.unique(self.lengths+self.offset, return_counts=True)
		d = {u:count[ku] for ku,u in enumerate(uniques)}
		x_pdf = np.arange(self.lengths.min(), self.lengths.max())
		self.pdf = np.array([d.get(x,0) for x in x_pdf])
		self.pdf = self.pdf/self.pdf.sum()
		self.cdf = np.cumsum(self.pdf)

	def sample(self, size:int):
		pdf_indexs = [np.where(self.cdf>r)[0][0] for r in np.random.uniform(size=int(size))]
		samples = pdf_indexs
		return samples
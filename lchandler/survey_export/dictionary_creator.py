from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import flamingchoripan.cuteplots.plots as cplots
from dask import dataframe as dd
from flamingchoripan.progress_bars import ProgressBar
import matplotlib.pyplot as plt
from flamingchoripan.files import save_pickle, load_pickle
import flamingchoripan.cuteplots.colors as cc
from ..flux_magnitude import get_flux_from_magnitude, get_flux_error_from_magnitude
from ..flux_magnitude import get_magnitude_from_flux, get_magnitude_error_from_flux
import lchandler.dataset_classes as dsc
import lchandler.lc_classes as lcc
import pandas as pd
import copy

###################################################################################################################################################

class LightCurveDictionaryCreator():
	def __init__(self, survey_name:str, detections_df:pd.DataFrame, labels_df:pd.DataFrame, band_dictionary:dict, df_index_names:dict,
		obs_is_flux:bool=True,
		label_to_class_dict:dict=None,
		remove_negative_fluxes:bool=True,
		zero_point:float=C_.DEFAULT_ZP,
		flux_scale:float=C_.DEFAULT_FLUX_SCALE,
		maximum_samples_per_class:int=None,
		):
		'''
		zero_point: used if obs_is_flux is False
		flux_scale: used if obs_is_flux is False
		-> flux
		'''
		self.survey_name = survey_name
		self.detections_df = detections_df
		self.labels_df_original = labels_df.copy()
		self.band_dictionary = band_dictionary
		self.df_index_names = df_index_names

		self.obs_is_flux = obs_is_flux
		self.label_to_class_dict_original = (self.generate_label_to_class_dict() if label_to_class_dict is None else label_to_class_dict.copy())
		self.class_to_label_dict_original = self.generate_class_to_label_dict(self.label_to_class_dict_original)
		self.remove_negative_fluxes = remove_negative_fluxes
		self.zero_point = zero_point
		self.flux_scale = flux_scale
		self.maximum_samples_per_class = maximum_samples_per_class
		self.refresh_dataframe()

	def generate_label_to_class_dict(self):
		labels_names, counts = np.unique(self.labels_df_original[self.df_index_names['label']].values, return_counts=True)
		label_to_class_dict = {k:k for k in labels_names}
		print(f'label_to_class_dict: {label_to_class_dict}')
		return label_to_class_dict

	def generate_class_to_label_dict(self, label_to_class_dict):
		class_to_label_dict = {label_to_class_dict[k]:k for k in label_to_class_dict.keys()}
		print('label_to_class_dict:', '\n\t'+'\n\t'.join([f'{k}: {label_to_class_dict[k]}' for k in label_to_class_dict.keys()]))
		print('class_to_label_dict:', '\n\t'+'\n\t'.join([f'{k}: {class_to_label_dict[k]}' for k in class_to_label_dict.keys()]))
		return class_to_label_dict

	def get_classes_from_df(self):
		labels_names, counts = np.unique(self.labels_df[self.df_index_names['label']].values, return_counts=True)
		print(f'labels_names: {labels_names} - counts: {counts}')
		classes_names = [self.label_to_class_dict.get(label, label) for label in labels_names]
		return classes_names, labels_names, len(classes_names)

	def refresh_dataframe(self):
		self.update_labels_df([], [], {})

	def update_labels_df(self, invalid_classes:list, query_classes:list, to_merge_classes_dict:dict):
		self.label_to_class_dict = self.label_to_class_dict_original.copy() # create
		self.class_to_label_dict = self.class_to_label_dict_original.copy() # create
		self.labels_df = self.labels_df_original.copy() # create
		
		### remove invalid labels/classes
		invalid_labels = [self.class_to_label_dict[k] for k in invalid_classes]
		self.labels_df = self.labels_df[~self.labels_df[self.df_index_names['label']].isin(invalid_labels)]

		### query the desired labels/classes
		query_labels = [self.class_to_label_dict[k] for k in query_classes]
		if len(query_labels)>0:
			self.labels_df = self.labels_df[self.labels_df[self.df_index_names['label']].isin(query_labels)]

		### merge classes given the dict
		for to_merge_classes_key in to_merge_classes_dict.keys():
			self.label_to_class_dict.update({to_merge_classes_key:to_merge_classes_key})
			self.class_to_label_dict.update({to_merge_classes_key:to_merge_classes_key})
			print(f'to_merge_classes_key: {to_merge_classes_key}')
			class_fusion_list = to_merge_classes_dict[to_merge_classes_key]
			labels_fusion_list = [self.class_to_label_dict[k] for k in class_fusion_list]
			for label_fusion in labels_fusion_list:
				self.labels_df.loc[self.labels_df[self.df_index_names['label']]==label_fusion, self.df_index_names['label']] = to_merge_classes_key
		
		print(f'labels_df:\n{self.labels_df}')
		### classes
		self.classes_names, self.labels_names, self.total_classes = self.get_classes_from_df()

		### crop
		if not self.maximum_samples_per_class is None:
			print(f'cropping dataset to maximum_samples_per_class: {self.maximum_samples_per_class:,} samples')
			for class_name in self.classes_names:
				to_drop_indexs = self.labels_df[self.labels_df[self.index_names['label']]==self.class_to_label_dict[class_name]].index
				max_samples = max(0, len(to_drop_indexs) - self.maximum_samples_per_class)
				print(f'to_drop_indexs: {len(to_drop_indexs):,} - max_samples: {max_samples:,}')
				self.labels_df = self.labels_df.drop(to_drop_indexs[:max_samples])

	def plot_class_distribution(self,
		figsize=None,
		uses_log_scale:bool=False,
		band_names:list=['g','r'],
		add_band_lengths:bool=False,
		rotate_xlabel:bool=False,
		):
		#band_names = list(self.band_dictionary.keys())[:3]
		label_samples = self.labels_df[self.df_index_names['label']].values
		to_plot = {'class samples':[self.label_to_class_dict[l] for l in label_samples]}
		title = f'classes & obs distributions\n'
		title += f'survey: {self.survey_name} - class samples: {len(label_samples):,}'

		if add_band_lengths: # slow process
			band_index = self.df_index_names['band']
			for b in band_names:
				b_key = self.band_dictionary[b]
				equiv = self.labels_df[self.df_index_names['label']].to_dict()
				detections_df = self.detections_df.reset_index()
				detections_df = detections_df.drop(detections_df[getattr(detections_df, band_index)!=b_key].index)
				curve_points_samples = detections_df[self.df_index_names['oid']].map(equiv).dropna()
				curve_points_samples = curve_points_samples.values
				to_plot[f'obs samples - band: {b}'] = [self.label_to_class_dict[l] for l in curve_points_samples]

		#print(to_plot)
		cmap = cc.colorlist_to_cmap([cc.NICE_COLORS_DICT['nice_gray']]+[C_.COLOR_DICT[b] for b in band_names])
		plt_kwargs = {
			'title':title,
			'uses_log_scale':uses_log_scale,
			'cmap':cmap,
			'legend_ncol':len(band_names),
			'rotate_xlabel':rotate_xlabel,
		}
		if not figsize is None:
			plt_kwargs['figsize'] = figsize 
		fig, ax = cplots.plot_hist_labels(to_plot, self.classes_names, **plt_kwargs)
		fig.tight_layout()
		plt.show()

	def get_dict_name(self, name_parameters:dict):
		name = ''
		for k in name_parameters.keys():
			name += f'{k}-{name_parameters[k]}_'
		return name[:-1]

	#################################### EXPORT

	def get_label(self, labels_df:pd.DataFrame, lcobj_name:str, easy_label_dict:dict):
		try:
			label = labels_df[self.df_index_names['label']][lcobj_name]
			uint_label = easy_label_dict[label]
			return uint_label, True
		except:
			return None, False

	def get_radec(self, labels_df:pd.DataFrame, lcobj_name:str):
		try:
			ra = labels_df[self.df_index_names['ra']][lcobj_name]
			dec = labels_df[self.df_index_names['dec']][lcobj_name]
			return ra, dec
		except:
			return None, None

	def export_dictionary(self, description:str, save_folder:str,
		to_export_bands:list=None,
		SCPD_probs:list=[1.],
		filename_extra_parameters:dict={},
		saves_every:int=100,
		npartitions:int=C_.N_DASK,
		):
		uses_saves_every = saves_every>0
		class_dfkey = self.df_index_names['label']
		band_dfkey = self.df_index_names['band']

		lcobj_names = list(self.labels_df[class_dfkey].keys().values)
		lcobj_names.sort()
		print(f'lcobj_names examples: {lcobj_names[:10]}')

		# separate bands for optimal
		to_export_bands = (list(self.band_dictionary.keys()) if to_export_bands is None else to_export_bands)
		print(f'to_export_bands: {to_export_bands}')

		# CLEAN THE DATAFRAME PLEASE!
		index_subset = [self.df_index_names['oid']]
		integer_subset = [self.df_index_names['band']]
		float_subset = [self.df_index_names['obs_day'], self.df_index_names['obs'], self.df_index_names['obs_error']]
		subset = index_subset + integer_subset + float_subset
		detections_df = self.detections_df
		detections_df = detections_df.reset_index()[subset]
		for set_ in float_subset:
			detections_df[set_] = detections_df[set_].astype(np.float32)

		print(f'cleaning the DataFrame... - original samples: {len(detections_df):,}')
		#print('detections_df',detections_df[detections_df[self.df_index_names['oid']]=='ZTF17aabwgdw'])

		detections_df = detections_df.loc[detections_df[self.df_index_names['band']].isin([self.band_dictionary[b] for b in to_export_bands])]
		print(f'remove_invalid_bands - samples: {len(detections_df):,}')
		#print('detections_df',detections_df[detections_df[self.df_index_names['oid']]=='ZTF17aabwgdw'])

		detections_df = detections_df.loc[detections_df[self.df_index_names['oid']].isin(lcobj_names)]
		print(f'remove_invalid_classes - samples: {len(detections_df):,}')
		#print('detections_df',detections_df[detections_df[self.df_index_names['oid']]=='ZTF17aabwgdw'])

		detections_df = detections_df.dropna(how='any') # VERY SLOW
		print(f'drop_nans - samples: {len(detections_df):,}')
		#print('detections_df',detections_df[detections_df[self.df_index_names['oid']]=='ZTF17aabwgdw'])

		if self.remove_negative_fluxes:
			detections_df = detections_df.loc[detections_df[self.df_index_names['obs']] > 0]
			print(f'remove_negative_fluxes - samples: {len(detections_df):,}')

		#print('detections_df',detections_df[detections_df[self.df_index_names['oid']]=='ZTF17aabwgdw'])
		detections_df = detections_df.set_index(self.df_index_names['oid'])
		print(f'creating dask DataFrame - npartitions: {npartitions} ...')
		npartitions = npartitions
		detections_dd = dd.from_pandas(detections_df, npartitions=npartitions)

		# PREPARE NEW DATASET
		lcset = dsc.LCSet(
			{},
			self.survey_name,
			description,
			to_export_bands,
			self.classes_names,
			True,
		)
		lcdataset = dsc.LCDataset()
		lcdataset.set_lcset('raw', lcset)

		# GET FILENAME
		filename_parameters = {
			'survey':self.survey_name,
			'bands':''.join(to_export_bands),
		}
		filename_parameters.update(filename_extra_parameters)
		filedir = f'{save_folder}/{self.get_dict_name(filename_parameters)}.{C_.EXT_RAW_LIGHTCURVE}'
		print(f'filedir: {filedir}')

		# EASY DICT
		easy_label_dict = {self.class_to_label_dict[c]:kc for kc,c in enumerate(self.classes_names)}
		print(f'easy_label_dict: {easy_label_dict}')

		# START LOOP
		correct_samples = 0
		calculated_cache = 0
		bar = ProgressBar(len(lcobj_names))
		for i,lcobj_name in enumerate(lcobj_names):
			try:
				lcobj = lcc.LCO()
				pass_cond = False
				calculated_cache += 1
				y = None
				lengths_bdict = None
				try:
					obj_df = detections_dd.loc[[lcobj_name]].compute() # FAST
					for kb,b in enumerate(to_export_bands):
						band_object_df = obj_df[obj_df[band_dfkey] == self.band_dictionary[b]]
						original_lc = band_object_df[[self.df_index_names['obs_day'], self.df_index_names['obs'], self.df_index_names['obs_error']]].values.astype(np.float32)
						band_lc_flux = self.get_band(original_lc)
						#print('band_lc_flux',band_lc_flux.shape,band_lc_flux)
						lcobj.add_b(b, band_lc_flux[:,0], band_lc_flux[:,1], band_lc_flux[:,2])

					lcobj.reset_day_offset_serial()

					### get lengths
					lengths_cond = np.any([len(lcobj.get_b(b))>=C_.MIN_POINTS_LIGHTCURVE_DEFINITION for b in to_export_bands])

					### get label
					y, y_cond = self.get_label(self.labels_df, lcobj_name, easy_label_dict)
					lcobj.set_y(y)
					pass_cond = lengths_cond and y_cond

				except KeyError:
					pass_cond = False

				if pass_cond:
					lengths_bdict = lcobj.get_length_bdict()
					ra, dec = self.get_radec(self.labels_df, lcobj_name)
					lcobj.ra = ra
					lcobj.dec = dec
					lcdataset['raw'].set_lcobj(lcobj_name, lcobj)
					correct_samples += 1
				
				text = f'obj: {lcobj_name} - y: {y} - lengths_bdict: {lengths_bdict}'
				text += f' - pass_cond: {pass_cond} - correct_samples: {correct_samples:,}'
				
				bar(text)
				# saving the dict for mental sanity
				if uses_saves_every and calculated_cache>=saves_every:
					calculated_cache = 0
					save_pickle(filedir, lcdataset)
				
				if i>500:
					#assert 0, 'test'
					pass
					
			except KeyboardInterrupt:
				bar.done()
				print('stopped!')
				break

			except:
				assert 0, 'report error and proceed'

		bar.done()
		print(f'last dictionary save! filedir: {filedir}')
		save_pickle(filedir, lcdataset)
		return lcdataset

	def get_band(self, curve):
		indexs = np.argsort(curve[:,C_.DAYS_INDEX]) # sorted
		curve = curve[indexs]

		if self.obs_is_flux:
			return curve
		else:
			mag = curve[:,C_.OBS_INDEX]
			mag_error = curve[:,C_.OBS_ERROR_INDEX]
			flux = get_flux_from_magnitude(mag, self.zero_point, self.flux_scale)
			flux_error = get_flux_error_from_magnitude(mag, mag_error, self.zero_point, self.flux_scale)
			curve[:,C_.OBS_INDEX] = flux
			curve[:,C_.OBS_ERROR_INDEX] = flux_error
			return curve
from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from dask import dataframe as dd
from flamingchoripan.progress_bars import ProgressBar
from flamingchoripan.level_bars import LevelBar
import matplotlib.pyplot as plt
from flamingchoripan.files import save_pickle, load_pickle
import flamingchoripan.cuteplots.colors as cc
from ..flux_magnitude import get_flux_from_magnitude, get_flux_error_from_magnitude
from ..flux_magnitude import get_magnitude_from_flux, get_magnitude_error_from_flux
from ..plots.dataframe import plot_class_distribution_df
import lchandler.dataset_classes as dsc
import lchandler.lc_classes as lcc
import pandas as pd
import copy

###################################################################################################################################################

class LightCurveDictionaryCreator():
	def __init__(self, survey_name:str, detections_df:pd.DataFrame, labels_df:pd.DataFrame, band_dictionary:dict, df_index_names:dict,
		dataframe_obs_uses_flux:bool=True,
		label_to_class_dict:dict=None,
		zero_point:float=C_.DEFAULT_ZP,
		flux_scale:float=C_.DEFAULT_FLUX_SCALE,
		):
		'''
		zero_point: used if dataframe_obs_uses_flux is False, to transform to flux
		flux_scale: used if dataframe_obs_uses_flux is False, to transform to flux
		-> flux
		'''
		self.survey_name = survey_name
		self.detections_df = detections_df # dont copy, too much memory
		self.raw_labels_df = labels_df.copy()
		self.labels_df = labels_df.copy()
		self.band_dictionary = band_dictionary.copy()
		self.df_index_names = df_index_names.copy()

		self.dataframe_obs_uses_flux = dataframe_obs_uses_flux
		self.label_to_class_dict_original = self.generate_label_to_class_dict() if label_to_class_dict is None else label_to_class_dict.copy()
		self.class_to_label_dict_original = self.generate_class_to_label_dict(self.label_to_class_dict_original)
		self.zero_point = zero_point
		self.flux_scale = flux_scale
		self.refresh_dataframe()

	def __repr__(self):
		labels_names, counts = np.unique(self.raw_labels_df[self.df_index_names['label']].values, return_counts=True)
		txt = LevelBar({l:c for l,c in zip(labels_names, counts)}, ncols=70).__repr__()
		return txt

	def generate_label_to_class_dict(self):
		labels_names = list(set(self.raw_labels_df[self.df_index_names['label']].values))
		label_to_class_dict = {k:k for k in labels_names}
		#print(f'label_to_class_dict={label_to_class_dict}')
		return label_to_class_dict

	def generate_class_to_label_dict(self, label_to_class_dict):
		class_to_label_dict = {label_to_class_dict[k]:k for k in label_to_class_dict.keys()}
		#print('label_to_class_dict:', '\n\t'+'\n\t'.join([f'{k}={label_to_class_dict[k]}' for k in label_to_class_dict.keys()]))
		#print('class_to_label_dict:', '\n\t'+'\n\t'.join([f'{k}={class_to_label_dict[k]}' for k in class_to_label_dict.keys()]))
		return class_to_label_dict

	def get_classes_from_df(self):
		labels_names, counts = np.unique(self.labels_df[self.df_index_names['label']].values, return_counts=True)
		#print(f'labels_names={labels_names} - counts={counts}')
		class_names = [self.label_to_class_dict.get(label, label) for label in labels_names]
		return class_names, labels_names, len(class_names)

	def refresh_dataframe(self):
		self.update_labels_df()

	def update_labels_df(self,
		invalid_classes:list=[],
		query_classes:list=[],
		merge_classes_dict:dict={},
		):
		self.label_to_class_dict = self.label_to_class_dict_original.copy() # create
		self.class_to_label_dict = self.class_to_label_dict_original.copy() # create
		self.labels_df = self.raw_labels_df.copy() # create
		
		### remove invalid labels/classes
		invalid_labels = [self.class_to_label_dict[k] for k in invalid_classes]
		self.labels_df = self.labels_df[~self.labels_df[self.df_index_names['label']].isin(invalid_labels)]

		### query the desired labels/classes
		query_labels = [self.class_to_label_dict[k] for k in query_classes]
		if len(query_labels)>0:
			self.labels_df = self.labels_df[self.labels_df[self.df_index_names['label']].isin(query_labels)]

		### merge classes given the dict
		for to_merge_classes_key in merge_classes_dict.keys():
			self.label_to_class_dict.update({to_merge_classes_key:to_merge_classes_key})
			self.class_to_label_dict.update({to_merge_classes_key:to_merge_classes_key})
			#print(f'to_merge_classes_key={to_merge_classes_key}')
			class_fusion_list = merge_classes_dict[to_merge_classes_key]
			labels_fusion_list = [self.class_to_label_dict[k] for k in class_fusion_list]
			for label_fusion in labels_fusion_list:
				self.labels_df.loc[self.labels_df[self.df_index_names['label']]==label_fusion, self.df_index_names['label']] = to_merge_classes_key
		
		### classes
		self.class_names, self.labels_names, self.total_classes = self.get_classes_from_df()
		return

	def plot_class_distribution(self,
		figsize=None,
		uses_log_scale:bool=False,
		band_names:list=['g','r'],
		add_band_lengths:bool=False,
		rotate_xlabel:bool=False,
		caption=None,
		):
		plot_class_distribution_df(self.labels_df, self.detections_df, self.label_to_class_dict, self.df_index_names, self.class_names, self.band_dictionary, self.survey_name,
			figsize,
			uses_log_scale,
			band_names,
			add_band_lengths,
			rotate_xlabel,
			caption,
		)

	def get_dict_name(self, name_parameters:dict):
		name = ''
		for k in name_parameters.keys():
			name += f'{k}={name_parameters[k]}~'
		return name[:-1]

	def get_label(self, labels_df:pd.DataFrame, lcobj_name:str, easy_label_dict:dict):
		label = labels_df[self.df_index_names['label']][lcobj_name]
		uint_label = easy_label_dict[label]
		return uint_label

	def get_radec(self, labels_df:pd.DataFrame, lcobj_name:str):
		try:
			ra = labels_df[self.df_index_names['ra']][lcobj_name]
			dec = labels_df[self.df_index_names['dec']][lcobj_name]
			return ra, dec
		except:
			return None, None

	def get_band(self, curve):
		indexs = np.argsort(curve[:,C_.DAYS_INDEX]) # need to be sorted
		curve = curve[indexs]

		if self.dataframe_obs_uses_flux:
			return curve
		else:
			mag = curve[:,C_.OBS_INDEX]
			mag_error = curve[:,C_.OBSE_INDEX]
			flux = get_flux_from_magnitude(mag, self.zero_point, self.flux_scale)
			flux_error = get_flux_error_from_magnitude(mag, mag_error, self.zero_point, self.flux_scale)
			curve[:,C_.OBS_INDEX] = flux
			curve[:,C_.OBSE_INDEX] = flux_error
			return curve

	###################################################################################################################################################

	def export_dictionary(self, description:str, save_folder:str,
		band_names:list=None,
		filename_extra_parameters:dict={},
		npartitions:int=C_.N_JOBS,
		any_band_points=C_.MIN_POINTS_LIGHTCURVE_SURVEY_EXPORT,
		outliers_df=None,
		):
		class_dfkey = self.df_index_names['label']
		band_dfkey = self.df_index_names['band']

		### separate bands for optimal
		band_names = list(self.band_dictionary.keys()) if band_names is None else band_names
		print(f'band_names={band_names}')

		### clean dataframe to speed up thing in the objects search
		detections_df = self.detections_df.reset_index()
		print(f'cleaning the DataFrame - samples={len(detections_df):,}')
		#print('detections_df',detections_df[detections_df[self.df_index_names['oid']]=='ZTF17aabwgdw'])

		detections_ddf = dd.from_pandas(detections_df, npartitions=npartitions)
		detections_df = detections_ddf.loc[detections_ddf[self.df_index_names['band']].isin([self.band_dictionary[b] for b in band_names])].compute()
		print(f'remove_invalid_bands > samples={len(detections_df):,}')

		detections_ddf = dd.from_pandas(detections_df, npartitions=npartitions)
		detections_df = detections_ddf.loc[detections_ddf[self.df_index_names['oid']].isin(list(set(self.labels_df.index)))].compute()
		print(f'remove_invalid_classes > samples={len(detections_df):,}')

		detections_ddf = dd.from_pandas(detections_df, npartitions=npartitions)
		detections_df = detections_ddf.loc[detections_ddf[self.df_index_names['obs']]>0].compute()
		print(f'remove_negative_obs > samples={len(detections_df):,}')
		detections_df = detections_df.set_index(self.df_index_names['oid'])

		### prepare dataset
		lcset = dsc.LCSet(
			{},
			self.survey_name,
			description,
			band_names,
			self.class_names,
			True,
		)
		lcdataset = dsc.LCDataset()
		lcdataset.set_lcset('outliers', lcset.copy())
		lcdataset.set_lcset('faint', lcset.copy())
		lcdataset.set_lcset('raw', lcset.copy())

		### get filename
		filename_parameters = {
			'survey':self.survey_name,
			'bands':''.join(band_names),
		}
		filename_parameters.update(filename_extra_parameters)
		save_filedir = f'{save_folder}/{self.get_dict_name(filename_parameters)}.{C_.EXT_RAW_LIGHTCURVE}'
		print(f'save_filedir={save_filedir}')

		### easy variables
		outliers = [] if outliers_df is None else list(outliers_df['outliers'].values) 
		easy_label_dict = {self.class_to_label_dict[c]:kc for kc,c in enumerate(self.class_names)}
		print(f'easy_label_dict={easy_label_dict}')

		# start loop
		correct_samples = 0
		detections_ddf = dd.from_pandas(detections_df, npartitions=npartitions)
		lcobj_names = sorted(list(set(detections_df.index)))
		bar = ProgressBar(len(lcobj_names))
		for k,lcobj_name in enumerate(lcobj_names):
			try:
				lcobj = lcc.LCO()

				### get detections
				obj_df = detections_ddf.loc[lcobj_name].compute() # FAST
				for kb,b in enumerate(band_names):
					band_object_df = obj_df[obj_df[band_dfkey] == self.band_dictionary[b]]
					original_lc = band_object_df[[self.df_index_names['obs_day'], self.df_index_names['obs'], self.df_index_names['obs_error']]].values
					band_lc_flux = self.get_band(original_lc)
					lcobj.add_b(b, band_lc_flux[:,0], band_lc_flux[:,1], band_lc_flux[:,2])

				lcobj.clean_small_cadence()
				lcobj.reset_day_offset_serial()

				### get label
				y = self.get_label(self.labels_df, lcobj_name, easy_label_dict)
				lcobj.set_y(y)

				### check lengths
				if lcobj.any_band_eqover_length(any_band_points):
					ra, dec = self.get_radec(self.labels_df, lcobj_name)
					lcobj.ra = ra
					lcobj.dec = dec
					lcset_name = 'raw'
					if lcobj_name in outliers:
						lcset_name = 'outliers'
					elif lcobj.get_snr()<C_.MIN_SNR:
						lcset_name = 'faint'
					lcdataset[lcset_name].set_lcobj(lcobj_name, lcobj)
					correct_samples += 1
				else:
					pass
					#print(lcobj_name)
				bar(f'obj={lcobj_name} - y={y} - c={self.class_names[y]} - lengths_bdict={lcobj.get_length_bdict()} - correct_samples (any-band>={any_band_points})={correct_samples:,}')
					
			except KeyboardInterrupt:
				bar.done()
				print('stopped!')
				break

		bar.done()
		save_pickle(save_filedir, lcdataset)
		return lcdataset
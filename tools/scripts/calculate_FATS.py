#!/usr/bin/env python3
import os, sys
sys.path.append('../../TESIS')

import argparse
from flamingChoripan.myUtils.files import load_pickle, save_pickle
from flamingChoripan.myUtils.time import Cronometer
from flamingChoripan.myUtils.progress_bars import ProgressBar

import signal
from multiprocessing import Pool
import multiprocessing as mp

import turbofats as FATS
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from tqdm import tqdm
from src import C_

###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################

#FATS_EXT = 'fats'
FATS_EXT = 'tfatsh' # turbo fats harmonics
FATS_FEATURES = 58

def get_features_list():
		gal_feat_list = [
			'Amplitude',
			'AndersonDarling',
			'Autocor_length',
			'Beyond1Std',
			'CAR_sigma',
			'CAR_mean',
			'CAR_tau',
			'Con',
			'Eta_e',
			'FluxPercentileRatioMid20',
			'FluxPercentileRatioMid35',
			'FluxPercentileRatioMid50',
			'FluxPercentileRatioMid65',
			'FluxPercentileRatioMid80',
			'Gskew',
			'LinearTrend',
			'MaxSlope',
			'Mean',
			'Meanvariance',
			'MedianAbsDev',
			'MedianBRP',
			'PairSlopeTrend',
			'PercentAmplitude',
			'PercentDifferenceFluxPercentile',
			'PeriodLS',
			'Period_fit',
			'Psi_CS',
			'Psi_eta',
			'Q31',
			'Rcs',
			'Skew',
			'SmallKurtosis',
			'Std',
			'StetsonK',
			#'VariabilityIndex',
		]

		# This features are slow, refactor?
		harmonics_features = []
		for f in range(3): # 3
			for index in range(4): # 4
				harmonics_features.append("Freq" + str(f+1) + "_harmonics_amplitude_" + str(index))
				harmonics_features.append("Freq" + str(f+1) + "_harmonics_rel_phase_" + str(index))

		return gal_feat_list + harmonics_features

featureList = get_features_list()
excludeList = [
	'Color',
	'Eta_color',
	'Q31_color',
	'StetsonJ',
	'StetsonL',
	'StetsonK_AC',
	'SlottedA_length',
]
data_format = ['magnitude','time','error'] # <<<<<<<<<< SUPERBUG!
FATS_FS = FATS.FeatureSpace(featureList=featureList, excludeList=excludeList, Data=data_format)

class GracefulExiter():
	def __init__(self):
		self.state = False
		signal.signal(signal.SIGINT, self.change_state)

	def change_state(self, signum, frame):
		print(' > KeyboardInterrupt!!!')
		#print("exit flag set to True (repeat to exit now)")
		signal.signal(signal.SIGINT, signal.SIG_DFL)
		self.state = True

	def exit(self):
		return self.state

def init_worker():
	# https://stackoverflow.com/questions/5114292/break-interrupt-a-time-sleep-in-python
	# https://stackoverflow.com/questions/1408356/keyboard-interrupts-with-pythons-multiprocessing-pool
	signal.signal(signal.SIGINT, signal.SIG_IGN)

# Disable
def blockPrint():
	sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
	sys.stdout = sys.__stdout__

def get_list_chunks(list_, chuncks_size):
	chuncks = []
	#l = len(list_)//min(chuncks_size, len(list_))
	l = chuncks_size
	index = 0
	while index<len(list_):
		chuncks.append(list_[index:index+l])
		index += l
	return chuncks

def get_features_from_lc(lc):
	try:
		features = FATS_FS.calculateFeature(lc)
		features = np.array(list(features.result().tolist()))
		return features
	except AssertionError as error: # from TurboFATS
		print('>>>>> AssertionError... ', end='')
	except ValueError as error: # from TurboFATS
		print('>>>>> ValueError... ', end='')
	except RuntimeError as error: # from TurboFATS
		# Optimal parameters not found: gtol=0.000000 is too small, func(x) is orthogonal to the columns of
		print('>>>>> RuntimeError... ', end='')
	print('THIS IS A WEIRD ERROR ... <<<<<<<')
	return np.full((FATS_FEATURES,), NAN_VALUE)

def curve_to_FATS(curve):
	lc_arr = np.array([curve[:,C_.OBS_INDEX], curve[:,C_.DAYS_INDEX], curve[:,C_.OBS_ERROR_INDEX]], dtype=C_.PYTORCH_FLOAT_FORMAT) # <<<<<<<<<< SUPERBUG!
	#print('lc_arr',lc_arr.shape)
	curve_len = lc_arr.shape[-1]
	features = np.full((curve_len, FATS_FEATURES), NAN_VALUE, dtype=C_.PYTORCH_FLOAT_FORMAT)
	total_nans = 0

	blockPrint()
	for k in range(C_.MIN_POINTS_LIGHTCURVE_DEFINITION_FATS-1, curve_len):
		sub_lc_arr = lc_arr[:,:k+1] # nxt
		#print('sub_lc_arr',sub_lc_arr.shape,'mean',np.mean(sub_lc_arr[1,:]))
		#print('k',k,'curve_len',curve_len,'sub_lc_arr',sub_lc_arr,sub_lc_arr.shape)
		fats = get_features_from_lc(sub_lc_arr) # VERY SLOW
		features[k,:] = fats
		total_nans += np.count_nonzero(np.isnan(fats))
	enablePrint()

	test_f_ind = 17
	#print(featureList[ind])
	#print('lc_arr.shape',lc_arr.shape)	
	#print('features',features.shape)		
	#print('features',features.shape,features[:,test_f_ind])
	return features, total_nans

def hard_process(args):
	obj_dic, obj_name = args
	obj_dic['xf_mb'] = {}
	x_mb = obj_dic['x_mb']
	
	total_nans = 0
	for b in x_mb.keys():
		curve = x_mb[b]
		fats, nans = curve_to_FATS(curve) # SLOW
		total_nans += nans

		if not np.isnan(NAN_VALUE):
			nan_inds = np.where(np.isnan(fats))
			fats[nan_inds] = NAN_VALUE

		#print('curve: {} - fats: {}'.format(curve.shape, fats.shape))
		obj_dic['xf_mb'][b] = fats
		#obj_dic['obs_days_mb'][b] = curve[:,C_.DAYS_INDEX] # add days
	
	#print('new_obj: {}'.format(new_obj))
	return obj_dic, obj_name, total_nans

###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################

NAN_VALUE = np.nan
NAN_VALUE = 0
NAN_VALUE = -999

def load_lcdic(filename):
	assert filename.split('.')[-1]==C_.EXT_SPLIT_LIGHTCURVE_DIC
	return load_pickle(filename)

if __name__== "__main__":
	parser = argparse.ArgumentParser('calculate_FATS')
	parser.add_argument('-filedir','--filedir',  type=str, default=None)
	#parser.add_argument('-dbn','--database-name',  type=str, default=None, help='name of database to FATS')
	#parser.add_argument('--file-format',  type=str, default='lcdic', help='file_format')
	args = parser.parse_args()

	filedir = args.filedir
	root_folder = '/'.join(filedir.split('/')[:-1])
	filename = '.'.join(filedir.split('/')[-1].split('.')[:-1])
	lcdic = load_lcdic(filedir)
	print('keys:',lcdic.keys())

	#####################################

	#processes = 0 # NO-MULTITHREAD
	#processes = 8
	processes = mp.cpu_count()
	#processes = 2
	#chunck_size = 2
	chunck_size = processes

	FATS_info = {
		'fats_expected_features':FATS_FEATURES,
		'fats_process_time':0,
		'fats_featureList':featureList,
		'fats_excludeList':excludeList,
		'fats_nan_value':NAN_VALUE,
		'fats_total_nans':0,
	}
	save_filename = '{}/{}.{}'.format(root_folder, filename, C_.EXT_FATS_LIGHTCURVE_DIC)
	print('save_filename: {}'.format(save_filename))
	print('='*50)
	for set_name in ['train_data', 'vali_data']:
		data_dic = lcdic[set_name]
		total_keys = list(data_dic.keys())
		lengths = []
		print('processing: {}'.format(set_name))
		print('ordering keys by length to optimize multithread')
		for key in total_keys:
			length = [len(data_dic[key]['x_mb'][b]) for b in lcdic['band_names']]
			lengths.append(sum(length))

		sorted_indexs = np.argsort(np.array(lengths))
		total_keys = list(np.array(total_keys)[sorted_indexs])

		chuncks = get_list_chunks(total_keys, chunck_size)
		args = [processes, len(total_keys), chunck_size, len(chuncks)]
		print('processes: {} - total_keys: {:,} - chunck_size: {} - chuncks: {:,}'.format(*args))
		print('-'*50)

		#####################################

		pool = Pool(processes, init_worker)
		saves_every = len(total_keys)//10
		#saves_every = 100
		print('saves_every: {}'.format(saves_every))

		bar = ProgressBar(len(chuncks))
		flag = GracefulExiter()
		processed_samples = 0
		time_cumulated = 0
		cr = Cronometer()
		for chunk_id, keys in enumerate(chuncks):
			try:
				total_keys = len(keys)
				FATS_info['fats_process_time'] = cr.dt()
				args = [chunk_id, processed_samples, FATS_info['fats_process_time'], FATS_info['fats_total_nans']]
				bar_text = 'chunk_id: {:,} - p_samples: {:,} - t_cum: {:2f} [mins] - nans: {:,}'.format(*args)
				bar(bar_text)
				
				input_args = [(data_dic[key], key) for key in keys]
				return_list = pool.map(hard_process, input_args) # MAPPING
				processed_samples += total_keys
				#raise Exception('Test')

				for returned in return_list:
					obj_dic, obj_name, total_nans = returned
					data_dic[obj_name] = obj_dic
					FATS_info['fats_total_nans'] += total_nans

				if processed_samples > saves_every:
					processed_samples = 0
					lcdic.update(FATS_info)
					save_pickle(save_filename, lcdic)

				if flag.exit():
					#bar.done()
					break

			except:
				bar.done() # END BAR
				pool.terminate();pool.join() # END POOL
				raise # exit

		bar.done() # END BAR
		pool.terminate();pool.join() # END POOL

	print('last save!!')
	lcdic.update(FATS_info)
	save_pickle(save_filename, lcdic)
from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import pandas as pd
from dask import dataframe as dd
import warnings

###################################################################################################################################################

def get_valid_classes_objs(df, df_index_names, target_classes):
	new_df = df.reset_index()
	valid_objs = new_df.loc[new_df[df_index_names['label']].isin(target_classes)][df_index_names['oid']]
	return list(valid_objs.values)

def subset_df_columns(df, subset_cols):
	df_cols = list(df.columns)
	return df[[c for c in subset_cols if c in df_cols]]

def delete_invalid_detections(df, index_name,
	uses_corr:True,
	npartitions=C_.N_DASK,
	):
	days_col = 'mjd'
	obs_col = 'magpsf_corr' if uses_corr else 'magpsf'
	obse_col = 'sigmapsf_corr' if uses_corr else 'sigmapsf'
	ddf = dd.from_pandas(df, npartitions=npartitions)
	if uses_corr:
		df = ddf[~(
			(ddf['isdiffpos']==-1) | # bad photometry
			(ddf[days_col].isna()) | # delete nans
			(ddf[obs_col].isna()) | # delete nans
			(ddf[obse_col].isna()) | # delete nans
			(ddf[obse_col]>=100) # 100 error only with corr version
		)].compute() # FAST
	else:
		df = ddf[~(
			(ddf['isdiffpos']==-1) | # bad photometry
			(ddf[days_col].isna()) | # delete nans
			(ddf[obs_col].isna()) | # delete nans
			(ddf[obse_col].isna()) # delete nans
		)].compute() # FAST
	return df

def delete_invalid_objs(df, new_index_name,
	npartitions=C_.N_DASK,
	):
	df = df.set_index([new_index_name])
	ddf = dd.from_pandas(df, npartitions=npartitions)
	invalid_df = ddf[(ddf['isdiffpos']==-1)].compute() # FAST
	invalid_objs = list(set(invalid_df.index))
	df = df.drop(invalid_objs)
	return df.reset_index()

def keep_only_valid_objs(df, valid_objs):
	return df[df.index.isin(valid_objs)]

def drop_duplicates_mjd(df, new_index_name,
	npartitions=C_.N_DASK,
	):
	ddf = dd.from_pandas(df, npartitions=npartitions)
	return ddf.drop_duplicates(subset=[new_index_name, 'fid','mjd']).compute() # FAST

def drop_duplicates(df,
	npartitions=C_.N_DASK,
	):
	ddf = dd.from_pandas(df, npartitions=npartitions)
	return ddf.drop_duplicates().compute() # FAST

###################################################################################################################################################

def process_df_detections(df, index_name, new_index_name, detections_cols,
	uses_corr=True,
	npartitions=C_.N_DASK,
	clean_detections=True,
	):
	assert df.index.name==index_name
	if not uses_corr:
		warnings.warn('only use uses_corr=False with SNe objects')
	df.index.rename(new_index_name, inplace=True) # rename index
	df = df.reset_index()
	df = drop_duplicates_mjd(df, new_index_name) # delete more samples
	#df = drop_duplicates(df) # some samples can bypass this as there are different obs in same days
	if clean_detections:
		df = delete_invalid_detections(df, new_index_name, uses_corr, npartitions)
		#df = delete_invalid_objs(df, new_index_name) # deletes a lot of objects
		
	df = subset_df_columns(df, detections_cols+[new_index_name]) # sub sample columns
	df = df.set_index([new_index_name])
	objs = list(set(df.index))
	return df, objs

def process_df_labels(df, new_index_name, det_objs):
	df = drop_duplicates(df)
	df = df.set_index([new_index_name])
	objs = list(set(df.index))
	return df, objs
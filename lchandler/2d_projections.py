from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.manifold import TSNE
from umap import UMAP

###################################################################################################################################################

def get_fitted_umap_b(lcset, b:str):
	random_state = 42
	umap_kwargs = {
		#'metric':'cosine', # default: euclidean
		'min_dist':0.1, # default: 0.1
		'n_neighbors':50, # default: 15
		'random_state':random_state,
		'transform_seed':random_state,
	}
	tsne_kwargs = {
		'perplexity':50.0, # default: 30
		'random_state':random_state,
	}
	pm_scaler = QuantileTransformer(n_quantiles=5000, random_state=random_state, output_distribution='normal')
	#pm_scaler = StandardScaler()
	pm_scaler = MinMaxScaler()
	pm_umap = UMAP(n_components=2, **umap_kwargs)
	pm_tsne = TSNE(n_components=2, **tsne_kwargs)
	#pm_pca = FastICA(n_components=2)#, kernel='rbf', gamma=0.1)
	pm_pca = PCA(n_components=3)
	#pm_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)

	valid_keys = [key for key in lcset.data_keys() if not lcset.data[key].get_b(b).pm_args is None]
	pm_args_tensor_x = np.array([[lcset.data[key].get_b(b).pm_args[pk] for pk in lcset.data[key].get_b(b).pm_args.keys()] for key in valid_keys])
	pm_args_tensor_y = np.array([lcset.data[key].y for key in valid_keys])

	#pm_scaler.fit(pm_args_tensor_x) # fit
	#pm_args_tensor_x = pm_scaler.transform(pm_args_tensor_x)
	#pm_pca.fit(pm_args_tensor_x)#, y=pm_args_tensor_y) # fit
	#pm_umap.fit(pm_args_tensor_x)#, y=pm_args_tensor_y) # fit
	#pm_tsne.fit(pm_args_tensor_x)#, y=pm_args_tensor_y) # fit
	return {'scaler':pm_scaler, 'umap':pm_umap, 'tsne':pm_tsne, 'pca':pm_pca}

def get_fitted_umap(lcdataset, set_name):
	lcset = lcdataset.get(set_name)
	pm_umap_results = {}
	for kb,b in enumerate(lcset.band_names):
		pm_umap = get_fitted_umap_b(lcset, b)
		pm_umap_results[b] = pm_umap
	return pm_umap_results

def get_transformed_umap(lcdataset, pm_umap_results, set_name_train, set_name_test):
	lcset_train = lcdataset.get(set_name_train)
	lcset_test = lcdataset.get(set_name_test)
	pm_args_embd_results_train = {}
	pm_args_embd_results_test = {}
	for kb,b in enumerate(lcset_train.band_names):
		pm_scaler = pm_umap_results[b]['scaler']
		pm_umap = pm_umap_results[b]['umap']
		pm_tsne = pm_umap_results[b]['tsne']
		pm_pca = pm_umap_results[b]['pca']
		
		valid_keys_train = [key for key in lcset_train.data_keys() if not lcset_train.data[key].get_b(b).pm_args is None]
		pm_args_tensor_x_train = np.array([[lcset_train.data[key].get_b(b).pm_args[pk] for pk in lcset_train.data[key].get_b(b).pm_args.keys()] for key in valid_keys_train])
		
		valid_keys_test = [key for key in lcset_test.data_keys() if not lcset_test.data[key].get_b(b).pm_args is None]
		pm_args_tensor_x_test = np.array([[lcset_test.data[key].get_b(b).pm_args[pk] for pk in lcset_test.data[key].get_b(b).pm_args.keys()] for key in valid_keys_test])

		pm_args_tensor_x = np.concatenate([pm_args_tensor_x_train, pm_args_tensor_x_test], axis=0) # merge for reproducibility
		pm_args_tensor_x = pm_scaler.fit_transform(pm_args_tensor_x)
		#pm_args_x_umap = np.array([pm_umap.transform(pm_args_tensor_x[i][None,...]) for i in range(len(pm_args_tensor_x))])
		pm_args_x_pca = pm_pca.fit_transform(pm_args_tensor_x)

		pm_args_x_umap = pm_umap.fit_transform(pm_args_x_pca)
		pm_args_x_tsne = pm_tsne.fit_transform(pm_args_x_pca)
		
		#pm_args_x_pca = pm_args_tensor_x[:,:2]

		pm_args_embd_results_train[b] = {
			'keys':valid_keys_train,
			'x_umap':pm_args_x_umap[:len(valid_keys_train)],
			'x_tsne':pm_args_x_tsne[:len(valid_keys_train)],
			'x_pca':pm_args_x_pca[:len(valid_keys_train)],
			}
		pm_args_embd_results_test[b] = {
			'keys':valid_keys_test,
			'x_umap':pm_args_x_umap[-len(valid_keys_test):],
			'x_tsne':pm_args_x_tsne[-len(valid_keys_test):],
			'x_pca':pm_args_x_pca[-len(valid_keys_test):],
			}

	return pm_args_embd_results_train, pm_args_embd_results_test
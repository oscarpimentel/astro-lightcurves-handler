import numpy as np

###################################################################################################################################################

### EXPORT
N_DASK = 4

### SYNTHETIC
NORMAL_STD_SCALE = 0.5
EPS = 1e-12

### INDEXS
DAYS_INDEX = 0
OBS_INDEX = 1
OBS_ERROR_INDEX = 2

INDEXS_DICT = {
	'days':DAYS_INDEX,
	'obs':OBS_INDEX,
	'obse':OBS_ERROR_INDEX,
}
SHORT_NAME_DICT = {
	'days':'days',
	'd_days':'$\\Delta$days',

	'obs':'obs',
	'log_obs':'log-obs',
	'd_obs':'$\\Delta$obs',

	'obse':'obs errors',
	'log_obse':'log-obs errors',
	'd_obse':'$\\Delta$obs errors',
}
LONG_NAME_DICT = {
	'days':'days',
	'd_days':'$\\Delta$days',

	'obs':'observations',
	'log_obs':'log-observations',
	'd_obs':'$\\Delta$observations',

	'obse':'observation errors',
	'log_obse':'log-observation errors',
	'd_obse':'$\\Delta$observation errors',
}
SYMBOLS_DICT = {
	'days':'$\{\{t_{ij}\}_j^{L_i}\}_i^N$',
	'd_days':'$\{\{\\Delta t_{ij}\}_j^{L_i}\}_i^N$',

	'obs':'$\{\{x_{ij}\}_j^{L_i}\}_i^N$',
	'log_obs':'$\{\{\log(x_{ij})\}_j^{L_i}\}_i^N$',
	'd_obs':'$\{\{\\Delta x_{ij}\}_j^{L_i}\}_i^N$',

	'obse':'$\{\{\sigma_{xij}\}_j^{L_i}\}_i^N$',
	'log_obse':'$\{\{\log(\sigma_{xij})\}_j^{L_i}\}_i^N$',
	'd_obse':'$\{\{\\Delta \sigma_{xij}\}_j^{L_i}\}_i^N$',
}
XLABEL_DICT = {
	'days':'$t_{ij}$ values',
	'd_days':'$\\Delta t_{ij}$ values',

	'obs':'$x_{ij}$ values',
	'log_obs':'$\log(x_{ij})$ values',
	'd_obs':'$\\Delta x_{ij}$ values',

	'obse':'$\sigma_{xij}$ values',
	'log_obse':'$\log(\sigma_{xij})$ values',
	'd_obse':'$\\Delta \sigma_{xij}$ values',
}

### LENGTHS
MIN_POINTS_LIGHTCURVE_DEFINITION = 5 # 2, 3
MIN_POINTS_LIGHTCURVE_DEFINITION_FATS = 4
MIN_POINTS_LIGHTCURVE_TO_PMFIT = 5

### FILE TYPES
EXT_RAW_LIGHTCURVE = 'rawlcd' # no split, as RAW ZTF/FSNes
EXT_SPLIT_LIGHTCURVE = 'slcd' # with proper train/vali split, vali is balanced in classes
EXT_PARAMETRIC_LIGHTCURVE = 'plcd' # with sigma clipping and fitted parametric model
EXT_FATS_LIGHTCURVE = 'flcd' # with sigma clipping and FATS

### BANDS
COLOR_DICT = {
	'u':'#0396A6',
	'g':'#6ABE4F',
	'r':'#F25E5E',
	'i':'#B6508A',
	'z':'#F2E749',
	'y':'#404040',
}

#DEFAULT_SCPD_PS = [0.0, 0.25, 0.5, 0.75]
DEFAULT_ZP = 48.6
#DEFAULT_FLUX_SCALE = 1
DEFAULT_FLUX_SCALE = 1e26
DEFAULT_MAG_SCALE = 1
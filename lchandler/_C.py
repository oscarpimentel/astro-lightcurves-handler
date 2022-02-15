import numpy as np
from fuzzytools.matplotlib.colors import hexcolor2rbg

###################################################################################################################################################
EPS = 1e-5

DEFAULT_ZP = 48.6
DEFAULT_FLUX_SCALE = 1e26 # 1e26
CHECK = False
N_JOBS = -1

SERIAL_CHAR = '*'

### THRESHOLDS
MIN_POINTS_LIGHTCURVE_SURVEY_EXPORT = 5
MIN_POINTS_LIGHTCURVE_DEFINITION = 5
CADENCE_THRESHOLD = 12./24.

### FILE TYPES
EXT_RAW_LIGHTCURVE = 'ralcds' # no split, as raw light-curve-data-set
EXT_SPLIT_LIGHTCURVE = 'splcds' # with proper train/val/test split light-curve-data-set

### DF
SET_NAME_STR = 'dataset'

### INDEXS
DAYS_INDEX = 0
OBS_INDEX = 1
OBSE_INDEX = 2

### DICTS
INDEXS_DICT = {
	'days':DAYS_INDEX,
	'obs':OBS_INDEX,
	'obse':OBSE_INDEX,
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

	'obse':'observation-errors',
	'log_obse':'log-observation-errors',
	'd_obse':'$\\Delta$observation errors',
}
SYMBOLS_DICT = {
	'days':'$\{\{t_{ij}\}_j^{L_i}\}_i^N$',
	'd_days':'$\{\{\\Delta t_{ij}\}_j\\in J_i\}_{i=1}^N$',

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

COLOR_DICT = {
	'u':'#0396A6',
	'g':'#6ABE4F',
	'r':'#F25E5E',
	'i':'#B6508A',
	'z':'#F2E749',
	'y':'#404040',
	SERIAL_CHAR:'k',
}

CLASSES_STYLES = {
	'SLSN':{'c':hexcolor2rbg('#F23535'), 'marker':'^', 'markerprop':1.5, 'zorder':0},
	'SNIbc':{'c':[c/255 for c in (75, 131, 242)], 'marker':'o', 'markerprop':1, 'zorder':-1},
	'SNII*':{'c':hexcolor2rbg('#F2A922'), 'marker':'p', 'markerprop':1, 'zorder':-2},
	'SNIa':{'c':[c/255 for c in (89, 89, 89)], 'marker':'D', 'markerprop':.9, 'zorder':-3},
	}
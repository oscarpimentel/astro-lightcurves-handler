from __future__ import print_function
from __future__ import division
from . import C_

import random

def split_lcdataset(lcdataset,
	train_size:float=0.8,
	):
	train_populations_dict = {c:int(lcdataset.raw.get_populations()[c]*train_size) for c in lcdataset.raw.class_names}

	lcdataset.set_raw_train(lcdataset.raw.copy({}))
	lcdataset.set_raw_val(lcdataset.raw.copy({}))
	
	for kc,c in enumerate(lcdataset.raw.class_names):
		class_keys = [key for key in lcdataset.raw.data_keys() if lcdataset.raw.data[key].y==kc]
		nct = train_populations_dict[c]
		train_keys, val_keys = class_keys[:nct], class_keys[nct:]
		[lcdataset.raw_train.data.update({key:lcdataset.raw.data[key].copy()}) for key in train_keys]
		[lcdataset.raw_val.data.update({key:lcdataset.raw.data[key].copy()}) for key in val_keys]

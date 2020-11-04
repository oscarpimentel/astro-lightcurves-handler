from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from flamingchoripan.progress_bars import ProgressBar
from .synthetic_curve_generators import SynSNeGeneratorCF, SynSNeGeneratorMCMC

###################################################################################################################################################

def generate_synthetic_dataset(lcdataset, set_name, obse_sampler_bdict, length_sampler_bdict,
	method='curve_fit',
	synthetic_samples_per_curve:float=2,
	add_original=True,
	):
	generator_class_dict = {
		'curve_fit':SynSNeGeneratorCF,
		'mcmc':SynSNeGeneratorMCMC,
	}
	lcset = lcdataset[set_name]
	lcobj_names = lcset.get_lcobj_names()
	band_names = lcset.band_names

	synth_lcset = lcset.copy({}) # copy
	lcdataset.set_lcset(f'synth_{set_name}', synth_lcset)

	bar = ProgressBar(len(lcset))
	for lcobj_name in lcobj_names:
		#bar(f'add_original: {add_original} - set_name: {set_name} - lcobj_name: {lcobj_name} - lcobj_name: {lcobj_name} - pm_args: {pm_args}')
		bar(f'method: {method} - add_original: {add_original} - set_name: {set_name} - lcobj_name: {lcobj_name}')
		lcobj = lcset[lcobj_name]
		sne_generator = generator_class_dict[method](lcobj, band_names, obse_sampler_bdict, length_sampler_bdict)
		new_lcobjs = sne_generator.sample_curves(synthetic_samples_per_curve)
		for knl,new_lcobj in enumerate(new_lcobjs):
			new_lcobj_name = f'{lcobj_name}.{knl+1}'
			new_lcobj.reset_day_offset_serial()
			synth_lcset.set_lcobj(new_lcobj_name, new_lcobj)

		if add_original:
			synth_lcset.set_lcobj(f'{lcobj_name}.0', lcobj.copy())

	bar.done()
	return
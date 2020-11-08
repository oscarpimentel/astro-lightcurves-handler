from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from flamingchoripan.progress_bars import ProgressBar
from flamingchoripan.files import save_pickle
from .synthetic_curve_generators import SynSNeGeneratorCF, SynSNeGeneratorMCMC
from ..plots.lc import plot_synthetic_samples

###################################################################################################################################################

def get_syn_sne_generator(method_name):
	if method_name=='curve_fit':
		return SynSNeGeneratorCF,
	elif method_name=='mcmc':
		return SynSNeGeneratorMCMC

def generate_synthetic_dataset(lcdataset, set_name, obse_sampler_bdict, length_sampler_bdict,
	method='curve_fit',
	synthetic_samples_per_curve:float=4,
	add_original=True,
	ignored_lcobj_names=[],
	save_rootdir=None,
	):
	lcset = lcdataset[set_name]
	lcobj_names = lcset.get_lcobj_names()
	band_names = lcset.band_names
	class_names = lcset.class_names

	synth_lcset = lcset.copy({}) # copy
	lcdataset.set_lcset(f'synth_{set_name}', synth_lcset)

	fit_errors_bdict_list = []
	can_be_in_loop = True
	bar = ProgressBar(len(lcset))
	for lcobj_name in lcobj_names:
		try:
			if can_be_in_loop:
				#bar(f'add_original: {add_original} - set_name: {set_name} - lcobj_name: {lcobj_name} - lcobj_name: {lcobj_name} - pm_args: {pm_args}')
				bar(f'method: {method} - add_original: {add_original} - set_name: {set_name} - lcobj_name: {lcobj_name}')
				if lcobj_name in ignored_lcobj_names:
					continue
				lcobj = lcset[lcobj_name]
				sne_generator = get_syn_sne_generator(method)(lcobj, class_names, band_names, obse_sampler_bdict, length_sampler_bdict)
				new_lcobjs, new_lcobjs_pm, fit_errors_bdict = sne_generator.sample_curves(synthetic_samples_per_curve)
				fit_errors_bdict_list.append(fit_errors_bdict)
				save_filedir = None if save_rootdir is None else f'{save_rootdir}/{lcset.survey}/{method}/{lcobj_name}.ferror'
				to_save = {
					'lcobj_name':lcobj_name,
					'band_names':band_names,
					'c':class_names[lcobj.y],
					'error_bdict':fit_errors_bdict,
					}
				save_pickle(save_filedir, to_save, verbose=0) # save
				plot_kwargs = {
					'fit_errors_bdict':fit_errors_bdict,
					'save_rootdir':save_rootdir,
				}
				plot_synthetic_samples(lcdataset, set_name, method, lcobj_name, new_lcobjs, new_lcobjs_pm, **plot_kwargs)
				for knl,new_lcobj in enumerate(new_lcobjs):
					new_lcobj_name = f'{lcobj_name}.{knl+1}'
					new_lcobj.reset_day_offset_serial()
					synth_lcset.set_lcobj(new_lcobj_name, new_lcobj)

				if add_original:
					synth_lcset.set_lcobj(f'{lcobj_name}.0', lcobj.copy())

		except KeyboardInterrupt:
			can_be_in_loop = False

	bar.done()
	return fit_errors_bdict_list
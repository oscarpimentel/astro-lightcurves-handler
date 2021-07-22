#!/usr/bin/env python3
# -*- coding: utf-8 -*
import sys
sys.path.append('../') # or just install the module
sys.path.append('../../fuzzy-tools') # or just install the module
sys.path.append('../../astro-lightcurves-handler') # or just install the module

###################################################################################################################################################
import argparse
from fuzzytools.prints import print_big_bar

parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--method',  type=str, default='.', help='method')
main_args = parser.parse_args()
print_big_bar()

###################################################################################################################################################
import numpy as np
import fuzzytools.lists as lists
from fuzzytools.files import load_pickle, save_pickle, get_dict_from_filedir
from fuzzytools.matplotlib.utils import save_fig
import matplotlib.pyplot as plt
from lchandler.plots.lc import plot_lightcurve
from fuzzytools.files import save_time_stamp

methods = ['linear-fstw', 'bspline-fstw', 'spm-mle-fstw', 'spm-mle-estw', 'spm-mcmc-fstw', 'spm-mcmc-estw'] if main_args.method=='.' else main_args.method
methods = [methods] if isinstance(methods, str) else methods

for method in methods:
	filedir = f'../../surveys-save/survey=alerceZTFv7.1~bands=gr~mode=onlySNe~method={method}.splcds'
	# filedir = f'../../surveys-save/survey=alerceZTFv7.1~bands=gr~mode=onlySNe.splcds'
	filedict = get_dict_from_filedir(filedir)
	rootdir = filedict['_rootdir']
	cfilename = filedict['_cfilename']
	lcdataset = load_pickle(filedir)
	lcset_info = lcdataset['raw'].get_info()
	print(lcdataset)
	
	lcset_names = lcdataset.get_lcset_names()
	for lcset_name in lcset_names:
		lcset = lcdataset[lcset_name]
		for lcobj_name in lcset.get_lcobj_names():
			print(f'method={method}; lcset_name={lcset_name}; lcobj_name={lcobj_name}')
			figsize = (12,5)
			fig, ax = plt.subplots(1,1, figsize=figsize)
			lcobj = lcset[lcobj_name]
			c = lcset.class_names[lcobj.y]
			for kb,b in enumerate(lcset.band_names):
				plot_lightcurve(ax, lcobj, b, label=f'{b} obs')
			title = f'set={lcset.survey} [{lcset_name}]; obj={lcobj_name} [{lcset.class_names[lcobj.y]}]'
			ax.set_title(title)
			ax.set_xlabel('time [days]')
			ax.set_ylabel('observation [flux]')
			ax.grid(alpha=0.5)
			ax.legend()
			save_filedir = f'../save/{cfilename}/{lcset_name}/{c}/{lcobj_name}.png'
			save_fig(save_filedir, fig)
			save_time_stamp(f'../save/{cfilename}')
			#break
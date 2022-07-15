from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
import random
import copy
import fuzzytools.datascience.statistics as fstats
import fuzzytools.datascience.splits as splits
from fuzzytools.datascience.xerror import XError
from fuzzytools.strings import get_bar
from fuzzytools.level_bars import LevelBar
from fuzzytools.lists import get_bootstrap
from .lc_classes import diff_vector
import pandas as pd
from copy import copy
from fuzzytools.boostraping import BalancedCyclicBoostraping

MIN_POINTS_LIGHTCURVE_DEFINITION = _C.MIN_POINTS_LIGHTCURVE_DEFINITION
SET_NAME_STR = _C.SET_NAME_STR
SET_NAME_STR = _C.SET_NAME_STR
DAYS_INDEX = _C.DAYS_INDEX
OBS_INDEX = _C.OBS_INDEX
OBSE_INDEX = _C.OBSE_INDEX
RESET_TIME_OFFSET = True
SHUFFLE = True
RANDOM_STATE = None

###################################################################################################################################################

def get_sigma_clipping_indexing(x, dist_mean, dist_sigma, sigma_m:float,
    apply_lower_bound:bool=True,
    ):
    valid_indexs = np.ones((len(x),), dtype=bool)
    valid_indexs &= x < dist_mean+dist_sigma*sigma_m # is valid if is in range
    if apply_lower_bound:
        valid_indexs &= x > dist_mean-dist_sigma*sigma_m # is valid if is in range
    return valid_indexs

def search_over_sigma_samples(lcset, b:str, dist_mean, dist_sigma, sigma_m,
    apply_lower_bound:bool=True,
    ):
    total_deleted_points = 0
    for lcobj_name in lcset.get_lcobj_names():
        sigmas = lcset[lcobj_name].get_b(b).obse
        valid_indexs = get_sigma_clipping_indexing(sigmas, dist_mean, dist_sigma, sigma_m, apply_lower_bound)
        deleted_points = np.sum(~valid_indexs)
        total_deleted_points += deleted_points
        lcset.data[lcobj_name].get_b(b).apply_valid_indexs_to_attrs(valid_indexs)

    return total_deleted_points

###################################################################################################################################################

class LCDataset():
    def __init__(self,
        lcsets={},
        ):
        self.lcsets = lcsets
        self.reset()

    def reset(self):
        self.kfolds = []

    def set_lcset(self, lcset_name:str, lcset):
        self.lcsets[lcset_name] = lcset
        return lcset

    def get_lcset_names(self):
        return list(self.lcsets.keys())

    def exists(self, lcset_name):
        return lcset_name in self.get_lcset_names()

    def del_lcset(self, lcset_name):
        self.lcsets.pop(lcset_name, None) # pop and lost reference

    def only_keep_kf(self, kf):
        for lcset_name in self.get_lcset_names():
            if '@' in lcset_name and lcset_name.split('@')[0]==kf:
                pass
            else:
                self.del_lcset(lcset_name)
        return self

    def __getitem__(self, lcset_name):
        return self.lcsets[lcset_name]

    def __repr__(self):
        txt = 'LCDataset:\n'
        for lcset_name in self.get_lcset_names():
            txt += f'[{lcset_name}; samples {len(self[lcset_name]):,}]\n{self[lcset_name]}\n'
            txt += get_bar()+'\n'
        return txt

    def clean_empty_obs_keys(self):
        '''
        Along all lcsets
        Use this to delete empty light curves=0 obs in all bands
        '''
        for lcset_name in self.get_lcset_names():
            deleted_keys = self[lcset_name].clean_empty_obs_keys(verbose=0)
            print(f'({lcset_name}) deleted keys={deleted_keys}')

    def split(self, to_split_lcset_name,
        shuffle=SHUFFLE,
        random_state=RANDOM_STATE,
        outlier_obj_names=[],
        ):
        to_split_lcset = self[to_split_lcset_name]
        class_names = to_split_lcset.class_names
        obj_names = to_split_lcset.get_lcobj_names()
        obj_classes = [class_names[to_split_lcset[obj_name].y] for obj_name in obj_names]
        obj_names_kdict, class_names, kfolds, populations_cdict = splits.stratifiedf_kfold_cyclic_311(obj_names, obj_classes,
            shuffle=shuffle,
            random_state=random_state,
            outlier_obj_names=outlier_obj_names,
            prefix_str=f'{to_split_lcset_name}_',
            returns_populations_cdict=True,
            )
        self.kfolds = kfolds
        for new_set_name in obj_names_kdict.keys():
            self.set_lcset(new_set_name, to_split_lcset.copy({}))
            obj_names = obj_names_kdict[new_set_name]
            for obj_name in obj_names:
                lcobj = to_split_lcset[obj_name].copy()
                self[new_set_name].data.update({obj_name:lcobj})
        return populations_cdict

    def sigma_clipping(self, lcset_name, new_lcset_name,
        sigma_n:int=1,
        sigma_m:float=3.,
        apply_lower_bound:bool=True,
        verbose:int=1,
        remove_old_lcset=True,
        ):
        lcset = self.set_lcset(new_lcset_name, self[lcset_name].copy())
        #print(f'survey={lcset.survey}; after processing={lcset_name} (>{new_lcset_name})')
        total_deleted_points = {b:0 for b in lcset.band_names}
        for k in range(0, sigma_n):
            #print(f'k={k}')
            for b in lcset.band_names:
                sigma_values = lcset.get_all_values_b(b, 'obse')
                sigma_samples = len(sigma_values)
                mean = np.mean(sigma_values)
                sigma = np.std(sigma_values)
                deleted_points = search_over_sigma_samples(lcset, b, mean, sigma, sigma_m, apply_lower_bound)
                #print(f'\tband={b}; sigma_samples={sigma_samples:,}; mean={mean}; std={sigma}')
                #print(f'\tdeleted_points={deleted_points:,}')
                total_deleted_points[b] += deleted_points
        
            lcset.clean_empty_obs_keys()
            lcset.reset_all_day_offset_serial() # remove day offset!
            sigma_samples = len(lcset.get_all_values_b(b, 'obse'))
            #print(f'sigma_samples={sigma_samples:,}; total_deleted_points={total_deleted_points}')

        if remove_old_lcset:
            self.del_lcset(lcset_name)
        return total_deleted_points

    def get_serial_stats_idf(self,
        lcset_names=None,
        ):
        dfs = []
        lcset_names = self.get_lcset_names() if lcset_names is None else lcset_names
        for lcset_name in lcset_names:
            df = self[lcset_name].get_serial_stats_idf(lcset_name)
            dfs.append(df)
        return pd.concat(dfs)

    def get_bstats_idf(self, b,
        lcset_names=None,
        ):
        dfs = []
        lcset_names = self.get_lcset_names() if lcset_names is None else lcset_names
        for lcset_name in lcset_names:
            df = self[lcset_name].get_bstats_idf(b, lcset_name)
            dfs.append(df)
        return pd.concat(dfs)

    def __copy__(self):
        return self.copy()

    def copy(self):
        lcsets = {k:copy(self.lcsets[k]) for k in self.lcsets.keys()}
        return LCDataset(lcsets)

###################################################################################################################################################

class LCSet():
    def __init__(self,
        data:dict,
        survey:str,
        description:str,
        band_names:list,
        class_names:list,
        obs_is_flux:bool,
        ):
        self.data = data
        self.survey = survey
        self.description = description
        self.band_names = band_names.copy()
        self.class_names = class_names.copy()
        self.obs_is_flux = obs_is_flux
        self.reset()

    def reset(self):
        pass

    def generate_boostrap(self,
        batch_prop=1,
        ):
        lcobj_names = self.get_lcobj_names()
        lcobj_classes = [self.class_names[self[lcobj_name].y] for lcobj_name in lcobj_names]
        self.boostrap = BalancedCyclicBoostraping(lcobj_names, lcobj_classes,
            batch_prop=batch_prop,
            )

    def get_boostrap_samples(self):
        boostrap_samples = self.boostrap.get_samples()
        return boostrap_samples

    def __getitem__(self, lcobj_name):
        return self.data[lcobj_name]

    def get_info(self):
        info = {
            'survey':self.survey,
            'description':self.description,
            'band_names':self.band_names,
            'class_names':self.class_names,
            'obs_is_flux':self.obs_is_flux,
        }
        return info

    def get_lcobj_names(self,
        c=None,
        ):
        if c is None:
            return list(self.data.keys())
        else:
            return [k for k in self.data.keys() if self.class_names[self.data[k].y]==c]

    def get_lcobjs(self,
        c=None,
        ):
        return [self[lcobj_name] for lcobj_name in self.get_lcobj_names(c)]

    def clean_empty_obs_keys(self,
        length_to_keep=MIN_POINTS_LIGHTCURVE_DEFINITION,
        verbose:int=0,
        ):
        lcobj_names = self.get_lcobj_names()
        to_delete_lcobj_names = [lcobj_name for lcobj_name in lcobj_names if not any([len(self[lcobj_name].get_b(b))>=length_to_keep for b in self.band_names])]
        deleted_lcobjs = len(to_delete_lcobj_names)

        if verbose:
            print(f'deleted lcobjs={deleted_lcobjs}')

        for lcobj_name in to_delete_lcobj_names:
            self.data.pop(lcobj_name, None)

        return deleted_lcobjs

    def get_random_lcobj_name(self):
        lcobj_names = self.get_lcobj_names()
        return lcobj_names[random.randint(0, len(lcobj_names)-1)]

    def get_random_lcobj(self,
        return_key:bool=True,
        ):
        lcobj_name = self.get_random_lcobj_name()
        if return_key:
            return self[lcobj_name], lcobj_name
        return self[lcobj_name]

    def set_lcobj(self, lcobj_name, lcobj):
        self.data[lcobj_name] = lcobj

    def set_diff_parallel(self, attr:str):
        '''
        Along all keys
        '''
        for lcobj_name in self.get_lcobj_names():
            self[lcobj_name].set_diff_parallel(attr)

    def keys(self):
        return self.__dict__.keys()

    def get_lcobj_labels(self):
        return [lcobj.y for lcobj in self.get_lcobjs()]

    def get_lcobj_classes(self):
        '''
        Used for classes histogram
        '''
        lcobj_labels =self.get_lcobj_labels()
        return [self.class_names[y] for y in lcobj_labels]

    def get_nof_samples_cdict(self):
        return fstats.get_nof_samples_cdict(self.get_lcobj_classes(), self.class_names)

    def get_class_balanced_weights_cdict(self):
        pop_cdict = self.get_nof_samples_cdict()
        w = {c:1/(pop_cdict[c]*len(self.class_names)) for c in self.class_names} # 1/(N_c*C)
        return w

    def get_mean_length_df_bdict(self,
        index=None,
        ):
        df_bdict = {}
        for kb,b in enumerate(self.band_names):
            info_dict = {}
            for kc,c in enumerate(self.class_names):
                lcobjs = self.get_lcobjs(c)
                info_dict[f'{c}{b}-$N_c$'] = sum([len(lcobj.get_b(b)) for lcobj in lcobjs])/len(lcobjs)
            df = pd.DataFrame.from_dict({id(self) if index is None else index:info_dict}, orient='index')
            df.index.rename(SET_NAME_STR, inplace=True)
            df_bdict[b] = df

        return df_bdict

    def get_class_stats_idf(self,
        index=None,
        ):
        info_dict = {}
        for kc,c in enumerate(self.class_names):
            lcobjs = self.get_lcobjs(c)
            info_dict[f'{c}-$N_c$'] = len(lcobjs)
        df = pd.DataFrame.from_dict({id(self) if index is None else index:info_dict}, orient='index')
        df.index.rename(SET_NAME_STR, inplace=True)
        return df, self.get_mean_length_df_bdict()

    def get_serial_stats_idf_c(self, c):
        lcobjs = self.get_lcobjs(c)
        if len(lcobjs)>0:
            xs = [lcobj.get_x_serial() for lcobj in lcobjs]
            info_dict = {
                f'{c}-$x$':XError(np.concatenate([x[:,OBS_INDEX] for x in xs])),
                f'{c}-$L$':XError([len(lcobj) for lcobj in lcobjs]),
                f'{c}-$\Delta T$':XError([lcobj.get_serial_days_duration() for lcobj in lcobjs]),
                f'{c}-$\Delta t$':XError(np.concatenate([diff_vector(x[:,DAYS_INDEX]) for x in xs])),
            }
        else:
            info_dict = {
                f'{c}-$x$':XError([]),
                f'{c}-$L$':XError([]),
                f'{c}-$\Delta T$':XError([]),
                f'{c}-$\Delta t$':XError([]),
            }
        return info_dict

    def get_serial_stats_idf(self,
        index=None,
        ):
        info_dict = {}
        for kc,c in enumerate(self.class_names):
            info_dict.update(self.get_serial_stats_idf_c(c))

        info_dict = {id(self) if index is None else index:info_dict}
        df = pd.DataFrame.from_dict(info_dict, orient='index').reindex(list(info_dict.keys()))
        df.index.rename(SET_NAME_STR, inplace=True)
        return df

    def get_bstats_idf_c(self, c, b,
        index=None,
        ):
        lcobjs = self.get_lcobjs(c)
        parallel_diff_days = []
        for lcobj in lcobjs:
            parallel_diff_days += [lcobj.get_parallel_diff_days()[b]]

        if len(lcobjs)>0:
            info_dict = {
                f'{c}-$x$':XError(np.concatenate([lcobj.get_b(b).obs for lcobj in lcobjs])),
                f'{c}-$L$':XError([len(lcobj.get_b(b)) for lcobj in lcobjs]),
                f'{c}-$\Delta T$':XError([lcobj.get_b(b).get_days_duration() for lcobj in lcobjs if len(lcobj.get_b(b))>=1]),
                f'{c}-$\Delta t$':XError(np.concatenate(parallel_diff_days, axis=0)),
                f'{c}-tmax':XError([lcobj.get_b(b).get_max_brightness_time() for lcobj in lcobjs if not np.isnan(lcobj.get_b(b).get_max_brightness_time())]),
            }
        else:
            info_dict = {
                f'{c}-$x$':XError([]),
                f'{c}-$L$':XError([]),
                f'{c}-$\Delta T$':XError([]),
                f'{c}-$\Delta t$':XError([]),
                f'{c}-tmax':XError([]),
            }
        
        info_dict = {id(self) if index is None else index:info_dict}
        df = pd.DataFrame.from_dict(info_dict, orient='index').reindex(list(info_dict.keys()))
        df.index.rename(SET_NAME_STR, inplace=True)
        return df

    def get_bstats_idf(self, b,
        index=None,
        ):
        dfs = []
        for kc,c in enumerate(self.class_names):
            df = self.get_bstats_idf_c(c, b, index)
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        return df

    def __repr__b(self, b):
        df = self.get_bstats_idf(b)
        lengths = sum([df[f'{c}-$L$'].values[0] for c in self.class_names])
        durations = sum([df[f'{c}-$\Delta T$'].values[0] for c in self.class_names])
        cadences = sum([df[f'{c}-$\Delta t$'].values[0] for c in self.class_names])
        tmaxs = sum([df[f'{c}-tmax'].values[0] for c in self.class_names])
        txt = f'({b}) obs_samples={lengths.sum():,}; min_len={lengths.min()}; tmax={tmaxs.p50}; max_dur={durations.max():.1f} [days]; dur(p50)={durations.p50:.1f} [days]; cadence(p50)={cadences.p50:.1f} [days]\n'
        return txt

    def __repr_serial(self):
        df = self.get_serial_stats_idf()
        lengths = sum([df[f'{c}-$L$'].values[0] for c in self.class_names])
        durations = sum([df[f'{c}-$\Delta T$'].values[0] for c in self.class_names])
        cadences = sum([df[f'{c}-$\Delta t$'].values[0] for c in self.class_names])
        txt = f'(.) obs_samples={lengths.sum():,}; min_len={lengths.min()}; max_dur={durations.max():.1f}[days]; dur(p50)={durations.p50:.1f}[days]; cadence(p50)={cadences.p50:.1f}[days]\n'
        return txt

    def __repr__(self):
        if len(self)>0:
            txt = self.__repr_serial()
            for b in self.band_names:
                txt += self.__repr__b(b)
            nof_samples_cdict = self.get_nof_samples_cdict()
            txt += str(LevelBar(nof_samples_cdict, ' '*3))
        else:
            txt = 'empty lcset\n'
        return txt[:-1]

    def __len__(self):
        return len(self.get_lcobj_names())

    def get_lcobj_obs_classes_b_cdict(self, b:str):
        '''
        Used for obs histogram
        '''
        classes = [[self.class_names[self.data[k].y]]*len(self.data[k].get_b(b)) for k in self.data.keys()]
        classes = sum(classes, []) # flat lists
        return classes

    def get_lcobj_obsmean_b_cdict(self, b:str):
        population_dict = self.get_populations_cdict()
        uniques, counts = np.unique(self.get_lcobj_obs_classes_b_cdict(b), return_counts=True)
        return {c:counts[list(uniques).index(c)]/population_dict[c] for c in self.class_names}

    def get_max_length_serial(self):
        return max([len(self.data[k]) for k in self.data.keys()])

    def __copy__(self):
        return self.copy()

    def copy(self,
        data:dict=None,
        ):
        new_data = {k:copy(self.data[k]) for k in self.data.keys()} if data is None else data
        new_set = LCSet(
            new_data,
            self.survey,
            self.description,
            self.band_names,
            self.class_names,
            self.obs_is_flux,
            )
        return new_set

    def get_min_population(self):
        pop_cdict = self.get_populations_cdict()
        min_index = np.argmin([pop_cdict[c] for c in self.class_names])
        min_populated_class = self.class_names[min_index]
        min_population = pop_cdict[min_populated_class]
        return min_populated_class, min_population

    def get_random_stratified_lcobj_names(self, nc):
        lcobj_names = self.get_lcobj_names()
        return fstats.get_random_stratified_keys(lcobj_names, self.get_lcobj_classes(), self.class_names, nc)

    #########################

    def get_class_lcobjs(self, target_class):
        lcobjs = [lcobj for lcobj in self.get_lcobjs() if (target_class is None or target_class==self.class_names[lcobj.y])]
        return lcobjs

    def get_all_values_b(self, b:str, attr:str,
        target_class:str=None,
        ):
        lcobjs = self.get_class_lcobjs(target_class)
        values = [getattr(lcobj.get_b(b), attr) for lcobj in lcobjs]
        values = np.concatenate(values, axis=0)
        return values

    def get_all_values(self, attr:str,
        target_class=None,
        ):
        '''
        Get values of attr along all bands
        '''
        values = [self.get_all_values_b(b, attr, target_class=target_class) for b in self.band_names]
        return np.concatenate(values, axis=0)

    def get_all_min_value_b(self, b:str, attr,
        target_class=None,
        ):
        values = self.get_all_values_b(b, attr, target_class=target_class)
        return min(values)

    def get_all_max_value_b(self, b:str, attr,
        target_class=None,
        ):
        values = self.get_all_values_b(b, attr, target_class=target_class)
        return max(values)

    def get_all_parallel_diff_days_b(self, b,
        target_class=None,
        generates_mb=True,
        ):
        lcobjs = self.get_class_lcobjs(target_class)
        parallel_diff_days = []
        for lcobj in lcobjs:
            parallel_diff_days += [lcobj.get_parallel_diff_days(generates_mb=generates_mb)[b]]
        return np.concatenate(parallel_diff_days, axis=0)

    def get_all_parallel_diff_days(self,
        target_class=None,
        generates_mb=True,
        ):
        values = [self.get_all_parallel_diff_days_b(b, target_class=target_class, generates_mb=generates_mb) for b in self.band_names]
        return np.concatenate(values, axis=0)

    def reset_all_day_offset_serial(self,
        target_class=None,
        ):
        for lcobj in self.get_class_lcobjs(target_class):
            lcobj.reset_day_offset_serial() # remove day offset!

    def generate_all_mb(self,
        target_class=None,
        ):
        for lcobj in self.get_class_lcobjs(target_class):
            lcobj.generate_mb()

    def __add__(self, other):
        new = self.copy()
        lcobj_names = other.get_lcobj_names()
        for lcobj_name in lcobj_names:
            new.set_lcobj(lcobj_name, other[lcobj_name].copy())
        return new
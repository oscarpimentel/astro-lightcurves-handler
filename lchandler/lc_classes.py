from __future__ import print_function
from __future__ import division

import random
from copy import copy

import numpy as np
from fuzzytools import numba as ftnumba

from . import flux_magnitude as flux_magnitude
from . import _C


OBS_NOISE_RANGE = 1
CHECK = _C.CHECK
MIN_POINTS_LIGHTCURVE_DEFINITION = _C.MIN_POINTS_LIGHTCURVE_DEFINITION
CADENCE_THRESHOLD = _C.CADENCE_THRESHOLD
EPS = _C.EPS
RESET_TIME_OFFSET = True
SERIAL_CHAR = _C.SERIAL_CHAR

BYPASS_PROB_WINDOW = 0
BYPASS_PROB_DROPOUT = 0
BYPASS_PROB_OBS = 0
DS_MODE = {'random': 1.}
MIN_WINDOW_LENGTH_FRAC = 75 / 100
DF = 2  # 1 2 5 np.inf
OBSE_STD_SCALE = 1 / 2
DS_PROB = 10 / 100
BYPASS_PROB = .0


def diff_vector(x,
                uses_prepend=True,
                prepended_value=None,
                ):
    if len(x) == 0:
        return x
    if uses_prepend:
        x0 = x[0] if prepended_value is None else prepended_value
        new_x = np.concatenate([x0[None], x], axis=0)
    else:
        new_x = x
    dx = new_x[1:] - new_x[:-1]
    return dx


def get_new_noisy_obs(obs, obse, obs_min_lim,
                      std_scale=OBSE_STD_SCALE,
                      df=DF,
                      obs_noise_range=OBS_NOISE_RANGE,
                      ):
    """
    the clipping operation could be replaced with a truncating operation
    """
    assert df >= 0
    dtype = obs.dtype
    std = obse * std_scale
    if df == np.inf:
        new_obs = obs + std * np.random.standard_normal(size=len(obs)).astype(dtype)
    else:
        new_obs = obs + std * np.random.standard_t(df, size=len(obs)).astype(dtype)

    # bounds to avoid too disperse observations
    bar_size = (1.645 * 2) * obse  # for .95 percentile used in plot
    min_lim = obs - bar_size * obs_noise_range / 2
    max_lim = obs + bar_size * obs_noise_range / 2
    new_obs = np.clip(new_obs, min_lim, max_lim)
    new_obs = np.clip(new_obs, obs_min_lim, None)
    return new_obs


class SubLCO():
    """
    Dataclass object used to store an astronomical light curve
    """
    def __init__(self, days, obs, obse,
                 y=None,
                 dtype=np.float32,
                 flux_type=True,
                 ):
        self.days = days
        self.obs = obs
        self.obse = obse
        self.y = y
        self.dtype = dtype
        self.flux_type = flux_type
        self.reset()

    def reset(self):
        self.set_values(self.days, self.obs, self.obse)
        self.set_synthetic_mode(None)

    def convert_to_magnitude(self):
        if self.flux_type:
            flux = copy(self.obs)
            flux_error = copy(self.obse)
            self._set_obs(flux_magnitude.get_magnitude_from_flux(flux))
            self._set_obse(flux_magnitude.get_magnitude_error_from_flux(flux, flux_error))
            self.flux_type = False
        else:
            pass

    def convert_to_flux(self):
        if self.flux_type:
            pass
        else:
            mag = copy(self.obs)
            mag_error = copy(self.obse)
            self._set_obs(flux_magnitude.get_flux_from_magnitude(mag))
            self._set_obse(flux_magnitude.get_flux_error_from_magnitude(mag, mag_error))
            self.flux_type = True

    def get_synthetic_mode(self):
        return self.synthetic_mode

    def set_synthetic_mode(self, synthetic_mode):
        self.synthetic_mode = synthetic_mode

    def is_synthetic(self):
        return self.synthetic_mode is not None

    def set_values(self, days, obs, obse):
        """
        Always use this method to set new values!
        """
        assert len(days) == len(obs)
        assert len(days) == len(obse)
        tdays = copy(days).astype(self.dtype) if isinstance(days, np.ndarray) else np.array(days, dtype=self.dtype)
        tobs = copy(obs).astype(self.dtype) if isinstance(obs, np.ndarray) else np.array(obs, dtype=self.dtype)
        tobse = copy(obse).astype(self.dtype) if isinstance(obse, np.ndarray) else np.array(obse, dtype=self.dtype)
        self._set_days(tdays)
        self._set_obs(tobs)
        self._set_obse(tobse)

    def _set_days(self, days):
        assert len(days.shape) == 1
        if CHECK:
            assert np.all((diff_vector(days, uses_prepend=False) > 0))  # check if obs-days are in order
        self.days = days

    def _set_obs(self, obs):
        assert len(obs.shape) == 1
        if CHECK:
            assert np.all(obs >= 0)  # check all obs-levels are positive
        self.obs = obs

    def _set_obse(self, obse):
        assert len(obse.shape) == 1
        if CHECK:
            assert np.all(obse >= 0)  # check all obs-errors are positive
        self.obse = obse

    def add_day_values(self, values):
        """
        This method overrides information!
        Always use this method to add values
        calcule d_days again
        """
        assert len(self) == len(values)
        new_days = self.days + values
        self.days = new_days  # bypass _set_days() because non-sorted asumption
        valid_indexs = np.argsort(new_days)  # must sort before the values to mantain sequenciality
        self.apply_valid_indexs_to_attrs(valid_indexs)  # apply valid indexs to all

    def add_obs_values(self, values):
        """
        This method overrides information!
        Always use this method to add values
        calcule d_obs again
        """
        assert len(self) == len(values)
        new_obs = self.obs + values
        self._set_obs(new_obs)

    def apply_data_augmentation(self,
                                ds_mode=DS_MODE,
                                ds_prob=DS_PROB,
                                obs_min_lim=0,
                                min_valid_length=MIN_POINTS_LIGHTCURVE_DEFINITION,
                                min_window_length_frac=MIN_WINDOW_LENGTH_FRAC,
                                bypass_prob_window=BYPASS_PROB_WINDOW,
                                bypass_prob_dropout=BYPASS_PROB_DROPOUT,
                                std_scale=OBSE_STD_SCALE,
                                df=DF,
                                obs_noise_range=OBS_NOISE_RANGE,
                                bypass_prob_obs=BYPASS_PROB_OBS,
                                bypass_prob=BYPASS_PROB,
                                ):
        if random.random() > bypass_prob:
            self.apply_downsampling_window(
                                           ds_mode=ds_mode,
                                           ds_prob=ds_prob,
                                           min_valid_length=min_valid_length,
                                           min_window_length_frac=min_window_length_frac,
                                           bypass_prob_window=bypass_prob_window,
                                           bypass_prob_dropout=bypass_prob_dropout,
                                           )
            self.add_obs_noise_gaussian(obs_min_lim,
                                        std_scale=std_scale,
                                        df=df,
                                        obs_noise_range=obs_noise_range,
                                        bypass_prob_obs=bypass_prob_obs,
                                        )
        return

    def apply_downsampling_window(self,
                                  ds_mode=DS_MODE,
                                  ds_prob=DS_PROB,
                                  min_valid_length=MIN_POINTS_LIGHTCURVE_DEFINITION,
                                  min_window_length_frac=MIN_WINDOW_LENGTH_FRAC,
                                  bypass_prob_window=BYPASS_PROB_WINDOW,
                                  bypass_prob_dropout=BYPASS_PROB_DROPOUT,
                                  ):
        if len(self) <= min_valid_length:
            return
        valid_mask = np.ones((len(self)), dtype=np.bool)

        # mask
        if random.random() > bypass_prob_window:
            if ds_mode is None or len(ds_mode) == 0:
                ds_mode = {'none': 1}

            keys = list(ds_mode.keys())
            mode = np.random.choice(keys, p=[ds_mode[k] for k in keys])
            window_length = max(min_valid_length, int(min_window_length_frac * len(self)))
            if mode == 'none':
                pass

            elif mode == 'left':
                valid_mask[:] = False
                new_length = random.randint(window_length, len(self))  # [a,b]
                valid_mask[:new_length] = True

            elif mode == 'random':
                valid_mask[:] = False
                new_length = random.randint(window_length, len(self))  # [a,b]
                index = random.randint(0, len(self) - new_length)  # [a,b]
                valid_mask[index: index + new_length] = True
            else:
                raise Exception(f'no mode {mode}')

        # random dropout
        if random.random() > bypass_prob_dropout:
            assert ds_prob >= 0 and ds_prob <= 1
            if ds_prob > 0:
                ber_valid_mask = ftnumba.bernoulli(1 - ds_prob, len(self))
                valid_mask = valid_mask & ber_valid_mask

            if valid_mask.sum() < min_valid_length:  # extra case. If by change the mask implies a very short curve
                valid_mask = np.zeros((len(self)), dtype=np.bool)
                valid_mask[:min_valid_length] = True
                valid_mask = valid_mask[np.random.permutation(len(valid_mask))]

        # calcule again as the original values changed
        self.apply_valid_indexs_to_attrs(valid_mask)
        return

    def add_obs_noise_gaussian(self, obs_min_lim: float,
                               std_scale=OBSE_STD_SCALE,
                               df=DF,
                               obs_noise_range=OBS_NOISE_RANGE,
                               bypass_prob_obs=BYPASS_PROB_OBS,
                               ):
        """
        This method overrides information!
        """
        if std_scale == 0:
            return
        if random.random() > bypass_prob_obs:
            obs_values = get_new_noisy_obs(self.obs, self.obse, obs_min_lim,
                                           std_scale,
                                           df,
                                           obs_noise_range,
                                           )
            self.add_obs_values(obs_values - self.obs)
        return

    def apply_valid_indexs_to_attrs(self, valid_indexs):
        """
        Be careful, this method can remove info
        calcule d_days again
        calcule d_obs again
        fixme: this function is not opimized... specially due the d_days and that kind of variables
        """
        original_len = len(self)
        for key in self.__dict__.keys():
            x = self.__dict__[key]
            if isinstance(x, np.ndarray):  # apply same mask to all in the object
                assert len(x.shape) == 1  # 1D
                assert original_len == len(x), f'{key} {original_len}=={len(x)}'
                new_x = x[valid_indexs]
                setattr(self, key, new_x)

    def get_valid_indexs_max_day(self, max_day):
        return self.days <= max_day

    def clip_attrs_given_max_day(self, max_day):
        """
        Be careful, this method remove info!
        """
        valid_indexs = self.get_valid_indexs_max_day(max_day)
        self.apply_valid_indexs_to_attrs(valid_indexs)

    def get_valid_indexs_max_duration(self, max_duration):
        return self.days - self.get_first_day() <= max_duration

    def clip_attrs_given_max_duration(self, max_duration):
        """
        Be careful, this method remove info!
        """
        valid_indexs = self.get_valid_indexs_max_duration(max_duration)
        self.apply_valid_indexs_to_attrs(valid_indexs)

    def get_x(self):
        attrs = ['days', 'obs', 'obse']
        return self.get_custom_x(attrs)

    def get_attr(self, attr: str):
        return getattr(self, attr)

    def get_custom_x(self, attrs: list):
        values = [self.get_attr(attr)[..., None] for attr in attrs]
        x = np.concatenate(values, axis=-1)
        return x

    def get_first_day(self):
        return self.days[0]

    def get_last_day(self):
        return self.days[-1]

    def get_days_duration(self):
        if len(self) == 0:
            return None
        first_day = self.get_first_day()
        last_day = self.get_last_day()
        assert last_day >= first_day
        return last_day - first_day

    def copy(self):
        return copy(self)

    def __copy__(self):
        new_sublco = SubLCO(
                            copy(self.days),
                            copy(self.obs),
                            copy(self.obse),
                            self.y,
                            self.dtype,
                            )
        new_sublco.set_synthetic_mode(self.get_synthetic_mode())

        for key in self.__dict__.keys():
            if key in ['days', 'obs', 'obse']:
                continue
            v = self.__dict__[key]
            if isinstance(v, np.ndarray):
                setattr(new_sublco, key, copy(v))
        return new_sublco

    def __len__(self):
        nof_days = len(self.days)
        assert nof_days == len(self.obs)
        assert nof_days == len(self.obse)
        return nof_days

    def __repr__(self):
        txt = f'[d={self.days}{self.days.dtype}'
        txt += f'; o={self.obs}{self.obs.dtype}'
        txt += f'; oe={self.obse}{self.obse.dtype}]'
        return txt

    def clean_small_cadence(self,
                            dt=CADENCE_THRESHOLD,
                            mode='expectation',
                            verbose=0,
                            ):
        ddict = {}
        i = 0
        while i < len(self.days):
            day = self.days[i]
            valid_indexs = np.where((self.days >= day) & (self.days < day + dt))[0]
            ddict[day] = valid_indexs
            i += len(valid_indexs)

        new_days = []
        new_obs = []
        new_obse = []
        for k in ddict.keys():
            if verbose:
                _days = self.days[ddict[k]]
                if len(_days) > 1:
                    print(_days, max(_days) - min(_days))
            if mode == 'mean':
                new_days.append(np.mean(self.days[ddict[k]]))
                new_obs.append(np.mean(self.obs[ddict[k]]))
                new_obse.append(np.mean(self.obse[ddict[k]]))
            elif mode == 'min_obse':
                i = np.argmin(self.obse[ddict[k]])
                new_days.append(self.days[ddict[k]][i])
                new_obs.append(self.obs[ddict[k]][i])
                new_obse.append(self.obse[ddict[k]][i])
            elif mode == 'expectation':
                obse_exp = np.exp(-np.log(self.obse[ddict[k]] + EPS))
                assert len(np.where(obse_exp == np.inf)[0]) == 0
                dist = obse_exp / obse_exp.sum()
                new_days.append(np.sum(self.days[ddict[k]] * dist))
                new_obs.append(np.sum(self.obs[ddict[k]] * dist))
                new_obse.append(np.sum(self.obse[ddict[k]] * dist))
            else:
                raise Exception(f'mode={mode}')

        old_len = len(self)
        self.set_values(new_days, new_obs, new_obse)
        removed_obs = old_len - len(self)
        return removed_obs

    def get_snr(self,
                alpha=10,
                beta=1e-10,
                max_len=None,
                ):
        if len(self) == 0:
            return np.nan
        else:
            max_len = len(self) if max_len is None else max_len
            snr = (self.obs[:max_len]**2) / (alpha * self.obse[:max_len]**2 + beta)
            return np.mean(snr)

    def get_min_brightness(self,
                           return_idx=False,
                           ):
        idx = None,
        min_brightness = np.nan
        if len(self) > 0:
            if self.flux_type:
                idx = np.argmin(self.obs)
            else:
                idx = np.argmax(self.obs)
            min_brightness = self.obs[idx]

        if return_idx:
            return min_brightness, idx
        else:
            return min_brightness

    def get_max_brightness(self,
                           return_idx=False,
                           ):
        idx = None,
        max_brightness = np.nan
        if len(self) > 0:
            if self.flux_type:
                idx = np.argmax(self.obs)
            else:
                idx = np.argmin(self.obs)
            max_brightness = self.obs[idx]

        if return_idx:
            return max_brightness, idx
        else:
            return max_brightness

    def get_mean_brightness(self):
        if len(self) == 0:
            return np.nan
        else:
            return np.mean(self.obs)

    def get_max_brightness_time(self):
        if len(self) == 0:
            return np.nan
        else:
            _, idx = self.get_max_brightness(return_idx=True)
            tmax = self.days[idx]
            return tmax

    def __add__(self, other):
        if self is None or self == 0:
            return copy(other)

        if other is None or other == 0:
            return copy(self)

        if type(self) == SubLCO and type(other) == SubLCO:
            new_days = np.concatenate([self.days, other.days], axis=0)
            new_obs = np.concatenate([self.obs, other.obs], axis=0)
            new_obse = np.concatenate([self.obse, other.obse], axis=0)
            valid_indexs = np.argsort(new_days)
            new_lco = SubLCO(
                             new_days[valid_indexs],
                             new_obs[valid_indexs],
                             new_obse[valid_indexs],
                             self.y,
                             self.dtype,
                             )
            return new_lco

        assert 0

    def __radd__(self, other):
        return self + other

    def astype(self, dtype):
        self.dtype = dtype
        for key in self.__dict__.keys():
            x = self.__dict__[key]
            if isinstance(x, np.ndarray):  # apply same mask to all in the object
                new_x = x.astype(self.dtype)
                setattr(self, key, new_x)
        return self


class LCO():
    """
    Dataclass object used to store a multi-band astronomical light-curve
    """
    def __init__(self,
                 is_flux=True,
                 y=None,
                 ra=None,
                 dec=None,
                 z=None,
                 ):
        self.is_flux = is_flux
        self.set_y(y)
        self.ra = ra
        self.dec = dec
        self.z = z
        self.reset()

    def reset(self):
        self.bands = []

    def convert_to_magnitude(self):
        for b in self.bands:
            self.get_b(b).convert_to_magnitude()

    def convert_to_flux(self):
        for b in self.bands:
            self.get_b(b).convert_to_flux()

    def add_bands(self, band_dict,
                  reset_time_offset=RESET_TIME_OFFSET,
                  ):
        bands = band_dict.keys()
        for b in bands:
            args = band_dict[b]
            self.add_b(b, *args)
        if reset_time_offset:
            self.reset_day_offset_serial()

    def add_b(self, b: str, days, obs, obse):
        """
        Always use this method
        """
        sublcobj = SubLCO(days, obs, obse,
                          y=self.y,
                          )
        self.add_sublcobj_b(b, sublcobj)

    def add_sublcobj_b(self, b: str, sublcobj):
        assert not b == SERIAL_CHAR
        setattr(self, b, sublcobj)
        if b not in self.bands:
            self.bands += [b]

    def copy_only_metadata(self):
        new_lco = LCO(
            is_flux=self.is_flux,
            y=self.y,
            ra=self.ra,
            dec=self.dec,
            z=self.z,
        )
        return new_lco

    def copy(self):
        return copy(self)

    def __copy__(self):
        new_lco = LCO(
            is_flux=self.is_flux,
            y=self.y,
            ra=self.ra,
            dec=self.dec,
            z=self.z,
        )
        for b in self.bands:
            new_sublcobj = copy(self.get_b(b))
            new_lco.add_sublcobj_b(b, new_sublcobj)
        return new_lco

    def set_y(self, y: int):
        """
        Always use this method
        """
        self.y = None if y is None else int(y)

    def __repr__(self):
        txt = ''
        for b in self.bands:
            obj = self.get_b(b)
            txt += f'({b}:{len(obj)}) - {str(obj)}\n'
        return txt

    def __len__(self):
        return sum([len(self.get_b(b)) for b in self.bands])

    # serial/multi-band important methods
    def add_first_day(self, first_day):
        for b in self.bands:
            self.get_b(b).days = self.get_b(b).days + first_day

    def compute_global_first_day(self):
        first_days = [self.get_b(b).get_first_day() for b in self.get_bands() if len(self.get_b(b)) > 0]
        assert len(first_days) > 0
        global_first_day = min(first_days)
        return global_first_day

    def reset_day_offset_serial(self):
        """
        delete day offset acording to the first day along any day!
        """
        global_first_day = self.compute_global_first_day()
        self.add_first_day(-global_first_day)
        return self

    def get_sorted_days_indexs_serial(self):
        values = [self.get_b(b).days for b in self.get_bands()]
        all_days = np.concatenate(values, axis=0)
        sorted_days_indexs = np.argsort(all_days)
        return sorted_days_indexs

    def get_onehot_serial(self):
        onehot = np.zeros((len(self), len(self.get_bands())), dtype=np.bool)
        index = 0
        for kb, b in enumerate(self.get_bands()):
            length = len(getattr(self, b))
            onehot[index:index + length, kb] = True
            index += length
        sorted_days_indexs = self.get_sorted_days_indexs_serial()
        onehot = onehot[sorted_days_indexs]
        return onehot

    def get_x_serial(self,
                     attrs=['days', 'obs', 'obse'],
                     ):
        values = [self.get_b(b).get_custom_x(attrs) for b in self.get_bands()]
        x = np.concatenate(values, axis=0)
        sorted_days_indexs = self.get_sorted_days_indexs_serial()
        x = x[sorted_days_indexs]
        return x

    def get_serial_days(self):
        serial_days = self.get_x_serial(['days'])
        return serial_days

    def get_serial_diff_days(self):
        serial_days = self.get_serial_days()[:, 0]
        serial_diff_days = diff_vector(serial_days,
                                       uses_prepend=True,
                                       prepended_value=None,
                                       )
        return serial_diff_days

    def get_parallel_days(self):
        parallel_days = {}
        for b in self.get_bands():
            days = self.get_b(b).days
            parallel_days[b] = days
        return parallel_days

    def get_parallel_diff_days(self,
                               generates_mb=True,
                               ):
        global_first_day = self.compute_global_first_day()
        bands = copy(self.get_bands())
        if generates_mb:
            self.generate_mb()
        if hasattr(self, 'merged_band'):
            bands += [SERIAL_CHAR]
        parallel_diff_days = {}
        for b in bands:
            days = self.get_b(b).days
            diff_days = diff_vector(days,
                                    uses_prepend=True,
                                    prepended_value=global_first_day,
                                    )
            parallel_diff_days[b] = diff_days
        return parallel_diff_days

    def get_serial_days_duration(self):
        """
        Duration in days of complete light curve
        """
        serial_days = self.get_serial_days()
        duration = np.max(serial_days) - np.min(serial_days)
        return duration

    def get_b(self, b: str):
        if b == SERIAL_CHAR:
            return self.get_mb()
        else:
            return getattr(self, b)

    def generate_mb(self):
        self.merged_band = sum([self.get_b(b) for b in self.get_bands()])  # generate

    def get_mb(self):
        self.generate_mb()
        return self.merged_band

    def clip_attrs_given_max_day(self, max_day):
        for b in self.get_bands():
            self.get_b(b).clip_attrs_given_max_day(max_day)

    def get_bands(self):
        return self.bands

    def get_length_b(self, b: str):
        return len(self.get_b(b))

    def get_length_bdict(self):
        return {b: self.get_length_b(b) for b in self.get_bands()}

    def any_synthetic(self):
        return any([self.get_b(b).is_synthetic() for b in self.get_bands()])

    def all_synthetic(self):
        return all([self.get_b(b).is_synthetic() for b in self.get_bands()])

    def any_real(self):
        return any([not self.get_b(b).is_synthetic() for b in self.get_bands()])

    def all_real(self):
        return all([not self.get_b(b).is_synthetic() for b in self.get_bands()])

    def any_band_eqover_length(self,
                               th_length=MIN_POINTS_LIGHTCURVE_DEFINITION,
                               ):
        return any([len(self.get_b(b)) >= th_length for b in self.get_bands()])

    def clean_small_cadence(self,
                            dt=CADENCE_THRESHOLD,
                            mode='expectation',
                            ):
        removed_obs = 0
        for b in self.get_bands():
            removed_obs += self.get_b(b).clean_small_cadence(dt, mode)
        self.reset_day_offset_serial()
        return removed_obs

    def get_snr(self):
        snr_d = {b: self.get_b(b).get_snr() for b in self.get_bands()}
        return snr_d

    def get_tmax(self):
        tmax_d = {b: self.get_b(b).get_tmax() for b in self.bands}
        return tmax_d

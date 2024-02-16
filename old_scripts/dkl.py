"""Estimating information gained over initial uncertainty.

The U component (uncertainty) for future estimates can be replaced with the remaining uncertainty (DKL)
"""
import warnings
from collections.abc import Iterable

import numpy as np

# np.seterr(divide='ignore')
# warnings.filterwarnings('ignore')
warnings.simplefilter('error')

class DKL:
    def __init__(self, o, f):
        self.o = o
        self.f = f

        assert np.max(f) < 1.0
        assert np.min(f) > 0.0

        # Number of unique forecast probs
        self.k = np.unique(self.f)

        # Number of unique classes
        self.c = np.unique(self.o)

        # Frequency of obs
        self.o_bar = np.mean(self.o)

    @staticmethod
    def __compute_dkl(T, P, epsilon=1e-12):
        """
        Generalised Kullback-Leibler divergence computation, where T is "truth" and P is "prediction."
        This avoids assuming that 1 > f > 0.
        """

        # Ensure T and P are iterable
        T = np.atleast_1d(T)
        P = np.atleast_1d(P)

        # Add epsilon for numerical stability
        adjusted_T = np.clip(T, epsilon, 1 - epsilon)
        adjusted_P = np.clip(P, epsilon, 1 - epsilon)

        # Calculate DKL in a single line, taking advantage of NumPy's element-wise operations
        all_dkl = (1 - adjusted_T) * np.log2((1 - adjusted_T) / (1 - adjusted_P)) + adjusted_T * np.log2(
            adjusted_T / adjusted_P)

        return all_dkl, np.mean(all_dkl)

    def compute_dkl(self, from_components=False):
        if from_components:
            U = self.compute_unc()
            R = self.compute_rel()
            D = self.compute_dsc()
            return R - D + U
        all_dkl, raw_dkl = self.__compute_dkl(self.o,self.f)
        dkl = np.nan_to_num(raw_dkl)
        return dkl

    def compute_dsc(self):
        N = len(self.o)
        K = len(self.k)
        ok_bar_1d = np.zeros([K])
        dsc_1d = np.zeros([K])
        fk_list = []
        nk_1d = np.zeros([K])
        dkl_1d = np.zeros([K])
        for ik,k in enumerate(self.k):
            ok_bar_1d[ik] = np.mean(self.o[self.f==k])
            fk_list.append(self.f[self.f==k])
            nk_1d[ik] = len(fk_list[ik])
            dkl_all, dkl_mean = self.__compute_dkl(ok_bar_1d[ik],self.o_bar)
            dkl_1d[ik] = np.mean(dkl_all)
            dsc_1d[ik] = nk_1d[ik] * dkl_1d[ik]
        return np.sum(dsc_1d)/N

    def compute_unc(self):
        term1 = (1-self.o_bar) * np.log2(1-self.o_bar)
        term2 = self.o_bar * np.log2(self.o_bar)
        unc = term1+term2
        return -unc

    def compute_rel(self):
        N = len(self.o)
        K = len(self.k)
        rel_1d = np.zeros([K])
        ok_bar_1d = np.zeros([K])
        fk_list = []
        nk_1d = np.zeros([K])
        dkl_1d = np.zeros([K])
        for ik,k in enumerate(self.k):
            ok_bar_1d[ik] = np.mean(self.o[self.f==k])
            fk_list.append(self.f[self.f==k])
            nk_1d[ik] = len(fk_list[ik])
            dkl_all, dkl_mean = self.__compute_dkl(ok_bar_1d[ik],fk_list[ik])
            dkl_1d[ik] = dkl_mean
            rel_1d[ik] = (nk_1d[ik]*dkl_1d[ik])
        return np.sum(rel_1d)/N

    @classmethod
    def compute_info_gain(cls, o, f1, f2, from_components=False):
        DKL1 = cls.__compute_dkl(o, f1)[1]
        DKL2 = cls.__compute_dkl(o, f2)[1]
        return DKL1 - DKL2

    def compute_bs(self):
        return np.mean((self.o-self.f)**2)

    def compute_bss(self):
        bs = self.compute_bs()
        bs_unc = self.o_bar * (1-self.o_bar)
        return 1 - (bs/bs_unc)

    def compute_skill_score(self, return_components=True):
        U = self.compute_unc()
        R = self.compute_rel()
        D = self.compute_dsc()
        D_ss = D/U
        R_ss = R/U
        SS = D_ss - R_ss
        if return_components:
            return SS, D_ss, R_ss
        return SS

    def compute_info_gain_over_climo(self):
        R = self.compute_rel()
        D = self.compute_dsc()

        # Info gain over using the base-rate for forecasts
        return (D - R)
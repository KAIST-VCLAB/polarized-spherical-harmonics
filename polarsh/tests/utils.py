import numpy as np
from typing import Optional, Callable
from numpy.typing import ArrayLike

from polarsh.array import *
from polarsh.sphere import *

def rate_info(mask):
    assert mask.dtype == bool
    n_true = mask.sum()
    n_total = mask.size
    return f"{n_true}/{n_total} ({n_true/n_total*100:.2f}%)"

class ReportOption:
    """
    do_tolinfo: `Info`rmation of how much amount are inside the `tol`erance
    do_showex:  `Show` an `ex`ample of out-of-tolarance entry
    do_assert:  `Assert` all entries are in the tolerance
    """
    def __init__(self, do_tolinfo = True, do_showex = False, do_assert = False):
        self.do_tolinfo = do_tolinfo
        self.do_showex = do_showex
        self.do_assert = do_assert

class Report:
    fform = lambda x: "%.8f" % x # Float formatter
    @classmethod
    def set_fform(cls, a_fform):
        cls.fform = a_fform

    @classmethod
    def diff(cls, a, b):
        return a - b
    
    @classmethod
    def isclose_args(cls, a, b):
        return a, b
    
    @classmethod
    def at(cls, A: ArrayLike, idx: tuple):
        return A[idx]
    
    @classmethod
    def str_idx(cls, idx: tuple):
        return str(idx)[1:-1]

    @classmethod
    def print_example(cls, A, isclose, name_A="A", indent=""):
        I = np.where(~isclose)
        if I[0].size:
            idx = tuple([i.ravel()[0] for i in I])
            # print(f"[DEBUG] {isclose.shape = }, {len(I) = }, {I[0].shape = }, {idx = }")
            print(f"{indent}{name_A}[{cls.str_idx(idx)}] = {cls.at(A,idx)}")
    
    @classmethod
    def report_post(cls,
                    A:          ArrayLike, # [*]
                    B:          ArrayLike, # [*]
                    isclose:    ArrayLike, # [*] bool
                    indent:     Optional[str] = "",
                    option:     Optional[ReportOption] = ReportOption()
                   ) ->         np.ndarray[bool]: # [*]
        if not isclose.all():
            if option.do_tolinfo:
                print(f"{indent}Tolerance: {rate_info(isclose)}")
            if option.do_showex:
                cls.print_example(A, isclose, indent=f"{indent}e.g.: ")
                cls.print_example(B, isclose, indent=f"{indent}      ", name_A="B")
                cls.print_example(A-B, isclose, indent=f"{indent}      ", name_A="A-B")
            if option.do_assert:
                raise AssertionError()

    @classmethod
    def report(cls,
               A:          ArrayLike, # [*]
               B:          ArrayLike, # [*]
               rtol:       Optional[float] = 1e-05,
               atol:       Optional[float] = 1e-08,
               indent:     Optional[str] = "",
               option:     Optional[ReportOption] = ReportOption()
              ) ->         np.ndarray[bool]: # [*]
        A, B = np.broadcast_arrays(A, B)
        diff = np.abs(cls.diff(A, B))
        isclose = np.isclose(*cls.isclose_args(A, B), rtol=rtol, atol=atol)
        isnan_diff = np.isnan(diff)
        if isnan_diff.any():
            nan_info = f"[w/o {rate_info(isnan_diff)} NaNs]"
        else:
            nan_info = ""
        print(f"{indent}mean.abs: {cls.fform(np.nanmean(diff))},",
                      f"rms: {cls.fform(rms(diff, allow_nan=True))},",
                      f"max.abs: {cls.fform(np.nanmax(diff))}", nan_info)
        cls.report_post(A, B, isclose, indent=indent, option=option)
        return isclose
    
    @classmethod
    def report_ge(cls,
                  A:      ArrayLike, # [*]
                  B:      ArrayLike, # [*]
                  atol:   Optional[float] = 1e-08,
                  indent: Optional[str] = "",
                  option:     Optional[ReportOption] = ReportOption()
                  ) ->     np.ndarray[bool]: # [*]
        A, B = np.broadcast_arrays(A, B)
        count = A >= B - atol
        isnan = np.isnan(A)
        if isnan.any():
            nan_info = f"[w/o {rate_info(isnan)} NaNs]"
        else:
            nan_info = ""
        print(f"{indent}min: {cls.fform(np.nanmin(A))}", nan_info)
        cls.report_post(A, B, count, indent=indent, option=option)
        return count

    @classmethod
    def report_le(cls,
                  A:      ArrayLike,
                  B:      ArrayLike,
                  atol:   Optional[float] = 1e-08,
                  indent: Optional[str] = "",
                  option:     Optional[ReportOption] = ReportOption()
                  ) ->     np.ndarray[bool]: # [*]
        A, B = np.broadcast_arrays(A, B)
        count = A <= B + atol
        isnan = np.isnan(A)
        if isnan.any():
            nan_info = f"[w/o {rate_info(isnan)} NaNs]"
        else:
            nan_info = ""
        print(f"{indent}max: {cls.fform(np.nanmax(A))}", nan_info)
        cls.report_post(A, B, count, indent=indent, option=option)
        return count

def assert_error(func: Callable) -> Callable:
    """ Decorator which asserts given `func` raise an error. """
    def wrapper(*args, **kargs):
        try:
            res = func(*args, **kargs)
        except Exception as err:
            print(f"[True posivie] {type(err).__name__}: {err}")
        else:
            raise RuntimeError(f"False negative error!")
    return wrapper
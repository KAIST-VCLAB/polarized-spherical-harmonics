from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike
from polarsh.SH import CodType

def assert_coeff_s2s(coeff: ArrayLike, level: int, shape_chan: Tuple[int], cod_type: CodType) -> None:
    if cod_type == CodType.POLAR2:
        assert coeff is None or coeff.size == 0
        return None
    elif cod_type == CodType.POLAR4:
        assert coeff.shape == shape_chan + (level, 2, 2)
        assert coeff.dtype == np.float64
    else: # CodType.SCALAR or CodType.POLAR3
        assert coeff.shape == shape_chan + (level,)
        assert coeff.dtype == np.float64
    return coeff

def assert_coeff_sv(coeff: ArrayLike, level: int, shape_chan: Tuple[int], cod_type: CodType) -> None:
    '''
    Both for s2v and v2s
    '''
    if cod_type in [CodType.SCALAR, CodType.POLAR2]:
        assert coeff is None or coeff.size == 0
        return None
    elif cod_type == CodType.POLAR3:
        assert coeff.shape == shape_chan + (level,)
        assert coeff.dtype == np.complex128
    else: # CodType.POLAR4
        assert coeff.shape == shape_chan + (level, 2)
        assert coeff.dtype == np.complex128
    return coeff

def assert_coeff_v2v(coeffa: ArrayLike, coeffb: ArrayLike, level: int, shape_chan: Tuple[int], cod_type: CodType) -> None:
    if cod_type == CodType.SCALAR:
        assert coeffa is None or coeffa.size == 0
        assert coeffb is None or coeffb.size == 0
        return None, None
    else: # CodType.POLAR2 or CodType.POLAR3 or CodType.POLAR4
        assert coeffa.shape == shape_chan + (level,)
        assert coeffa.dtype == np.complex128
        assert coeffb.shape == shape_chan + (level,)
        assert coeffb.dtype == np.complex128
    return coeffa, coeffb

def get_U(mi: ArrayLike, mo: ArrayLike) -> Tuple[np.ndarray, np.ndarray]: # [*] for each
    '''
    Assume U is 3x3, U[0,0] positive, U[1,1] zero, and U[2,2] negative. Do not apply mask
    '''
    mi = np.asarray(mi)
    mo = np.asarray(mo)
    # i = np.where(mo > 0, 0, np.where(mo == 0, 1, 2))
    # j = np.where(mi > 0, 0, np.where(mi == 0, 1, 2))
    
    phase = (-1) ** (mi % 2)
    zero = np.zeros(mi.shape)
    one = np.ones(mi.shape)
    rt2 = one * np.sqrt(2)
    U_s2v = np.select([mo > 0, mo == 0], [np.select([mi > 0, mi == 0], [one, zero], -one*1j),
                                          np.select([mi > 0, mi == 0], [zero, rt2], zero)],
                                          np.select([mi > 0, mi == 0], [phase, zero], phase*1j))

    U_v2s = np.select([mo > 0, mo == 0], [np.select([mi > 0, mi == 0], [one, zero], phase),
                                          np.select([mi > 0, mi == 0], [zero, rt2], zero)],
                                          np.select([mi > 0, mi == 0], [one*1j, zero], -phase*1j)).conj()
    U_s2v /= np.sqrt(2)
    U_v2s /= np.sqrt(2)
    return U_s2v, U_v2s
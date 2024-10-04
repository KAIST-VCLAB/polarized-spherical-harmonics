'''
For testing, see "test_SH.py".
'''
from __future__ import annotations
from time import time
from enum import IntEnum
import numpy as np
from numpy.typing import ArrayLike
import scipy.special as sp
from scipy.spatial.transform import Rotation as scipyRotation
import quaternionic as quat
import spherical

from .array import *
from .sphere import *

class DomType(IntEnum):
    ''' Enumeration type for domain of spherical functions '''
    UNI    = 0 # f(omega)
    BI     = 1 # f(omega_i,omega_o)
    ISOBI  = 2 # f(omega_i,omega_o) with azimuthal symmetry
    ISOBI2 = 3 # DomType.ISOBI with (mi,mo) = (\pm m, \pm m) w/o same order of symbols

    @classmethod
    def _missing_(cls, value):
        value = value.upper()
        for member in cls:
            if str(member) == value:
                return member
        return None
    
    @property
    def label(self):
        if self == DomType.UNI:
            return "UNI"
        elif self == DomType.BI:
            return "BI"
        elif self == DomType.ISOBI:
            return "ISOBI"
        else:
            return "ISOBI2"
    
    def __str__(self):
        return self.label

class CodType(IntEnum):
    ''' Enumeration type for codomain of spherical functions '''
    SCALAR = 1 # scalar radiance, s0
    POLAR2 = 2 # linear polarization, [s1, s2]
    POLAR3 = 3 # up to linear polarization, [s0, s1, s2]
    POLAR4 = 4 # full stokes parameters, [s0, s1, s2, s3]

    @classmethod
    def _missing_(cls, value):
        value = value.upper()
        for member in cls:
            if str(member) == value:
                return member
        return None
    
    def __str__(self):
        if self == CodType.SCALAR:
            return "SCALAR"
        elif self == CodType.POLAR2:
            return "POLAR2"
        elif self == CodType.POLAR3:
            return "POLAR3"
        else:
            return "POLAR4"

class SHType(IntEnum):
    ''' Enumeration type for spherical harmonics coefficient type'''
    REAL = 0 # real spherical harmonics
    COMP = 1 # complex spherical harmonics

    @classmethod
    def _missing_(cls, value):
        value = value.upper()
        for member in cls:
            if str(member) == value:
                return member
        return None
    
    @property
    def str1(self):
        if self == SHType.REAL:
            return "R"
        else:
            return "C"
    
    @property
    def str4(self):
        if self == SHType.REAL:
            return "REAL"
        else:
            return "COMP"
    
    def __str__(self):
        return self.str4


'''
Following should be satisfied:
`set(cidx_scalar(cod_type)) + set(cidx_lpolar(cod_type)) == set(range(cod_type))`
NOTE Be careful that returning tuples rather than slice instances
     yield different result in some context.
     (tuples indexing makes copy of arrays, but slicing does not)
'''
def cidx_scalar(cod_type: CodType, out_type: type = slice) -> Tuple[int,...]:
    if not out_type in [slice, tuple]:
        raise ValueError(f"{out_type = } should be `slice` or `tuple`.")
    # if cod_type == CodType.SCALAR:
        # return None
    if cod_type == CodType.POLAR2:
        if out_type == slice:
            return np.s_[0:0]
        else:
            return ()
    elif cod_type >= CodType.POLAR3:
        if out_type == slice:
            return np.s_[0::3]
        else:
            return tuple(range(0,cod_type,3))
    else:
        msg = f"cod_type should be one of CodType.POLAR2-4, but {cod_type} is given."
        raise ValueError(msg)

def cidx_lpolar(cod_type: CodType) -> Tuple[int,...]:
    if cod_type == CodType.POLAR2:
        return np.s_[:2]
    elif cod_type in [CodType.POLAR3, CodType.POLAR4]:
        return np.s_[1:3]
    else:
        msg = f"cod_type should be one of CodType.POLAR2-4, but {cod_type} is given."
        raise ValueError(msg)

def cidx_to(cod_type_from: CodType, cod_type_to: CodType) -> Union[int, slice]:
    """
    Depending on the pair of arguments, it behaves as follows:
    Case 1. increasing `cod_type` or `POLAR2`->`SCALAR`
        * It raises an error.
    Case 2. `SCALAR`->`SCALAR`
        * It does not raise an error but return a meaningless value.
    Case 3. `POLAR*`->(`SCALAR`|`POLAR*`), non-increasing
        * It returns `ret: Union[int|slice]` such that `coeff_from[*, ret]` == `coeff_to`
    """
    try:
        cod_type_from = CodType(cod_type_from)
        cod_type_to = CodType(cod_type_to)
    except Exception as e:
        print(f"{cod_type_from = }, {cod_type_to = }")
        raise e
    assert cod_type_from >= cod_type_to, \
           f"The {cod_type_to=} argument cannot be larger than {cod_type_from=}."
    assert not (cod_type_from, cod_type_to) == (CodType.POLAR2, CodType.SCALAR), \
           f"CodType.POLAR2 cannot be cut into CodType.SCALAR."
    
    if cod_type_from == CodType.SCALAR:
        # NOTE This case should be treated manually at the outside where this function is called.
        return ""
    elif cod_type_from == CodType.POLAR2:
        return np.s_[:]
    else:
        match cod_type_to:
            case CodType.SCALAR:
                return 0
            case CodType.POLAR2:
                return np.s_[1:3]
            case CodType.POLAR3:
                return np.s_[:3]
            case _: # CodType.POLAR4
                return np.s_[:]
    
            
            
# def level2num(level: int, dom_type: DomType, codom_type: CodType
def level2num(level: int, dom_type: DomType
                 ) -> int: # =: N(L)
    # Number of SH coefficients for an isotropic bidirectional function upto l < level
    # |{(li,lo,m) | li,lo < level, -l <= m <= l, l=min(li,lo)}|
    dom_type = DomType(dom_type)
    if dom_type == DomType.UNI:
        return (level**2)
    elif dom_type == DomType.BI:
        return level**4
    elif dom_type == DomType.ISOBI:
        return level*(2*level*level + 1) // 3
    else:
        assert dom_type == DomType.ISOBI2, f"Invalid domain type: {dom_type}"
        return level*(4*level*level - 3*level + 2) // 3


def level2num_inv(N: ArrayLike, dom_type: DomType
                     ) -> float: # == N^-1(N), but N is allowed to be float
    ## w/o considering type of codomain (assume CodType.SCALAR)
    N = np.asarray(N)
    dom_type = DomType(dom_type)
    if dom_type == DomType.UNI:
        return np.sqrt(N)
    elif dom_type == DomType.BI:
        return np.power(N, 0.25)
    elif dom_type == DomType.ISOBI:
        ## Reference: Factorizing 3-parameter BSDF.md
        A = 27*N
        B = np.sqrt(A**2.0 + 6) + A
        ## Here, use "**2.0" rather than "**2" due to overflow.
        ## Note that np.int32 raises overflow earlier than python int.
        ## Test following:
        ## >>> np.int32(103640)**2
        ## >>> 103640**2
        C1 = np.power(B, 1/3)
        C2 = np.power(6/B, 1/3)
        return np.power(6, -2/3)*(C1 - C2)
    else:
        assert dom_type == DomType.ISOBI2, f"Invalid domain type: {dom_type}"
        A = 81*(8*N - 1)
        B = 15**3
        C = A*A + B
        D = A + np.sqrt(C)
        return (3 + D**(1/3) - (B/D)**(1/3)) / 12


def num2level_assert(numcoeff, dom_type) -> int:
    ## Find L s.t. numcoeff == N(L), and
    ## assert such L exists
    ## without considering type of codomain
    ## TODO: generalize for cod_type
    L = np.floor(level2num_inv(numcoeff-0.5, dom_type)).astype(np.int32)+1
    N_recon = level2num(L, dom_type)
    assert numcoeff == N_recon, f"{numcoeff}, {L}->{N_recon}"
    return L

def idx2lmax(idx:      ArrayLike, # *
             dom_type: DomType
            )       -> ArrayLike: # *
    # minimum lmax s.t. idx < N(lmax+1)
    # without considering type of codomain
    idx = np.asarray(idx)
    return np.floor(level2num_inv(idx+0.5, dom_type)).astype(np.int32)

def lms2idx(lms:      ArrayLike, # *xd, d == 2,      3,   4 
            dom_type: DomType #    for DomType.UNI, DomType.ISOBI, DomType.ISOBI2/DomType.BI, resp.
           )       -> ArrayLike: # *
    """
    's' in lm's'2idx indicate the -s suffix for pluralization in English. 
    """
    lms = np.asarray(lms)
    dom_type = DomType(dom_type)
    sh = lms.shape
    if dom_type == DomType.UNI:
        assert sh[-1] == 2, f"The shape of lms must be (...,2), but currently: {sh}"
        lms = lms.reshape(-1,2)
        l = lms[:,0]
        m = lms[:,1]
        idx = level2num(l, dom_type) + m + l
    elif dom_type == DomType.BI: # [li, lo, mi, mo]
        assert sh[-1] == 4, f"The shape of lms must be (...,4), but currently: {sh}"
        lms = lms.reshape(-1,4)
        li, lo, mi, mo = npunstack(lms, axis=-1)
        lmax = np.maximum(li, lo)
        lmin = np.minimum(li, lo)
        
        idx_base2level = level2num(lmax, dom_type)
        idx_level2lmin = 2 * (2*lmax+1)*(lmin**2)
        idx_lmin2curr = (mi+li)*(2*lo+1) + (mo+lo)
        idx_lmin2curr += np.where(li > lo, (2*lmax+1)*(2*lmin+1), 0)
        idx = idx_base2level + idx_level2lmin + idx_lmin2curr
    elif dom_type == DomType.ISOBI: # [li, lo, m]
        assert sh[-1] == 3, f"The shape of lms must be (...,3), but currently: {sh}"
        lms = lms.reshape(-1,3)
        li, lo, m = npunstack(lms, axis=-1)
        lmax = np.maximum(li, lo)
        lmin = np.minimum(li, lo)
        
        idx_base2level = level2num(lmax, dom_type)
        idx_level2lmin = 2*level2num(lmin, DomType.UNI)
        idx_lmin2curr = m + lmin
        idx_lmin2curr += np.where(li > lo, 2*lmin+1, 0)
        idx = idx_base2level + idx_level2lmin + idx_lmin2curr
    else:                    # [li, lo, mi, mo], |mi|==|mo|
        assert dom_type == DomType.ISOBI2, f"Invalid domain type: {dom_type}"
        assert sh[-1] == 4, f"The shape of lms must be (...,4), but currently: {sh}"
        lms = lms.reshape(-1,4)
        li, lo, mi, mo = npunstack(lms, axis=-1)
        lmax = np.maximum(li, lo)
        lmin = np.minimum(li, lo)
        
        idx_base2level = level2num(lmax, dom_type)
        idx_level2lmin = 2 * lmin*(2*lmin-1)
        idx_lmin2curr = 2*(lmin+mi) + np.select([mi>mo, mi<mo], [-1, 1], 0)
        idx_lmin2curr += np.where(li > lo, 4*lmin+1, 0)
        idx = idx_base2level + idx_level2lmin + idx_lmin2curr

    return idx.reshape(sh[:-1])

def idx2lms(idx:      ArrayLike, # *
            dom_type: DomType,
            unstack:  bool = False
           )       -> ArrayLike: # *xd, d = |l,m,...|
    """
    's' in lm's'2idx indicate the -s suffix for pluralization in English. 
    """
    idx = np.asarray(idx)
    dom_type = DomType(dom_type)
    lmax = idx2lmax(idx, dom_type)
    idx_wrt_level = idx - level2num(lmax, dom_type)

    if dom_type == DomType.UNI:
        lms = np.stack([lmax, idx_wrt_level-lmax], axis=-1)

    elif dom_type == DomType.BI:
        # Order: lmax -> lmin -> li -> mi -> mo
        def foo1(level):
            # number of (mi,mo) for =lmax <lmin
            return (2*lmax+1)*(level**2)
        def foo1inv(N):
            return np.sqrt(N/(2*lmax+1))
        def foo2(idx):
            return np.floor(foo1inv(idx+0.5)).astype(np.int32)
        lmin = foo2(idx_wrt_level//2)
        idx_wrt_lmin = idx_wrt_level - 2*foo1(lmin)
        ms = (2*lmax+1)*(2*lmin+1)
        li = np.where(idx_wrt_lmin < ms, lmin, lmax)
        lo = np.where(idx_wrt_lmin < ms, lmax, lmin)

        idx_mm = idx_wrt_lmin % ms
        mi = idx_mm // (2*lo+1) - li
        mo = idx_mm %  (2*lo+1) - lo
        
        lms = np.stack([li, lo, mi, mo], axis=-1)

    elif dom_type == DomType.ISOBI:
        # Order: lmax -> lmin -> li -> m
        lmin = idx2lmax(idx_wrt_level//2, DomType.UNI)
        idx_wrt_lmin = idx_wrt_level - 2*level2num(lmin, DomType.UNI)
        ms = (2*lmin+1) # number of 'm'
        m = (idx_wrt_lmin % ms) - lmin
        li = np.where(idx_wrt_lmin < ms, lmin, lmax)
        lo = np.where(idx_wrt_lmin < ms, lmax, lmin)
        lms = np.stack([li, lo, m], axis=-1)
    
    else:
        assert dom_type == DomType.ISOBI2, f"Invalid domain type: {dom_type}"
        # Order: lmax -> lmin -> li -> mi -> mo
        def foo1(level):
            return level*(2*level-1)
        def foo1inv(N):
            return (np.sqrt(1+8*N)+1)/4
        def foo2(idx):
            return np.floor(foo1inv(idx+0.5)).astype(np.int32)
        lmin = foo2(idx_wrt_level//2)
        idx_wrt_lmin = idx_wrt_level - 2*foo1(lmin)
        ms = 4*lmin + 1
        li = np.where(idx_wrt_lmin < ms, lmin, lmax)
        lo = np.where(idx_wrt_lmin < ms, lmax, lmin)

        mi2 = idx_wrt_lmin % ms - 2*lmin
        mi = np.sign(mi2) * ((np.abs(mi2)+1)//2)
        mo = np.where(mi2 % 2 == 0, mi, -mi)
        lms = np.stack([li, lo, mi, mo], axis=-1)
    
    if unstack == False:
        return lms
    else:
        return npunstack(lms, axis=-1)


def level2lms(level:    int,
              dom_type: DomType,
              unstack:  bool = False
             )       -> ArrayLike: # N(L)xd, d = |l,m,...|
    N = level2num(level, dom_type)
    idx = np.arange(N)
    return idx2lms(idx, dom_type, unstack=unstack)


def SH_upto(theta:    ArrayLike, # [*]
            phi:      ArrayLike, # [*]
            level:    int,       # == L
            spin2:    bool,      # scalar SH for False, s2 SH for True
            sh_type:  Optional[SHType] = SHType.COMP,
            ang_type: Optional[AngType] = AngType.RAD,
            rotated:  Optional[ArrayLike] = None
           )       -> ArrayLike: # *xN(L)
    if spin2 == False: s = 0
    else:              s = 2
    sh_type = SHType(sh_type)
    theta, phi = radian_regularize(ang_type, theta, phi)
    wigner = spherical.Wigner(level - 1)
    R = quat.array.from_euler_angles(phi, theta, 0)
    
    if not rotated is None:
        rotated = rotation2quat(rotated)
        R = rotated.inverse * R

    SH_res = wigner.sYlm(s, R)
    ## NOTE spherical.Wigner.sYlm returns consistent size indep. of `s` with first zeros.
    if sh_type == SHType.REAL and spin2 == False:
        ## NOTE s2-SH is always complex
        lms = level2lms(level, DomType.UNI)
        l, m = lms[:,0], lms[:,1]
        idx_negm = lms2idx(np.stack([l, -m], axis=-1), DomType.UNI)
        SH_negm = SH_res[...,idx_negm]
        SH_res = np.select([m > 0, m < 0], [SH_res.real*np.sqrt(2), SH_negm.imag*np.sqrt(2)], SH_res.real)
        
    return SH_res

def s2SH_integral_uptol(level:   int,
                        m:       int,
                        theta_f: ArrayLike,      # [*]
                        theta_i: ArrayLike = 0.0 # [*]
                       )      -> ArrayLike:      # [*, level-2], level-2 =:N
    '''
    \int_{theta_i}^{theta_f}{ s2Ylm(theta, 0) d{theta}}
    See `Integral over cos theta` in `2022.06.23. Spin-weighted Spherical Harmonics.md` for more details
    '''
    l = np.arange(2, level)
    ## NOTE \int_[theta_i, theta_f] f(cos(theta))sin(theta)d(theta) == \int_[cos(theta_f), cos(theta_i)] f(x)d(x)
    zf = np.expand_dims(np.cos(theta_i), -1) # [*, N]
    zi = np.expand_dims(np.cos(theta_f), -1) # [*, N]
    zf, zi = np.broadcast_arrays(zf, zi)
    r2f = 1 - zf*zf
    r2i = 1 - zi*zi
    if m == -2:
        const = np.sqrt((2*l+1)/np.pi)/2
    else:
        raise NotImplementedError("Computing sqrt((l-2)!/(l+2)!)Alm is not implemented yet.")
    
    Pf = sp.lpmv(m, l, zf)
    Pi = sp.lpmv(m, l, zi)
    P_diff = Pf - Pi
    Int_P = np.zeros(Pf.shape) # \int LegendreP
    Int_Poverr2 = np.zeros(Pf.shape) # \int LegendreP / (1-z*z)
    res = np.zeros(Pf.shape)
    if abs(m) == 2:
        if m == 2:
            sign2 = 1
            # sign3 = 1
        else:
            sign2 = 1/24
            # sign3 = 1/120

        z2_sum = zi*zi + zf*zf
        z_diff = zf - zi
        z_sum = zf + zi
        Int_P[...,0] = sign2 * z_diff*( 3 - zi*zf - z2_sum )
            ## NOTE factorize 3*(zf-zi) - (zf**3-zi**3) to avoid catastrophic cancellation
        Int_Poverr2[...,0] = sign2 * 3 * z_diff
        res[...,0] = -2*Int_P[...,0] + m*P_diff[...,0] + m*m*Int_Poverr2[...,0]
    else:
        raise NotImplementedError()
    
    for l_curr in l[1:]:
        i = l_curr - 2
        if l_curr == l[1]:
            Int_P_pp = 0
            Int_Poverr2_pp = 0
        else:
            Int_P_pp = Int_P[...,i-2]
            Int_Poverr2_pp = Int_Poverr2[...,i-2]
        
        Int_P[...,i] = (l_curr-2)*(l_curr+m-1)*Int_P_pp - (2*l_curr-1)*(r2f*Pf[...,i-1] - r2i*Pi[...,i-1])
        Int_P[...,i] /= (l_curr+1)*(l_curr-m)

        Int_Poverr2[...,i] = -(2*l_curr-1)*P_diff[...,i-1] + l_curr*(l_curr+m-1)*Int_Poverr2_pp
        Int_Poverr2[...,i] /= (l_curr - m)*(l_curr - 1)
    
    lm1 = l-1
    res[...,1:] = -l[1:]*lm1[1:]*Int_P[...,1:] + 2/l[1:]*(
                  m*lm1[1:]*P_diff[1:] + (l[1:]+m)*P_diff[...,:-1] +
                  m* ( lm1[1:]*m*Int_Poverr2[...,1:] + (l[1:]+m)*Int_Poverr2[...,:-1] )
                )
    res *= const
    return res
from enum import Enum
from typing import Tuple, List, Union, IO, Optional

import numpy as np
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt

from .util import tuple_add
########################################
### Python support
########################################
global EPSILON
EPSILON = 1.0e-7

# MEMORY_PER_PASS = 80*(1024**3) # 80GiB
MEMORY_PER_PASS = 24*(1024**3) # 24GiB

def set_epsilon(eps):
    global EPSILON
    EPSILON = eps

class AngType(Enum):
    ''' Enumeration type for angles '''
    RAD = 0 # radian
    DEG = 1 # degree

def assert_error_bound(val:       float, # or ArrayLike
                       eps:       Optional[float] = None,
                       name:      Optional[str]   = "",
                       allow_nan: Optional[bool]  = False,
                       quiet:     Optional[bool]  = True):
    if not quiet:
        print("\n# assert error bound:")
    ## Deal with default eps
    global EPSILON
    if eps is None:
        eps = EPSILON
    
    if np.isscalar(val):
        scalar = abs(val)
    else:
        val_abs = abs(val)
        if allow_nan:
            if not quiet:
                n_nan, n_total = np.sum(np.isnan(val)), val.size
                print(f"[WARNING] # of NaN: {n_nan}/{n_total} ({n_nan/n_total*100:.6f}%)")
            min_val = np.nanmin(val_abs)
            max_val = np.nanmax(val_abs)
            mean_val = np.nanmean(val_abs)
        else:
            min_val = np.min(val_abs)
            max_val = np.max(val_abs)
            mean_val = np.mean(val_abs)
        rms_val = rms(val_abs, allow_nan=allow_nan)
        scalar = max_val

    if not quiet:
        if np.isscalar(val):
            print(f"{name}: scalar {val}")
        else:
            print(f"{name}: shape={val.shape}, ", end="")
            print("min=%s\tmean=%s\trms=%s\tmax=%s" % (min_val, mean_val, rms_val, max_val) )
    assert scalar < eps, f"{name} ({scalar}) is larger than the tolerance."

##############################
### NumPy support
##############################
def rowvec(x):
    return np.asarray(x).reshape((1,-1))
def colvec(x):
    return np.asarray(x).reshape((-1,1))
def vec1d(x):
    return np.asarray(x).reshape((-1,))

def npunstack(a: ArrayLike, axis: int = -1
           ) -> ArrayLike:
    '''
    Ref: https://stackoverflow.com/questions/64097426/is-there-unstack-in-numpy
    '''
    return np.moveaxis(a, axis, 0)

unit_x = np.array([1, 0, 0])
unit_y = np.array([0, 1, 0])
unit_z = np.array([0, 0, 1])
J_conj = np.array([[1,0], [0,-1]])
## NOTE For any complex array z, following holds:
## >>> np.allclose( J_conj @ comp2vec(z), comp2vec(z.conj()) )

def meansq(a:         ArrayLike,
           axis:      int = None,
           keepdims:  bool = np._NoValue,
           allow_nan: bool = False
          )        -> ArrayLike:
    sq = np.abs(a)**2
    if allow_nan:
        return np.nanmean(sq, axis=axis, keepdims=keepdims)
    else:
        return np.mean(sq, axis=axis, keepdims=keepdims)

def rms(a:         ArrayLike, /,
        axis:      int = None,
        keepdims:  bool = np._NoValue, *,
        allow_nan: bool = False
       )        -> ArrayLike:
    sq = np.abs(a)**2
    if allow_nan:
        return np.sqrt(np.nanmean(sq, axis=axis, keepdims=keepdims))
    else:
        return np.sqrt(np.mean(sq, axis=axis, keepdims=keepdims))

def is_zerovec(vec:      ArrayLike, /, # [*, n] (axis:-1)
               axis:     int = -1,
               keepdims: bool = False
               )      -> ArrayLike: # [*] (axis:-1, keepdims:False), [*, 1] (axis:-1, keepdims:True)
    ## Return a boolean mask whether each vector is the zero-vector exactly.
    return np.logical_and.reduce(vec == 0.0, axis=axis, keepdims=keepdims)

def vecdot(a, b, /, axis=-1):
    return (a*b).sum(axis)

def normalize(vec: ArrayLike, /,
              axis: Optional[int] = -1, *,
              out: Optional[np.ndarray] = None,
              where: Optional[ArrayLike] = True
             ) -> ArrayLike:
    """
    [WARN] The result with `where` mask might be meaningless
           unless `where.all(axis) == where.any(axis)`
    """
    if where is True and out is None:
        return vec / np.linalg.norm(vec, axis=axis, keepdims=True)
    else:
        return np.divide(vec,
                         np.linalg.norm(vec, axis=axis, keepdims=True),
                         out=out,
                         where=where)

def normalize_safe(vec: ArrayLike, /, axis: int = -1) -> ArrayLike:
    """
    Normalizes vectors, but return zero for the zero vectors.
    [WARN] Deciding use it should require carefully check your context.
    """
    vec = np.asarray(vec)
    norm = np.linalg.norm(vec, axis=axis, keepdims=True)
    mask = np.broadcast_to(norm != 0.0, vec.shape)
    res = np.divide(vec, norm, where=mask)
    res[~mask] = 0.0
    return res


def signcomplex(mat: ArrayLike) -> ArrayLike:
    mat = np.asarray(mat)
    return np.divide(mat, np.abs(mat),
                     where = mat!=0.0,
                     out = np.full(mat.shape, 0.0, dtype=mat.dtype))

def powmag(mat: ArrayLike, exponent: ArrayLike) -> ArrayLike:
    '''
    Taking power of only the magnitue, keeping the sign (might be complex)
    '''
    sign = signcomplex(mat)
    return np.where(mat == 0.0, 0.0, np.power(abs(mat), exponent) * sign)

def matmul_vec1d(mat: ArrayLike, # [*, m, n]
                 vec: ArrayLike, # [*, n]
                )  -> ArrayLike: # [*, m]
    """
    Product of a matrix `mat` and a vector `vec`
    when considering `vec.shape[:-1]` axes indicate just enumerating data for vectorized computation.
    If `vec.ndim == 1`, the result will be simply identical to `mat @ vec`.
    """
    return (mat @ np.expand_dims(vec, -1)).squeeze(-1)

def comp2vec(x:    ArrayLike,  # * complex
             axis: int = -1
            )   -> ArrayLike: # *x2 real (default axis)
    assert np.iscomplexobj(x)
    return np.stack([x.real, x.imag], axis=axis)

def vec2comp(x:    ArrayLike, # [*,2] real (default axis)
             axis: int = -1
            )   -> np.ndarray: # [*] complex
    x = np.asarray(x)
    assert np.isrealobj(x)
    x_reg = np.moveaxis(x, axis, -1)
    assert x_reg.shape[-1] == 2
    y = x_reg[...,1]*1j
    y += x_reg[...,0]
    return y

def comp2mat(x:    ArrayLike, # [*] complex
             axes: Tuple[int,int] = (-2,-1)
            )   -> ArrayLike: # [*,2,2] real (default axis)
    '''
    A field-isomorphic embedding of $\C$ into $\R^{2\times2}$.
    '''
    assert np.iscomplexobj(x)
    res = np.array([[x.real, -x.imag], [x.imag, x.real]])
    return np.moveaxis(res, (0,1), axes)

def mat2comppair(x:    ArrayLike, # [*,2,2] real (default axis)
                 axes: Tuple[int,int] = (-2,-1)
                )   -> Tuple[ArrayLike,ArrayLike]: # [*] complex for each
    '''
    Following should be satisfied for any array x[...,2,2]:
    >>> z1, z2 = mat2comppair(x)
    >>> np.allclose( comp2mat(z1) + comp2mat(z2) @ J_conj, x)
    
    NOTE `x` can be written in terms of `z1` and `z2` as follows:
    `x = np.array([[z1.real + z2.real, -z1.imag + z2.imag],
                   [z1.imag + z2.imag, z1.real - z2.real]])`
    '''
    x = np.asarray(x)
    assert np.isrealobj(x)
    x = np.moveaxis(x, axes, (-2,-1))
    assert x.shape[-2:] == (2,2)
    
    # z1r = (x[...,0,0] + x[...,1,1]) / 2
    # z1i = (x[...,1,0] - x[...,0,1]) / 2
    # z1 = z1r + 1j*z1i
    # del z1r, z1i
    z1 = 1j*(x[...,1,0] - x[...,0,1]) / 2
    z1 += (x[...,0,0] + x[...,1,1]) / 2
    
    # z2r = (x[...,0,0] - x[...,1,1]) / 2
    # z2i = (x[...,1,0] + x[...,0,1]) / 2
    # z2 = z2r + 1j*z2i
    # del z2r, z2i
    z2 = 1j*(x[...,1,0] + x[...,0,1]) / 2
    z2 += (x[...,0,0] - x[...,1,1]) / 2
    return z1, z2
    
## NOTE Delete these performs same as np.save and np.load
# def npsave(x: ArrayLike, filename: str):
#     with open(filename, 'wb') as f:
#         np.save(f, x)

# def npload(filename: str) -> ArrayLike:
#     with open(filename, 'rb') as f:
#         x = np.load(f)
#     return x

def interval_length(param_range: ArrayLike, lb: float, ub: float) -> ArrayLike:
    """
    For a 1D array `param_range`, compute integration weight `res`
    Constraint:
        `param_range`: increasing order
        `lb <= param_range[0] <= param_range[-1] <= ub`
        `res.sum() == ub - lb`
    TODO: testing
    """
    lb_mod = vec1d(2*lb - param_range[0])  # 'mod' named after 'modified'
    ub_mod = vec1d(2*ub - param_range[-1])
    pre = np.concatenate([lb_mod, param_range[:-1]])
    nex = np.concatenate([param_range[1:], ub_mod])
    return (nex-pre)/2

def parse_cfg_pass(cfg_pass: Tuple[int, int]):
    i_pass, n_pass = cfg_pass
    assert type(n_pass) is int and n_pass >= 1
    assert type(i_pass) is int and i_pass >= 0 and i_pass < n_pass
    return i_pass, n_pass

def slice_pass(array_list: List[ArrayLike], # same sizes of arrays
               cfg_pass:   None | Tuple[int, int]
              )         -> List[ArrayLike]:
    if cfg_pass is None:
        return array_list
    else:
        i_pass, n_pass = parse_cfg_pass(cfg_pass)
        res_list = []
        for idx,arr in enumerate(array_list):
            arr_1d = arr.view().reshape(-1)
            if idx == 0:
                N_grid = arr_1d.size
                i = N_grid*i_pass // n_pass
                j = N_grid*(i_pass+1) // n_pass
            else:
                assert N_grid == arr_1d.size
            res_list.append(arr_1d[i:j])
        return res_list
    
def prebc_chan(sh1:       Tuple[int,],
               sh2:       Tuple[int,],
               nd_common: int,
               chan_tdot: bool
               )       -> Tuple[Tuple, Tuple, Tuple]:
    '''
    inputs:
        sh1 == (*g, *c1)
        sh2 == (*g, *c2)
        nd_common == |*g|
    outputs:
        sh_ch12 == np.broadcast_shapes(c1, c2) | c1 + c2
        ax_exp1: tuple of nonnegative integers
        ax_exp2: tuple of nonnegative integers
        ax_exp_common: tuple of nonnegative integers
    -----
    Pseudo example (readibility rather than runnability)
    ```
    >>> sh1 = g + c1; sh2 = g + c2; nd_common = len(g)
    >>> arr1 = np.zeros(sh1)
    >>> arr2 = np.zeros(sh2)
    >>> arrg = np.zeros(g)
    >>>
    >>> sh_12, ax_exp1, ax_exp2, ax_exp_common = prebc_chan(sh1, sh2, nd_common, chan_tdot)
    >>> # All following three arrays are broadcastable
    >>> np.expand_dims(arr1, ax_exp1)
    >>> np.expand_dims(arr2, ax_exp2)
    >>> np.expand_dims(arrg, ax_exp_common)
    ```
    Named after "prepare braodcasting channel"
    '''
    sh_ch1 = sh1[nd_common:]
    sh_ch2 = sh2[nd_common:]
    ax_ch1 = range(nd_common, len(sh1))
    ax_ch2 = range(nd_common, len(sh2))
    if chan_tdot:
        sh_ch12 = sh_ch1 + sh_ch2
        ax_exp1 = tuple_add(ax_ch2, len(ax_ch1))
        ax_exp2 = tuple(ax_ch1)
        ax_exp_common = ax_exp2 + ax_exp1
    else:
        sh_ch12 = np.broadcast_shapes(sh_ch1, sh_ch2)
        ax_exp1 = tuple(range(nd_common, nd_common+len(sh_ch12)-len(sh_ch1)))
        ax_exp2 = tuple(range(nd_common, nd_common+len(sh_ch12)-len(sh_ch2)))
        ax_exp_common = tuple(range(nd_common, nd_common+len(sh_ch12)))
    return sh_ch12, ax_exp1, ax_exp2, ax_exp_common

def common_dtype(x: ArrayLike, y: ArrayLike) -> np.dtype:
    ## Return a common super-dtype for given two `numpy.ndarray`s
    return (np.asarray(x).ravel()[0] + np.asarray(y).ravel()[0]).dtype

def pltsavemat(x: ArrayLike, filename: str):
    plt.matshow(x)
    plt.colorbar()
    plt.savefig(filename)

########################################
### Argument regularizers
########################################

def radian_regularize(angle_type, *arrays):
    if angle_type == AngType.RAD:
        return arrays
    else: # AngType.DEG
        assert angle_type == AngType.DEG, f"Invalid angle_type:{angle_type}"
        res = []
        for arr in arrays:
            if arr is None:
                res.append(arr)
            else:
                res.append(np.deg2rad(arr))
        return res

def radian_deregularize(angle_type, *arrays):
    if angle_type == AngType.RAD:
        return arrays
    else: # AngType.DEG
        assert angle_type == AngType.DEG, f"Invalid angle_type:{angle_type}"
        res = []
        for arr in arrays:
            res.append(np.rad2deg(arr))
        return res

def angle_convert(angle_type_from, angle_type_to, *arrays):
    assert isinstance(angle_type_from, AngType) and isinstance(angle_type_to, AngType), f"Invalid angle_types:{angle_type_from}, {angle_type_to}"
    if angle_type_from == angle_type_to:
        return arrays
    else:
        if angle_type_from == AngType.RAD and angle_type_to == AngType.DEG:
            func_conv = np.rad2deg
        else: # angle_type_from == 1 and angle_type_to == 0:
            func_conv = np.deg2rad
        res = []
        for arr in arrays:
            if arr is None:
                res.append(arr)
            else:
                res.append(func_conv(arr))
        return res
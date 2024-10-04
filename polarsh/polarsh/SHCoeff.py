'''
NOTE Do not recommend to import this submodule as `from SHcoeff import *`.
     Please use `import SHcoeff` instead.
'''

from __future__ import annotations
import math
from typing import Sequence, Union, Optional
from tabulate import tabulate
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as scipyRotation
import quaternionic as quat
import spherical
import matplotlib.figure

import polarsh
from .util import row2str_short, format_complex
from .array import *
from .sphere import *
from .plot import *
from .SH import *
from .__conv_helpers import assert_coeff_s2s, assert_coeff_sv, assert_coeff_v2v, get_U

########################################
### Support functions
########################################
def assert_dctype(coeff:    ArrayLike , # [*, N] | [*, N, p] | [*, N, po, pi]
                  dom_type: DomType,    # Any    | UNI       | BI,ISOBI,ISOBI2
                  cod_type: CodType,    # SCALAR | POLARp    | POLARp
                 ) ->       Tuple[int, int]:
    '''
    Assert given SH coefficient array has an valid shape for given domain and codomain type.
    Return level, axis_N
    '''
    assert isinstance(dom_type, DomType)
    assert isinstance(cod_type, CodType)
    
    coeff = np.asarray(coeff)
    if cod_type == CodType.SCALAR:
        axis_N = -1
    else:
        if dom_type == DomType.UNI:
            axis_N = -2
            assert coeff.shape[-1] == cod_type
        else:
            axis_N = -3
            assert coeff.shape[-2:] == (cod_type, cod_type)
    N = coeff.shape[axis_N]
    level = num2level_assert(N, dom_type)

    return level, axis_N

def assert_unimat(coeff:    ArrayLike, # [*, No, Ni, (po, pi)]
                  cod_type: CodType,
                 ) ->       Tuple[int, int]: # (level, -1 or -3)
    assert isinstance(cod_type, CodType)
    
    coeff = np.asarray(coeff)
    if cod_type == CodType.SCALAR:
        axis_N = -1
    else:
        axis_N = -3
        assert coeff.shape[-2:] == (cod_type, cod_type)
    assert coeff.shape[axis_N] == coeff.shape[axis_N - 1]
    N = coeff.shape[axis_N]
    level = num2level_assert(N, DomType.UNI)

    return level, axis_N

def assert_shtype(coeff: ArrayLike, sh_type: SHType) -> np.ndarray:
    coeff = np.asarray(coeff)
    if not isinstance(sh_type, SHType):
        raise TypeError()
    if sh_type == SHType.REAL:
        assert np.isrealobj(coeff)
    elif sh_type == SHType.COMP:
        coeff = coeff.astype(complex)
    else:
        raise TypeError(f"Invalid argument {sh_type=}.")
    return coeff

def _get_channel(arr: np.ndarray, ndim_chan: int, ch: Union[None, int, Tuple[int]]):
    if ndim_chan == 0:
        assert ch is None or ch == (), f"{arr.shape = }, {ndim_chan = }, {ch = }"
        return arr
    elif ndim_chan == 1:
        if hasattr(ch, '__len__'):
            ch_, = ch
        elif ch is None:
            ch_ = 0
        else:
            ch_ = ch
        ch_ = int(ch_)
        return arr[ch_]
    else:
        if ch is None:
            ch_ = tuple([0] * ndim_chan)
        else:
            assert len(ch) == ndim_chan, f"{arr.shape = }, {ndim_chan = }, {ch = }"
            ch_ = ch
        return arr[ch_]

########################################
### class SHCoeff
########################################
class SHCoeff:
    def __imul__(self, x):
        self.coeff *= x
    def __itruediv__(self, x):
        self.coeff /= x

########################################
### class SHVec
########################################
class SHVec(SHCoeff):
    def __init__(self,
                 coeff:    ArrayLike, # [*, N] | [*, N, p]
                 cod_type: CodType,   # SCALAR | POLARp
                 sh_type:  SHType) -> SHVec:
        # ---------- Assertion ----------
        cod_type = CodType(cod_type)
        sh_type = SHType(sh_type)
        self.level, self.axis_N = assert_dctype(coeff, DomType.UNI, cod_type)
        self.coeff = assert_shtype(coeff, sh_type)
        # ---------- Attributes ----------
        self.N = self.coeff.shape[self.axis_N]
        self.cod_type = cod_type
        self.sh_type = sh_type
        # ---------- Protected ----------
        self.shape_chan = self.coeff.shape[:self.axis_N]
        self.ndim_chan = len(self.shape_chan)

    @classmethod
    def from_npy_file(cls, filename: Union[str, Path], sh_type: SHType) -> SHVec:
        """
        Read `.npy` file
        -----
        Input:
            filename: str | Path float64 binary file through `np.save` shape [*channel, N(level), cod_type]
                        NOTE `self.coeff` or `self.coeff[..., None]` depending on cod_type
        """
        filename = Path(filename)
        if filename.suffix != ".npy":
            raise ValueError(f"The method only support `.npy`. Given file name is: {filename}")
        
        coeff = np.load(filename)
        cod_type = CodType(coeff.shape[-1])
        if cod_type == CodType.SCALAR:
            coeff = coeff.squeeze(-1)
        
        return cls(coeff, cod_type, sh_type)
    
    @classmethod
    def from_npz_file(cls, filename_npz: Union[str, Path]) -> SHVec:
        filename_npz = Path(filename_npz)
        assert filename_npz.suffix == ".npz", f"Invalid file extension for {filename_npz}"
        npz = np.load(filename_npz)
        return cls(npz['coeff'], str(npz['cod_type']), str(npz['sh_type']))
    
    @classmethod
    def from_float_file(cls, filename: Union[str, Path], cod_type: CodType, sh_type: SHType) -> SHVec:
        """
        Read .float file
        -----
        Input:
            filename: str | Path binary file with flatten array of shape:
                [N(level), cod_type, channel], float32
            cod_type: CodType
            sh_type: SHType
        -----
        See also:
            polar-harmonics-code/python/file_simple_process.py: file_npy2float
        """
        filename = str(filename)
        if filename.suffix != ".float":
            raise ValueError(f"The method only support `.float`. Given file name is: {filename}")
        
        with open(filename, "rb") as f:
            coeff = np.fromfile(f, dtype=np.float32)
        coeff = np.moveaxis(coeff.reshape(-1, int(cod_type), 3), -1, 0)

        return cls(coeff, cod_type, sh_type)

    @classmethod
    def from_single_idx(cls, l: int, m: int, p: int, sh_type: SHType) -> SHVec:
        N = level2num(l+1, DomType.UNI)
        cod_type = p+2 if p == 1 else p+1
        if p == 0:
            coeff = np.zeros((N))
            coeff[lms2idx([l, m], DomType.UNI)] = 1.0
        else:
            coeff = np.zeros((N, cod_type))
            coeff[lms2idx([l, m], DomType.UNI), p] = 1.0
        cod_type = CodType(cod_type)
        return cls(coeff, cod_type, sh_type)

    @classmethod
    def zeros_like(cls, shv: SHVec) -> SHVec:
        return cls(np.zeros_like(shv.coeff), shv.cod_type, shv.sh_type)
    
    def __repr__(self) -> str:
        if self.cod_type == CodType.SCALAR:
            str_polar = "(SCALAR)"
        else:
            str_polar = f"| p: {int(self.cod_type)}"
        str_shapec = str(self.shape_chan)[1:-1]
        res = f"{self.__class__.__name__}[c: {str_shapec} | N: {self.N} {str_polar}][\n"
        res += f"  level = {self.level},\n"
        res += f"  cod_type = {self.cod_type!r},\n"
        res += f"  sh_type = {self.sh_type!r},\n"
        res += f"  coeff.shape = {self.coeff.shape},\n"
        res += f"  coeff.dtype = {self.coeff.dtype},\n"
        res += "]"
        return res

    # ---------- Trivial operations ---------
    def __add__(self, x) -> SHVec:
        if isinstance(x, SHVec):
            assert self.cod_type == x.cod_type
            assert self.sh_type == x.sh_type
            return SHVec(self.coeff + x.coeff, self.cod_type, self.sh_type)
        else:
            raise TypeError(f"Not supported data type: {type(x)}")
    
    def __sub__(self, x) -> SHVec:
        if isinstance(x, SHVec):
            assert self.cod_type == x.cod_type
            assert self.sh_type == x.sh_type
            return SHVec(self.coeff - x.coeff, self.cod_type, self.sh_type)
        else:
            raise TypeError(f"Not supported data type: {type(x)}")
        
    def __rmul__(self, x) -> SHVec:
        if np.isscalar(x):
            # NOTE even if x is complex and self.sh_type is REAL,
            #      the constructor will assert it.
            return SHVec(x*self.coeff, self.cod_type, self.sh_type)
        else:
            raise TypeError(f"Not supported data type: {type(x)}")
    
    # ---------- Matrix multiplication ----------
    def __rmatmul__(self, x) -> SHVec:
        # If x is SHMat or SHConv, their __matmul__ should have processed.
        flag_rotation = False
        if isinstance(x, quat.array):
            flag_rotation = True
            R = x
        elif isinstance(x, scipyRotation):
            flag_rotation = True
            R = scipyRotaion2quat(x)
        else:
            x = np.asarray(x)
            if x.shape in [(3,), (1, 3), (3, 1)]:
                flag_rotation = True
                R = quat.array.from_axis_angle(x.squeeze())
            else:
                raise TypeError()
            
        # ------------------------------------
        # ---------- Apply Rotation ----------
        # ------------------------------------
        if flag_rotation:
            wigner = spherical.Wigner(self.level-1)
            D = wigner.D(R).conj()
            
            if self.cod_type == CodType.SCALAR:
                coeff_rot = _SHcoeff_apply_rotation(self.coeff, R,
                                                    sh_type=self.sh_type, D_cache=D)
            else:
                coeff_rot = np.zeros_like(self.coeff)#, dtype=np.complex128)
                # TODO Can `dtype=np.complex128` above be generalized?

                #* Scalar components
                for p in cidx_scalar(self.cod_type, out_type = tuple):
                    coeff_rot[...,p] = _SHcoeff_apply_rotation(self.coeff[...,p], R,
                                                               sh_type=self.sh_type, D_cache=D)
            
                #* Linear polarization components
                iD1 = 0
                if self.cod_type == CodType.POLAR2:
                    p1, p2 = 0, 1
                else:
                    p1, p2 = 1, 2
                for l in range(self.level):
                    im1 = level2num(l, DomType.UNI)
                    im2 = level2num(l+1, DomType.UNI)
                    nm = im2 - im1
                    iD2 = iD1 + nm*nm
                    Dl = D[iD1:iD2].reshape(nm, nm)
                    coeff_rot[...,im1:im2,p1] \
                        = matmul_vec1d(Dl.real, self.coeff[...,im1:im2,p1]) \
                        - matmul_vec1d(Dl.imag, self.coeff[...,im1:im2,p2])
                    coeff_rot[...,im1:im2,p2] \
                        = matmul_vec1d(Dl.imag, self.coeff[...,im1:im2,p1]) \
                        + matmul_vec1d(Dl.real, self.coeff[...,im1:im2,p2])
                    iD1 = iD2
            # if self.sh_type == SHType.REAL:
            #     np.allclose(coeff_rot.imag, 0)
            #     coeff_rot = coeff_rot.real
            return SHVec(coeff_rot, self.cod_type, self.sh_type)

    def allclose(self, shv: SHVec, rtol: Optional[float]=1e-05, atol: Optional[float]=1e-08) -> bool:
        if not isinstance(shv, SHVec):
            raise TypeError(f"Invalid type: {type(shv)=}. The type must be `SHVec`.")
        if self.cod_type != shv.cod_type:
            raise ValueError(f"Attributes `cod_type` for `self` and `shv` must be equal, but currently:\n"
                             f"{self.cod_type = }, {shv.cod_type = }")
        if self.sh_type != shv.sh_type:
            raise ValueError(f"Attributes `sh_type` for `self` and `shv` must be equal, but currently:\n"
                             f"{self.sh_type = }, {shv.sh_type = }")
        return np.allclose(self.coeff, shv.coeff, rtol=rtol, atol=atol)
    
    @property
    def chan(self) -> _SHVec_channel_indexer:
        return _SHVec_channel_indexer(self)
    
    def copy(self) -> SHVec:
        return SHVec(self.coeff.copy(), self.cod_type, self.sh_type)
    
    def at_lm(self, l: ArrayLike, m: ArrayLike) -> np.ndarray:
        idx = lms2idx([l, m], DomType.UNI)
        if self.cod_type == CodType.SCALAR:
            return self.coeff[..., idx]
        else:
            return self.coeff[..., idx, :]
    
    def at(self,
           lm: Tuple[ArrayLike, ArrayLike],
           p:   Optional[ArrayLike] = None
          ) ->  np.ndarray:
        l, m = lm
        if self.cod_type == CodType.SCALAR:
            assert p is None
            return self.at_lm(l, m)
        else:
            if p is None:
                return self.at_lm(l, m)
            else:
                return self.at_lm(l, m)[..., p]
    
    def slice_l(self, l: int) -> np.ndarray:
        idx_i = level2num(l, DomType.UNI)
        idx_f = level2num(l+1, DomType.UNI)
        if self.cod_type == CodType.SCALAR:
            return self.coeff[..., idx_i:idx_f]
        else:
            return self.coeff[..., idx_i:idx_f, :]

    def cut(self,
            level:    Union[int, SHCoeff, None] = None,
            cod_type: Optional[CodType] = None
           ) ->       SHVec:
        # ---------- Cut level ----------
        if level is None:
            array_to = self.coeff
        else:
            if isinstance(level, SHCoeff):
                level = level.level
            if level > self.level:
                raise ValueError(f"The parameter {level=} cannot be larger than {self.level=}.")
            
            N = level2num(level, DomType.UNI)

            if self.cod_type == CodType.SCALAR:
                array_to = self.coeff[..., :N]
            else: # cod_type == CPOLARp
                array_to = self.coeff[..., :N, :]

        # ---------- Cut codomain type ---------- 
        if cod_type is None:
            cod_type_final = self.cod_type
        else:
            cod_type_final = cod_type
            cidx = cidx_to(self.cod_type, cod_type)
        if cod_type is not None and self.cod_type != CodType.SCALAR:
            array_to = array_to[..., cidx]
        return SHVec(array_to, cod_type_final, self.sh_type)

    def pad(self,
            level:    Union[int, SHCoeff, None],
            cod_type: Optional[CodType] = None
           ) ->    SHCoeff:
        # ---------- `shape_to` ----------
        if level is None:
            N_to = self.N
        else:
            if isinstance(level, SHCoeff):
                level = level.level
            if level < self.level:
                raise ValueError(f"The parameter {level=} cannot be smaller than {self.level=}.")
            N_to = level2num(level, DomType.UNI)
        if cod_type is None:
            cod_type = self.cod_type
        if int(cod_type) < int(self.cod_type):
            raise ValueError(f"The parameter {cod_type!r} cannot be smaller than {self.cod_type!r}.")
        if cod_type == CodType.SCALAR:
            shape_to = self.shape_chan + (N_to,)
        else:
            shape_to = self.shape_chan + (N_to, int(cod_type))

        # ---------- Main ----------
        array_to = np.zeros(shape_to, dtype=self.coeff.dtype)
        if cod_type == CodType.SCALAR:
            array_to[..., :self.N] = self.coeff
        else:
            cidx = cidx_to(cod_type, self.cod_type)
            array_to[..., :self.N, cidx] = self.coeff
        return SHVec(array_to, cod_type, self.sh_type)
    

    def to_shtype(self, sh_type: SHType) -> SHVec:
        '''
        Refer to `test_Sotkes_s2SH.ipynb`.
        '''
        sh_type = SHType(sh_type)
        if self.cod_type != CodType.SCALAR:
            idx_sc = cidx_scalar(self.cod_type)
            idx_lp = cidx_lpolar(self.cod_type)
        rt2 = math.sqrt(2)
        res = np.zeros_like(self.coeff, dtype=np.complex128)
                
        if self.cod_type == CodType.SCALAR:
            coeff_sc = self.coeff
            res_sc = res
        else:
            coeff_sc = np.moveaxis(self.coeff[..., idx_sc].view(), -1, -2)
            res_sc = np.moveaxis(res[..., idx_sc].view(), -1, -2)
            res[..., idx_lp] = self.coeff[..., idx_lp]
            res[..., idx_lp] = self.coeff[..., idx_lp]
        
        l, m = level2lms(self.level, DomType.UNI, unstack=True)
        phase = (-1)**(m % 2)
        idx_negm = lms2idx(np.stack([l, -m], axis=-1), DomType.UNI)
        coeff_negm = coeff_sc[..., idx_negm]
        
        match self.sh_type, sh_type: # (from, to)
            # Reference: the complex conjugate of Eqs. (S-53) and (S-54) in [Yi et al. 2024], resp.
            # Here, `coeff_sc` and `coeff_negm` indicate diagonal and off-diagonal elements, resp., in the both equations.
            case SHType.REAL, SHType.COMP:
                res_sc[:] = np.select([m > 0, m < 0],
                                    [(coeff_sc - 1j*coeff_negm)/rt2,
                                    phase*(1j*coeff_sc + coeff_negm)/rt2],
                                    coeff_sc)
            case SHType.COMP, SHType.REAL:
                # Reference: the complex conjugate of Eq. (S-52) in [Yi et al. 2024]
                res_sc[:] = np.select([m > 0, m < 0],
                                    [(coeff_sc + phase*coeff_negm)/rt2,
                                    (-phase*coeff_sc + coeff_negm)*1j/rt2],
                                    coeff_sc)
            case _:
                raise ValueError(f"shtype_from ({self.sh_type}) and shtype_to ({sh_type})" +
                                  " should be SHType and different each other.")  
        
        if sh_type == SHType.REAL:
            res = res.real
        return SHVec(res, self.cod_type, sh_type)
    
    def normsq_levels(self, sum_LP: Optional[bool] = False) -> np.ndarray:
        """
        return[*c, level]
        """
        if self.cod_type == CodType.SCALAR:
            shape_res0 = self.shape_chan + (self.level,)
        else:
            shape_res0 = self.shape_chan + (self.level, int(self.cod_type))
        norm0 = np.zeros(shape_res0, dtype=self.coeff.dtype)

        for l in range(self.level):
            subarr = self.slice_l(l)
            norm0[..., l, :] = np.sum(subarr.conj()*subarr, axis=self.axis_N)
        
        if sum_LP and int(self.cod_type) > int(CodType.SCALAR):
            idx_sc = cidx_scalar(self.cod_type)
            idx_lp = cidx_lpolar(self.cod_type)
            res_sc = norm0[..., idx_sc]
            res_lpsum = norm0[..., idx_lp].sum(-1)[..., None]
            norm0 = np.concatenate([res_sc[..., 0:1], res_lpsum, res_sc[..., 1:2]], axis=-1)
        return norm0

    def to_TPmat(self, level_to: int, cod_type: Optional[CodType] = CodType.SCALAR) -> SHMat:
        """ SH vector to triple product matrix """
        if level_to*2-1 > self.level:
            raise ValueError("For unbiased result, `level_to*2-1 <= level_from` should be satisfied, "
                            f"but {self.level=} and {level_to=} is given.")
        cod_type = CodType(cod_type)
        if self.cod_type != CodType.SCALAR:
            raise ValueError(f"{self.cod_type=} should be `CodType.SCALAR`.")
        if self.sh_type == SHType.REAL:
            self_comp = self.to_shtype(SHType.COMP)
        else:
            self_comp = self
        
        N_to = level2num(level_to, DomType.UNI)
        shape_res = self.shape_chan + (N_to, N_to)
        if cod_type != CodType.SCALAR:
            shape_res += (int(cod_type), int(cod_type))
        res_mat = np.zeros(shape_res, dtype=np.complex128)
        
        res_s0 = _SHVec_to_TPmat(self_comp, level_to, spin=0)
        if cod_type == CodType.SCALAR:
            res_mat[:] = res_s0
        else:
            cidx = cidx_scalar(cod_type, out_type=tuple) # NOTE we should use `out_type=tuple` here.
            res_mat[..., cidx, cidx] = res_s0[..., None]
        del res_s0

        if cod_type != CodType.SCALAR:
            res_s2 = _SHVec_to_TPmat(self_comp, level_to, spin=2)
            cidx = cidx_lpolar(cod_type)
            res_mat[..., cidx, cidx] = comp2mat(res_s2)
            del res_s2

        return SHMat(res_mat, DomType.UNI, cod_type, SHType.COMP).to_shtype(self.sh_type)
    
    def wrt_equirect(self,
                       h:   int,
                       w:   int
                      ) ->  np.ndarray: # [h, w, *c, p]
        # from .grid import SphereGrid, StokesField
        tpFF = polarsh.grid.SphereFrameField.from_equirect(h, w)
        if self.cod_type == CodType.SCALAR:
            return polarsh.grid.ScalarField.from_SHCoeff(self, tpFF.SphGrid).fval
        else:
            return polarsh.grid.StokesField.from_SHCoeff(self, tpFF).Stk

    def wrt_persp(self,
                  h:         int,
                  w:         int,
                  fov_x_deg: float,
                  to_world:  ArrayLike = np.eye(3) # [3, 3]
                 ) ->       np.ndarray: # [h, w, *c, p]
        """
        to_world: transform which maps an identity camera frame to the target frame, w.r.t. world coordinates
                = coordinate conversion matrix from the camera coordinates to world coordinates for a fixed point
                = [x_camera | y_camera | z_camera] where each local axes are written w.r.t. world coordinates
        """
        sphFF = polarsh.grid.SphereFrameField.from_persp(h, w, fov_x_deg, to_world)
        if self.cod_type == CodType.SCALAR:
            return polarsh.grid.ScalarField.from_SHCoeff(self, sphFF.SphGrid).fval
        else:
            return polarsh.grid.StokesField.from_SHCoeff(self, sphFF).Stk

    def tabulate(self, ch_show: Optional[int] = None, level_show_from: Optional[int] = 0
                ) -> str:
        coeff_mat = _get_channel(self.coeff, self.ndim_chan, ch_show)
        N0 = level2num(level_show_from, DomType.UNI)
        lm = level2lms(self.level, DomType.UNI)[N0:, :]
        if self.cod_type == CodType.SCALAR:
            headers = ["l,m", "value"]
            coeff_mat = coeff_mat[N0:, None]
        else:
            headers = ["l,m"] + [f"p={p}" for p in np.arange(self.cod_type)]
            coeff_mat = coeff_mat[N0:, :]
        table = []
        
        for i, row in enumerate(coeff_mat):
            if self.sh_type == SHType.COMP:
                row = [format_complex(x, " %.2e") for x in row]
            table.append([row2str_short(lm[i,:]), *row])
        
        res_str = tabulate(table, headers=headers, floatfmt=".2e")
        n_bar = len(res_str.split('\n')[1])
        if self.cod_type == CodType.SCALAR:
            res_title = f"SHVec.coeff[{ch_show},:]\n"
        else:
            res_title = f"SHVec.coeff[{ch_show},:,:]\n"
        res_str = f"{res_title}{'-'*n_bar}\n{res_str}"
        return res_str
    
    def matshow(self,
                ch_show: Optional[int] = None,
                level_show: Optional[int] = None,
                long: Optional[bool] = False,
                figsize: Optional[Tuple] = None,
                title:    Optional[str] = None,
                cmap: Optional[Colormap] = pltcmap_diverge,
                norm: Optional[Normalize] = CenteredNorm(),
                colorbar: Optional[bool] = True,
                fig: Optional[Figure] = None
                ) -> plt.figure:
        '''
        NOTE I am not sure which visualization is best

        Current version only supports 1-ndim channels such as RGB.
        CAUTION: `dom_type == DomType.UNI` is considered as a coefficient matrix of [..., DomType.UNI, DomType.UNI, ...].
                for coefficient 'vectors', see `SHvecshow` function.
        '''
        coeff_show = _get_channel(self.cut(level_show).coeff, self.ndim_chan, ch_show)
        if self.cod_type == CodType.SCALAR:
            coeff_show = coeff_show[..., None]
        
        if long == False or self.cod_type == CodType.SCALAR:
            if fig is None:
                fig = plt.figure(figsize=figsize)
            elif figsize is not None:
                raise ValueError(f"One of arguments `{figsize=}` and `{fig=}` must be `None`.")
            ax = fig.add_subplot(1,1,1)
            im = ax.matshow(coeff_show, cmap=cmap, norm=norm)
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_frame_on(True)
            if colorbar:
                plt.colorbar(im)
            if not title is None:
                fig.suptitle(title)
        else:
            fig = matmatshow(np.expand_dims(coeff_show, (-3, -1)),
                             figsize=figsize, title=title, cmap=cmap, norm=norm, colorbar=colorbar, fig=fig)
        return fig
        
    
    def save(self, file: Union[str, Path], level_out: Optional[int] = None, cod_type_out: Optional[CodType] = None):
        """
        `.npy`: [channel, N(level), cod_type] also for `CodType.SCALAR`
        `.npz`
        `.float`: [channel, N(level), cod_type] -> [N(level), cod_type, channel] for lighting
        Parameters
            file: str | path with extension `.npy` or `.float`
        Currently `.float` only supports single-axis channel, SHType.REAL
        """
        file = Path(file)
        ext = file.suffix
        self_cut = self.cut(level_out, cod_type_out)
        match ext:
            case ".npy":
                if self.cod_type == CodType.SCALAR:
                    coeff = self_cut.coeff[..., None]
                else:
                    coeff = self_cut.coeff
                np.save(file, coeff)
            case ".npz":
                np.savez(file, cod_type = str(self_cut.cod_type),
                               sh_type  = str(self_cut.sh_type),
                               coeff    = self_cut.coeff)
            case ".float":
                assert self_cut.coeff.ndim in [3, 4]
                assert self_cut.sh_type == SHType.REAL
                assert self_cut.coeff.dtype in [np.float32, np.float64]
                
                mat = self_cut.coeff
                mat = np.moveaxis(mat.astype(np.float32), 0, -1)
                # print(mat[:5,:,:])
                with open(file, 'wb') as f:
                    f.write(mat.tobytes())
            case _:
                raise ValueError(f"The method only supports `.npy` and `.float`. Given file name is: {file}")

class _SHVec_channel_indexer:
    """
    Examples
    -----
    shv.coeff.shape ==                 (3, 4, 5, 6, N=25, p=4)
    shv.chan[0].coeff.shape ==            (4, 5, 6, N=25, p=4)
    shv.chan[0, ...].coeff.shape ==       (4, 5, 6, N=25, p=4)
    shv.chan[..., 0].coeff.shape ==    (3, 4, 5,    N=25, p=4)
    shv.chan[0, ..., 0].coeff.shape ==    (4, 5,    N=25, p=4)
    """
    def __init__(self, obj: SHVec):
        self.obj = obj
    def __getitem__(self, key) -> SHVec:
        key_Np = tuple(slice(None, None, None) for _ in range(-self.obj.axis_N))
        if not isinstance(key, tuple):
            key = (key, )
        if (len(key) < len(self.obj.shape_chan)) and not any(... is i for i in key):
            key_res = key + (...,) + key_Np
        else:
            key_res = key + key_Np
        return SHVec(self.obj.coeff[key_res], self.obj.cod_type, self.obj.sh_type)
    
    def rgb2gray(self) -> SHVec:
        if self.obj.shape_chan != (3,):
            raise ValueError(f"Invalid channel shape for the `SHVec` instance: {self.obj.shape_chan}")
        return SHVec(rgb2gray(self.obj.coeff, 0), self.obj.cod_type, self.obj.sh_type)

########################################
### class SHMat
########################################
class SHMat(SHCoeff):
    """
    NOTE
    * DomType.UNI indicates unimat[*, *, No, Ni, (po, pi)]
    """
    def __init__(self,
                 coeff:    ArrayLike, # [*, N*, p*]
                 dom_type: DomType,   # UNI: N*=No, Ni | BI,ISOBI,ISOBI2: N*=N
                 cod_type: CodType,   # SCALAR: p*=.   | POLARp: p*=po, pi
                 sh_type:  SHType) -> SHMat:
        # ---------- Assertion ----------
        cod_type = CodType(cod_type)
        dom_type = DomType(dom_type)
        sh_type = SHType(sh_type)

        if dom_type == DomType.UNI:
            self.level, self.axis_N = assert_unimat(coeff, cod_type)
            self.axis_No, self.axis_Ni = self.axis_N - 1, self.axis_N
        else:
            self.level, self.axis_N = assert_dctype(coeff, dom_type, cod_type)
            self.axis_No, self.axis_Ni = self.axis_N, self.axis_N
        assert (self.axis_No <= self.axis_Ni) and (self.axis_Ni < 0)
        self.coeff = assert_shtype(coeff, sh_type)
        # ---------- Attributes ----------
        self.N = coeff.shape[self.axis_N]
        self.dom_type = dom_type
        self.cod_type = cod_type
        self.sh_type = sh_type
        # ---------- Protected ----------
        if self.dom_type != DomType.UNI:
            self.shape_chan = self.coeff.shape[:self.axis_N]
        else:
            self.shape_chan = self.coeff.shape[:self.axis_N-1]
        self.ndim_chan = len(self.shape_chan)

    @classmethod
    def from_npy_file(cls, filename_npy: Union[str, Path], dom_type: DomType, sh_type: SHType) -> SHMat:
        """
        This method do not support CodType.SCALAR
        """
        filename_npy = str(filename_npy)
        if filename_npy[-4:] != ".npy":
            raise ValueError(f"Unsupported file extension: {filename_npy[-4:]}")
        coeff = np.load(filename_npy)
        if coeff.shape[-1] < CodType.POLAR2 or coeff.shape[-1] > CodType.POLAR4:
            raise ValueError(f"The method only supports CodType.POLAR2-4, but array shape {coeff.shape} is given.")
        return cls(coeff, dom_type, coeff.shape[-1], sh_type)
    
    @classmethod
    def from_npz_file(cls, filename_npz: Union[str, Path]) -> SHMat:
        filename_npz = Path(filename_npz)
        assert filename_npz.suffix == ".npz", f"Invalid file extension for {filename_npz}"
        npz = np.load(filename_npz)
        return cls(npz['coeff'], str(npz['dom_type']), str(npz['cod_type']), str(npz['sh_type']))
    
    @classmethod
    def from_rotation(cls, rotation: ArrayLike, level: int,
                      cod_type: CodType, sh_type: SHType) -> SHMat:
        '''
        TODO: generalize for SHType.REAL (real SH for scalar components)
        '''
        # ---------- Parameters ----------
        R = quat.array.from_axis_angle(rotation)
        cod_type = CodType(cod_type)
        sh_type = SHType(sh_type)

        wigner = spherical.Wigner(level)
        D = wigner.D(R).conj() # from difference of convention
        N_UNI = level2num(level, DomType.UNI)
        
        
        if cod_type == CodType.SCALAR:
            res = np.zeros((N_UNI, N_UNI), dtype=np.complex128)
        else: # cod_type in [CodType.POLAR2, CodType.POLAR3, CodType.POLAR4]
            # NOTE IMPORTANT: cidx should be tuple, and pidx should be slice here.
            cidx = cidx_scalar(cod_type, out_type=tuple)
            pidx = cidx_lpolar(cod_type)

            res = np.zeros((N_UNI, N_UNI, cod_type, cod_type), dtype=np.complex128)
        
        iD1 = 0
        for l in range(level):
            im1 = level2num(l, DomType.UNI)
            im2 = level2num(l+1, DomType.UNI)
            nm = im2 - im1
            iD2 = iD1 + nm**2
            
            D_mat_curr = D[iD1:iD2].reshape(nm, nm)

            if cod_type == CodType.SCALAR:
                res[im1:im2, im1:im2] = D_mat_curr
            else:
                ## [nm, nm, |cidx|] due to tuple indexing
                res[im1:im2, im1:im2, cidx, cidx] = D_mat_curr[..., None]
                ## [nm, nm, 2, 2] due to slicing
                res[im1:im2, im1:im2, pidx, pidx] = comp2mat(D_mat_curr)
            iD1 = iD2
        
        shm_comp = cls(res, DomType.UNI, cod_type, SHType.COMP)
        if sh_type == SHType.COMP:
            return shm_comp
        else: # SHType.REAL
            return shm_comp.to_shtype(sh_type)

    @classmethod
    def zeros_like(cls, shm: SHMat) -> SHMat:
        return cls(np.zeros_like(shm.coeff), shm.dom_type, shm.cod_type, shm.sh_type)

    def __repr__(self) -> str:
        if self.dom_type == DomType.UNI:
            str_N = f"{self.N}, {self.N}"
        else:
            str_N = str(self.N)
        if self.cod_type == CodType.SCALAR:
            str_polar = "(SCALAR)"
        else:
            str_polar = f"| p: {int(self.cod_type)}, {int(self.cod_type)}"
        str_shapec = str(self.shape_chan)[1:-1]
        res = f"{self.__class__.__name__}[c: {str_shapec} | N: {str_N} {str_polar}][\n"
        res += f"  level = {self.level},\n"
        res += f"  dom_type = {self.dom_type!r},\n"
        res += f"  cod_type = {self.cod_type!r},\n"
        res += f"  sh_type = {self.sh_type!r},\n"
        res += f"  coeff.shape = {self.coeff.shape},\n"
        res += f"  coeff.dtype = {self.coeff.dtype},\n"
        res += "]"
        return res
    
    # ---------- Trivial operations ---------
    def __add__(self, x) -> SHMat:
        if isinstance(x, SHMat):
            assert self.dom_type == x.dom_type
            assert self.cod_type == x.cod_type
            assert self.sh_type == x.sh_type
            return SHMat(self.coeff + x.coeff, self.dom_type, self.cod_type, self.sh_type)
        else:
            raise TypeError(f"Not supported data type: {type(x)}")
    
    def __sub__(self, x) -> SHMat:
        if isinstance(x, SHMat):
            assert self.dom_type == x.dom_type
            assert self.cod_type == x.cod_type
            assert self.sh_type == x.sh_type
            return SHMat(self.coeff - x.coeff, self.dom_type, self.cod_type, self.sh_type)
        else:
            raise TypeError(f"Not supported data type: {type(x)}")

    def __mul__(self, x) -> SHMat:
        if np.isscalar(x):
            # NOTE even if x is complex and self.sh_type is REAL,
            #      the constructor will assert it.
            return SHMat(x*self.coeff, self.dom_type, self.cod_type, self.sh_type)
        else:
            raise TypeError(f"Not supported data type: {type(x)}")
    def __rmul__(self, x) -> SHMat:
        if np.isscalar(x):
            return self.__mul__(x)
        else:
            raise TypeError(f"Not supported data type: {type(x)}")
    def __truediv__(self, x) -> SHMat:
        if np.isscalar(x):
            return self.__mul__(1/x)
        else:
            raise TypeError(f"Not supported data type: {type(x)}")
    
    # ---------- Matrix multiplication ---------
    def __matmul__(self, x: Union[SHVec, SHMat, SHTransform]) -> SHMat:
        level = self.level
        cod_type = self.cod_type
        sh_type = self.sh_type
        
        if isinstance(x, SHCoeff):
            assert self.level == x.level
            assert self.cod_type == x.cod_type
            assert self.sh_type == x.sh_type
            self_unimat = self.to_domtype(DomType.UNI).coeff
            if isinstance(x, SHVec):
                # ---------- self @ SHVec ----------
                if cod_type == CodType.SCALAR:
                    res_mat = matmul_vec1d(self_unimat, x.coeff)
                else:
                    res_mat = np.einsum('...ijkl,...jl->...ik', self_unimat, x.coeff)
                return SHVec(res_mat, cod_type, sh_type)
            elif isinstance(x, SHMat):
                # ---------- self @ SHMat ----------
                x_unimat = x.to_domtype(DomType.UNI).coeff
                if cod_type == CodType.SCALAR:
                    res_mat = self_unimat @ x.coeff
                else:
                    res_mat = np.einsum('...ijpq,...jkqr->...ikpr', self_unimat, x_unimat)
                return SHMat(res_mat, DomType.UNI, cod_type, sh_type)
            else:
                raise TypeError(f"Not supported type {type(x)} for x.")
        elif isinstance(x, SHTransform):
            if isinstance(x, SHReflection):
                # ---------- self @ SHReflection ----------
                '''
                Return coeff @ (coeff mat. of reflection operator)
                TODO: Handle weired numpy array shape
                '''
                if self.dom_type not in [DomType.ISOBI2, DomType.BI]:
                    raise NotImplementedError()
                assert self.coeff.dtype == np.float64
                li,lo,mi,mo = level2lms(level, self.dom_type, unstack=True)
                
                phase_l = (-1) ** (li % 2)
                phase_lm = (-1) ** ((li + mi) % 2)
                idx_org2res = lms2idx(np.stack([li,lo,-mi,mo], -1), self.dom_type)
                res = np.zeros_like(self.coeff)
                
                # s_s = cidx_scalar(cod_type)
                # s_v = cidx_lpolar(cod_type)
                if cod_type == CodType.SCALAR:
                    res[...] = phase_lm * self.coeff
                else:
                    assert cod_type == CodType.POLAR3
                    res[..., 0] = np.expand_dims(phase_lm, -1) * self.coeff[..., :, :, 0]
                    res[..., 1] = np.expand_dims(phase_l, -1) * np.moveaxis(self.coeff[..., idx_org2res, :, 1], 0, -2)
                    res[..., 2] = -np.expand_dims(phase_l, -1) * np.moveaxis(self.coeff[..., idx_org2res, :, 2], 0, -2)
                return SHMat(res, self.dom_type, self.cod_type, self.sh_type)
            else:
                raise TypeError(f"Not supported type {type(x)} for x.")
        else:
                raise TypeError(f"Not supported type {type(x)} for x.")
    
    def allclose(self, shm: SHMat, rtol: Optional[float]=1e-05, atol: Optional[float]=1e-08) -> bool:
        if not isinstance(shm, SHMat):
            raise TypeError(f"Invalid type: {type(shm)=}. The type must be `SHMat`.")
        if self.dom_type != shm.dom_type:
            raise ValueError(f"Attributes `dom_type` for `self` and `shm` must be equal, but currently:\n"
                             f"{self.dom_type = }, {shm.dom_type = }")
        if self.cod_type != shm.cod_type:
            raise ValueError(f"Attributes `cod_type` for `self` and `shm` must be equal, but currently:\n"
                             f"{self.cod_type = }, {shm.cod_type = }")
        if self.sh_type != shm.sh_type:
            raise ValueError(f"Attributes `sh_type` for `self` and `shm` must be equal, but currently:\n"
                             f"{self.sh_type = }, {shm.sh_type = }")
        return np.allclose(self.coeff, shm.coeff, rtol=rtol, atol=atol)
    
    @property
    def chan(self) -> _SHMat_channel_indexer:
        return _SHMat_channel_indexer(self)
    
    def copy(self) -> SHMat:
        return SHMat(self.coeff.copy(), self.dom_type, self.cod_type, self.sh_type)
    
    def at_lms(self, *lms: Sequence[ArrayLike]) -> np.ndarray:
        # Order: li, lo, mi, mo
        if self.dom_type == DomType.UNI:
            li, lo, mi, mo = lms
            idxi = lms2idx([li, mi], DomType.UNI)
            idxo = lms2idx([lo, mo], DomType.UNI)
            if self.cod_type == CodType.SCALAR:
                return self.coeff[..., idxo, idxi]
            else:
                return self.coeff[..., idxo, idxi, :, :]
        else:
            idx = lms2idx(lms, DomType.UNI)
            if self.cod_type == CodType.SCALAR:
                return self.coeff[..., idx]
            else:
                return self.coeff[..., idx, :, :]
        
    def cut(self,
            level: Union[int, SHCoeff, None] = None,
            cod_type: Optional[CodType] = None
           ) ->    SHMat:
        # ---------- Cut level ----------
        if level is None:
            coeff_to = self.coeff
        else:
            if isinstance(level, SHCoeff):
                level = level.level
            if level > self.level:
                raise ValueError(f"The parameter {level=} cannot be larger than {self.level=}.")
            N = level2num(level, self.dom_type)
            if self.cod_type == CodType.SCALAR:
                if self.dom_type == DomType.UNI:
                    coeff_to = self.coeff[..., :N, :N]
                else: # dom_type in [DomType.BI, DomType.ISOBI, DomType.ISOBI2]
                    coeff_to = self.coeff[..., :N]
            else: # cod_type == CPOLARp
                if self.dom_type == DomType.UNI:
                    coeff_to = self.coeff[..., :N, :N, :, :]
                else: # dom_type in [DomType.BI, DomType.ISOBI, DomType.ISOBI2]
                    coeff_to = self.coeff[..., :N, :, :]

        # ---------- Cut codomain type ---------- 
        if cod_type is None:
            cod_type_final = self.cod_type
        else:
            cod_type_final = cod_type
            cidx = cidx_to(self.cod_type, cod_type)
        if cod_type is not None and self.cod_type != CodType.SCALAR:
            coeff_to = coeff_to[..., cidx, cidx]

        return SHMat(coeff_to, self.dom_type, cod_type_final, self.sh_type)
    
    def to_shtype(self,
                  sh_type: SHType
                 ) ->      SHMat: # [*c, N] | [*c, N, p] | [*c, N, po, pi]
        '''
        Refer to `test_Sotkes_s2SH.ipynb`.
        '''
        # ---------- Initial ----------
        if self.cod_type != CodType.SCALAR:
            idx_sc = cidx_scalar(self.cod_type)
            idx_lp = cidx_lpolar(self.cod_type)
        rt2 = math.sqrt(2)
        res = np.zeros_like(self.coeff, dtype=np.complex128)
        
        assert isinstance(self.dom_type, DomType)
        sh_type = SHType(sh_type)
        sh_type = SHType(sh_type)
        match self.sh_type, sh_type: # (from, to)
            case SHType.COMP, SHType.REAL:
                flag_C2R, flag_R2C = True, False
            case SHType.REAL, SHType.COMP:
                flag_C2R, flag_R2C = False, True
            case _: # fromn == to
                return self
        
        # ---------- ISOBI ----------
        if self.dom_type == DomType.ISOBI:
            raise NotImplementedError()
        # ---------- UNI mat & ISOBI2 & BI ----------
        else:
            if self.dom_type == DomType.UNI:
                l, m = level2lms(self.level, self.dom_type, unstack=True)
                mi = m
                mo = mi[:, None]
                idx_negm = lms2idx(np.stack([l, -m], axis=-1), self.dom_type)
                pax_org = (-4, -3)
                pax_view = (-2, -1)
            else: # self.dom_type in [DomType.ISOBI2, DomType.BI]
                li, lo, mi, mo = level2lms(self.level, self.dom_type, unstack=True)
                
                idx_negmi  = lms2idx(np.stack([li, lo, -mi,  mo], axis=-1), self.dom_type)
                idx_negmo  = lms2idx(np.stack([li, lo,  mi, -mo], axis=-1), self.dom_type)
                pax_org = -3
                pax_view = -1
            phase_i = (-1) ** (mi % 2)
            phase_o = (-1) ** (mo % 2)
            ## For mi
            if self.cod_type == CodType.SCALAR:
                coeff_sc = self.coeff
                res_sc = res
            else:
                # coeff_sc | res_sc [*c, po, pi, N*]
                coeff_sc = np.moveaxis(self.coeff[..., :, idx_sc].view(), pax_org, pax_view)
                res_sc = np.moveaxis(res[..., :, idx_sc].view(), pax_org, pax_view)
                res[..., idx_lp] = self.coeff[..., idx_lp]
            
            if self.dom_type == DomType.UNI:
                coeff_negmi = coeff_sc[..., :, idx_negm]
            else: # self.dom_type in [DomType.ISOBI2, DomType.BI]
                coeff_negmi = coeff_sc[..., idx_negmi]

            # NOTE conjugate for mi!! See `test_Stokes_s2SH.ipynb`
            if flag_R2C:
                res_sc[:] = np.select([mi > 0, mi < 0],
                                    [(coeff_sc + 1j*coeff_negmi)/rt2,
                                    phase_i*(-1j*coeff_sc + coeff_negmi)/rt2],
                                    coeff_sc)
            elif flag_C2R:
                res_sc[:] = np.select([mi > 0, mi < 0],
                                    [(coeff_sc + phase_i*coeff_negmi)/rt2,
                                    (-phase_i*coeff_sc + coeff_negmi)*-1j/rt2],
                                    coeff_sc)

            ## For mo (same as SHVec)
            if self.cod_type == CodType.SCALAR:
                    pass
            else:
                # res_sc [*c, po, pi, N*]
                res_sc = np.moveaxis(res[..., idx_sc, :].view(), pax_org, pax_view)
            if self.dom_type == DomType.UNI:
                res_negmo = res_sc[..., idx_negm, :]
            else:
                res_negmo = res_sc[..., idx_negmo]
            
            if flag_R2C:
                res_sc[:] = np.select([mo > 0, mo < 0],
                                    [(res_sc - 1j*res_negmo)/rt2,
                                    phase_o*(1j*res_sc + res_negmo)/rt2],
                                    res_sc)
            elif flag_C2R:
                res_sc[:] = np.select([mo > 0, mo < 0],
                                    [(res_sc + phase_o*res_negmo)/rt2,
                                    (-phase_o*res_sc + res_negmo)*1j/rt2],
                                    res_sc)
        # ---------- Final ----------  
        if sh_type == SHType.REAL:
            res = res.real
        return SHMat(res, self.dom_type, self.cod_type, sh_type)

    def to_domtype(self, dom_type) -> SHMat:
        """
        If `dom_type` is same as itself, data copy does not occur.
        """
        dom_type = DomType(dom_type)
        if self.dom_type == dom_type:
            return self
        
        match self.dom_type: # from
            case DomType.BI:
                mat_bi = self.coeff
            case DomType.UNI:
                mat_bi = _mat2DBI(self.coeff, self.cod_type)
            case _: # ISOBI, ISOBI2
                mat_bi = _toDBI(self.coeff, self.dom_type, self.cod_type)
        
        match dom_type: # to
            case DomType.BI:
                mat_res = mat_bi
            case DomType.UNI:
                mat_res = _DBI2mat(mat_bi, self.cod_type)
            case _: # ISOBI, ISOBI2
                mat_res = _fromDBI(mat_bi, dom_type, self.cod_type)
        
        return SHMat(mat_res, dom_type, self.cod_type, self.sh_type)

    def adjoint(self) -> SHMat:
        """ Get the Hermitian adjoint (transpose followed by complex conjutation) """
        self_unimat = self.to_domtype(DomType.UNI)
        if self.cod_type == CodType.SCALAR:
            esrule = '...ij->...ji'
        else:
            esrule = '...ijkl->...jilk'
        adj_unimat = np.einsum(esrule, self_unimat.coeff.conj())
        return SHMat(adj_unimat, DomType.UNI, self.cod_type, self.sh_type)
    
    def to_SHConv(self, weighted: Optional[bool] = False) -> SHConv:
        return SHConv.from_SHMat(self, weighted=weighted)

    def apply_rotation(self, rotation: Union[ArrayLike, SHMat]) -> SHMat:
        """
        params:
            rotation[3]: ArrayLike, rotation transform, used for scalars
        """
        self_comp = self.to_shtype(SHType.COMP)
        if isinstance(rotation, SHMat):
            shm_R_comp = rotation
        else:
            shm_R_comp = SHMat.from_rotation(rotation, self.level, self.cod_type, SHType.COMP)
        res_comp = shm_R_comp @ self_comp @ shm_R_comp.adjoint()
        return res_comp.to_shtype(self.sh_type)
    
    def set_normal(self, normal: ArrayLike) -> SHMat:
        '''
        params:
            self
            normal[3]: ArrayLike, rotation transform, used for scalars
        return:
            shm_rot:   SHMat with dom_type == DomType.
        NOTE Be careful the output always DomType.BI, as `SHcoeff_bi_apply_rotation` does.
        '''
        if not self.dom_type in [DomType.ISOBI and DomType.ISOBI2]:
            raise ValueError(f"The method `set_normal` only supports `DomType.ISOBI` and `Domtype.ISOBI2`, but currently {self.dom_type=}.")
        normal = np.asarray(normal).squeeze()
        if normal.shape != (3,):
            raise ValueError(f"The shape of `normal` should be (3,), but given squeezed shape is: {normal.shape=}.")
        rotation = normal2rotvec(normal)
        return self.apply_rotation(rotation)
    
    def tabulate(self, ch_show: Optional[int] = None, level_show_from: Optional[int] = 0) -> str:
        coeff_mat = _get_channel(self.coeff, self.ndim_chan, ch_show)
        N0 = level2num(level_show_from, self.dom_type)
        N_show = self.N - N0
        
        label_lms = {'UNI': "l,m", 'ISOBI': "li,lo,m", 'ISOBI2': "li,lo,mi,mo", 'BI': "li,lo,mi,mo"}[str(self.dom_type)]
        if self.cod_type == CodType.SCALAR:
            lm_ = level2lms(self.level, self.dom_type)[N0:, :] # [N_show, 2|3|4]
            if self.dom_type == DomType.UNI:
                headers = ["l,m"] + [row2str_short(row) for row in lm_]
            else:
                headers = [label_lms, "value"]
        else:
            if self.dom_type == DomType.UNI:
                l, m = np.expand_dims(level2lms(self.level, DomType.UNI, unstack=True), -1)[:, N0:, :]
                p = np.arange(self.cod_type)
                lm_ = np.stack(np.broadcast_arrays(l, m, p), -1).reshape(-1, 3)
                headers = ["l,m,p"] + [row2str_short(row) for row in lm_]
            else:
                lm_ = level2lms(self.level, self.dom_type)[N0:, :]
                pi = np.arange(self.cod_type)
                po = pi[:,None]
                pp = np.stack(np.broadcast_arrays(pi, po), -1).reshape(-1, 2)
                headers = [f"{label_lms}\\pi,po"] + [row2str_short(row) for row in pp]
        
        table = []
        
        # Change shape of `coeff_mat` =====
        if self.dom_type == DomType.UNI:
            coeff_mat = coeff_mat[N0:, N0:]
            if self.cod_type != CodType.SCALAR:
                # [No_show, Ni_show, po, pi] -> [(No_show,po), (Ni_show,pi)]
                coeff_mat = np.moveaxis(coeff_mat, 2, 1).reshape(N_show*int(self.cod_type), N_show*int(self.cod_type))
        else:
            coeff_mat = coeff_mat[N0:]
            if self.cod_type == CodType.SCALAR: 
                # [N_show, po, pi] -> [N_show, 1]
                coeff_mat = coeff_mat[:, None]
            else:
                # [N_show, po, pi] -> [N_show, (pi,po)]
                coeff_mat = np.moveaxis(coeff_mat, 2, 1).reshape(N_show, int(self.cod_type)**2)
        assert coeff_mat.ndim == 2, f"Something wrong!: {coeff_mat.shape = }\n{self = }"
        # ================= Fix `coeff_mat`

        for i, row in enumerate(coeff_mat):
            if self.sh_type == SHType.COMP:
                row = [format_complex(x, " %.2e") for x in row]
            table.append([row2str_short(lm_[i,:]), *row])
        
        label_colon = ",".join([":"]*(self.coeff.ndim - self.ndim_chan))
        res_str = tabulate(table, headers=headers, floatfmt=".2e")
        n_bar = len(res_str.split('\n')[1])
        return f"SHMat.coeff[{ch_show},{label_colon}]\n" \
               f"{'-'*n_bar}\n{res_str}"
    
    def matshow(self,
                ch_show:    Optional[int] = None,
                level_show: Optional[int] = None,
                figsize:    Union[None, Tuple[float, float]] = None,
                title:      Optional[str] = None,
                cmap:       Optional[Colormap] = pltcmap_diverge,
                norm:       Optional[Normalize] = CenteredNorm(),
                colorbar:   Optional[bool] = True,
                fig:        Union[None, Figure] = None
               ) ->         Figure:
        '''
        NOTE I am not sure which visualization is best

        Current version only supports 1-ndim channels such as RGB.
        CAUTION: `dom_type == DomType.UNI` is considered as a coefficient matrix of [..., DomType.UNI, DomType.UNI, ...].
                for coefficient 'vectors', see `SHvecshow` function.
        '''
        # assert level_show is None or self.dom_type != DomType.UNI, NotImplementedError()
        # NOTE I don't remember why I needed the above line.
        shm_cut_UNI = self.cut(level_show).to_domtype(DomType.UNI)
        coeff_mat = _get_channel(shm_cut_UNI.coeff, self.ndim_chan, ch_show)
        return matmatshow(coeff_mat,
                          figsize=figsize, title=title, cmap=cmap, norm=norm, colorbar=colorbar, fig=fig)

    def save(self, filename_npz: Union[str, Path], level_out: Optional[int] = None, cod_type_out: Optional[CodType] = None):
        filename_npz = Path(filename_npz)
        assert filename_npz.suffix == ".npz", f"Invalid file extension: {filename_npz}"
        shm_out = self.cut(level_out, cod_type_out)
        np.savez(filename_npz,
                 dom_type = str(shm_out.dom_type),
                 cod_type = str(shm_out.cod_type),
                 sh_type  = str(shm_out.sh_type),
                 coeff    = shm_out.coeff)

class _SHMat_channel_indexer:
    def __init__(self, obj: SHMat):
        self.obj = obj
    def __getitem__(self, key) -> SHMat:
        key_Np = tuple(slice(None, None, None) for _ in range(-self.obj.axis_No))
        if not isinstance(key, tuple):
            key = (key, )
        if (len(key) < len(self.obj.shape_chan)) and not any(... is i for i in key):
            key_res = key + (...,) + key_Np
        else:
            key_res = key + key_Np
        return SHMat(self.obj.coeff[key_res], self.obj.dom_type, self.obj.cod_type, self.obj.sh_type)
    
    def rgb2gray(self) -> SHMat:
        if self.obj.shape_chan != (3,):
            raise ValueError(f"Invalid channel shape for the `SHMat` instance: {self.obj.shape_chan}")
        return SHMat(rgb2gray(self.obj.coeff, 0), self.obj.dom_type, self.obj.cod_type, self.obj.sh_type)
    
########################################
### SHConv
########################################
class SHConv(SHCoeff):

    '''
    Real SH for scalars
    
    Naming convention:
        "s2v" in codes indicates "spin 0-to-2" in the paper.
        a: isomorphic part of R^2x2 -> (Ciso, Cconj)
        b: conjugation part of R^2x2 -> (Ciso, Cconj)
    
    Attributes:
    -----------
    cod_type: CodType, shapes of other attributes depends on this `cod_type`
                CodType.SCALAR | CodType.POLAR2 | CodType.POLAR3 | CodType.POLAR4
    coeff_s2s:  [*c, level]    | None           | [*c, level]    | [*c, level, 2, 2]
    coeff_s2v:  None           | None           | [*c, level] c  | [*c, level, 2] c
    coeff_v2s:  None           | None           | [*c, level] c  | [*c, level, 2] c
    coeff_v2va: None           | [*c, level] c  | [*c, level] c  | [*c, level] c
    coeff_v2vb: None           | [*c, level] c  | [*c, level] c  | [*c, level] c
    '''
    ndim_p_dict = {'s2s':  [0, 0, 0, 2],
                   's2v':  [0, 0, 0, 1],
                   'v2s':  [0, 0, 0, 1],
                   'v2va': [0, 0, 0, 0],
                   'v2vb': [0, 0, 0, 0]}
    shape_p_dict = {'s2s':  [(), (), (), (2, 2)],
                    's2v':  [(), (), (), (2,)],
                    'v2s':  [(), (), (), (2,)],
                    'v2va': [(), (), (), ()],
                    'v2vb': [(), (), (), ()]}
    
    def __init__(self,
                 cod_type:   CodType, # CodType.POLAR3     | CodType.POLAR4 (actually, more complicated)
                 coeff_s2s:  ArrayLike = None, # [*c, level] | [*c, level, 2, 2]
                 coeff_s2v:  ArrayLike = None, # [*c, level] | [*c, level, 2] complex
                 coeff_v2s:  ArrayLike = None, # [*c, level] | [*c, level, 2] complex
                 coeff_v2va: ArrayLike = None, # [*c, level] complex
                 coeff_v2vb: ArrayLike = None,  # [*c, level] complex
                 weighted: bool = False
                ) -> None:
        '''
        NOTE The `shape[-1] == level` rather than `level-2` even for linear polarization parts.
        '''
        self.cod_type = CodType(cod_type)
        self.sh_type = SHType.REAL
        if self.cod_type == CodType.SCALAR:
            self.level = coeff_s2s.shape[-1]
            self.shape_chan = coeff_s2s.shape[:-1]
        else:
            self.level = coeff_v2va.shape[-1]
            self.shape_chan = coeff_v2va.shape[:-1]
        self.ndim_chan = len(self.shape_chan)
        
        coeff_s2s = assert_coeff_s2s(coeff_s2s, self.level, self.shape_chan, self.cod_type)
        coeff_v2s = assert_coeff_sv(coeff_v2s, self.level, self.shape_chan, self.cod_type)
        coeff_s2v = assert_coeff_sv(coeff_s2v, self.level, self.shape_chan, self.cod_type)
        coeff_v2va, coeff_v2vb = assert_coeff_v2v(coeff_v2va, coeff_v2vb, self.level, self.shape_chan, self.cod_type)
        
        self.s2s = coeff_s2s
        self.s2v = coeff_s2v
        self.v2s = coeff_v2s
        self.v2va = coeff_v2va
        self.v2vb = coeff_v2vb
        assert isinstance(weighted, bool), f"Invalid argument type: {type(weighted)=}, where {weighted=}."
        self.weighted = weighted

        # ---------- Util attributes ----------
        l = np.arange(self.level)
        self.weight_l = np.sqrt(4*np.pi/(2*l+1)) # Constants!
        
        p_id = int(self.cod_type) - 1
        self.ndim_p_dict = {key: val[p_id] for key, val in SHConv.ndim_p_dict.items()}


    @classmethod
    def from_dict(cls, cod_type: CodType, dict_coeff: dict[str, ArrayLike], weighted: bool = False) -> SHConv:
        return cls(cod_type,
                   coeff_s2s  = dict_coeff['s2s'],
                   coeff_s2v  = dict_coeff['s2v'],
                   coeff_v2s  = dict_coeff['v2s'],
                   coeff_v2va = dict_coeff['v2va'],
                   coeff_v2vb = dict_coeff['v2vb'],
                   weighted   = weighted)
    
    @classmethod
    def from_coeff_Mueller(cls, coeff_mat: ArrayLike) -> SHConv:
        coeff_mat = np.asarray(coeff_mat)
        assert coeff_mat.shape[-2] == coeff_mat.shape[-1]
        cod_type = CodType(coeff_mat.shape[-1])

        idx_s = cidx_scalar(cod_type)
        idx_v = cidx_lpolar(cod_type)
        coeff_s2s = coeff_mat[..., idx_s, idx_s]
        coeff_s2v = vec2comp(coeff_mat[..., idx_v, idx_s], -2)
        coeff_v2s = vec2comp(coeff_mat[..., idx_s, idx_v], -1)
        coeff_v2va, coeff_v2vb = mat2comppair(coeff_mat[..., idx_v, idx_v])

        if cod_type == CodType.POLAR3:
            coeff_s2s = coeff_s2s.squeeze((-1, -2))
            coeff_s2v = coeff_s2v.squeeze(-1)
            coeff_v2s = coeff_v2s.squeeze(-1)
        return cls(cod_type, coeff_s2s, coeff_s2v, coeff_v2s, coeff_v2va, coeff_v2vb, weighted=True)


    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> SHConv:
        filename = Path(filename)
        assert filename.suffix == ".npy", f"Invalid file extension for {filename=}."

        with open(filename, 'rb') as f:
            weighted = bool(np.load(f))
            cod_type = CodType(np.load(f))
            coeff_s2s = np.load(f)
            coeff_s2v = np.load(f)
            coeff_v2s = np.load(f)
            coeff_v2va = np.load(f)
            coeff_v2vb = np.load(f)

        return cls(cod_type, coeff_s2s=coeff_s2s,
                             coeff_s2v=coeff_s2v,
                             coeff_v2s=coeff_v2s,
                             coeff_v2va=coeff_v2va,
                             coeff_v2vb=coeff_v2vb,
                             weighted=weighted)

    @classmethod
    def from_SHMat(cls, shm: SHMat, weighted: Optional[bool] = False) -> SHConv:
        assert shm.sh_type == SHType.REAL
        if shm.dom_type == DomType.UNI:
            shm = shm.to_domtype("BI")
        elif shm.dom_type == DomType.ISOBI:
            shm = shm.to_domtype("ISOBI2")
        assert shm.dom_type in [DomType.ISOBI2, DomType.BI]

        level = shm.level
        dom_type = shm.dom_type
        cod_type = shm.cod_type
        coeff_mat = shm.coeff

        shape_conv = shm.shape_chan + (level,)
        p_id = int(cod_type)-1
        exp_dict = {key: (None,) * val[p_id] for key,val in SHConv.ndim_p_dict.items()} # axes for `np.expand_dims(coeff_*2*, exp_dict['*2*'])`
        shape_dict = {key: shape_conv + (2,) * val[p_id] for key,val in SHConv.ndim_p_dict.items()}
        convc_s2s = np.zeros(shape_dict['s2s'], dtype=np.float64)      if cod_type != CodType.POLAR2 else None
        convc_s2v = np.zeros(shape_dict['s2v'], dtype=np.complex128)   if cod_type >= CodType.POLAR3 else None
        convc_v2s = np.zeros(shape_dict['v2s'], dtype=np.complex128)   if cod_type >= CodType.POLAR3 else None
        convc_v2va = np.zeros(shape_dict['v2va'], dtype=np.complex128) if cod_type >= CodType.POLAR2 else None
        convc_v2vb = np.zeros(shape_dict['v2vb'], dtype=np.complex128) if cod_type >= CodType.POLAR2 else None
        
        li_full, lo_full, mi_full, mo_full = level2lms(level, dom_type, unstack=True) # [N], for each
        mask_common = (li_full == lo_full) & (np.abs(mi_full) == np.abs(mo_full)) # [N] bool

        U_s2v_full, U_v2s_full = get_U(mi_full, mo_full) # [N] complex, for each

        for l in range(level):
            ## TODO weired numpy array shape. see moveaxis used here
            mask_l = mask_common & (l == li_full)
            mask_s2s = mask_l & (mi_full == mo_full)

            # ---------- [Scalar to scalar] ----------
            if cod_type == CodType.SCALAR:
                convc_s2s[..., l] = np.mean(coeff_mat[..., mask_s2s], -1)
            elif cod_type == CodType.POLAR3:
                convc_s2s[..., l] = np.mean(coeff_mat[..., mask_s2s, 0, 0], -1)
            elif cod_type == CodType.POLAR4:
                convc_s2s[..., l, :, :] = np.mean(coeff_mat[..., mask_s2s, ::3, ::3], -3)
            
            if cod_type >= CodType.POLAR3:
                # ---------- [Scalar to Stokes] ----------
                U_s2v_l = U_s2v_full[mask_l]
                # print(f"{U_s2v_l.shape = }")
                # print(f"{rbrdf[:, mask_l, 1:3, 0].shape = }")
                comp = vec2comp(coeff_mat[..., mask_l, 1:3, ::3], -2) # [N, 1|2]
                s2v_right = np.sum(U_s2v_l.conj()[:, None] * comp, -2) / np.sum(np.abs(U_s2v_l)**2)                    

                # ---------- [Stokes to scalar] ----------
                U_v2s_l = U_v2s_full[mask_l]
                # print(f"{U_v2s_l.shape = }")
                # print(f"{rbrdf[:, mask_l, 0, 1:3].shape = }")
                comp = vec2comp(coeff_mat[..., mask_l, ::3, 1:3], -1) # [N, 1|2]
                v2s_right = np.sum(U_v2s_l.conj()[:, None] * comp, -2) / np.sum(np.abs(U_v2s_l)**2)
                
                if cod_type == CodType.POLAR3:
                    convc_s2v[..., l] = s2v_right.squeeze(-1)
                    convc_v2s[..., l] = v2s_right.squeeze(-1)
                else:
                    convc_s2v[..., l, :] = s2v_right
                    convc_v2s[..., l, :] = v2s_right
            
            if cod_type >= CodType.POLAR2:
                # ---------- [Stokes to Stokes] ----------
                s_lp = cidx_lpolar(cod_type)
                comp, _ = mat2comppair(coeff_mat[..., mask_s2s, s_lp, s_lp])
                convc_v2va[..., l] = np.mean(comp, -1)
                
                mask_anti = mask_l & (mi_full == -mo_full)
                phase = (-1) ** (mi_full[mask_anti] % 2)
                _, comp = mat2comppair(coeff_mat[..., mask_anti, s_lp, s_lp])
                convc_v2vb[..., l] = np.mean(phase * comp, -1)

        return cls(shm.cod_type, convc_s2s, convc_s2v, convc_v2s,
                   convc_v2va, convc_v2vb, weighted=True).to_weighted(weighted)

    def __coeff(self, suffix: str, weighted: bool):
        """For `SHConv.coeff_*2*` methods"""
        assert isinstance(weighted, bool)
        coeff = self.get_dict()[suffix]
        if (self.weighted == weighted) or (coeff is None):
            return coeff
        else:
            w = self.weight_l if weighted else 1/self.weight_l
            axes = (None,) * self.ndim_p_dict[suffix]
            return coeff * w[:,  *axes]

    def coeff_s2s(self, weighted: Optional[bool]=False) -> np.ndarray:
        return self.__coeff('s2s', weighted)
    def coeff_s2v(self, weighted: Optional[bool]=False) -> np.ndarray:
        return self.__coeff('s2v', weighted)
    def coeff_v2s(self, weighted: Optional[bool]=False) -> np.ndarray:
        return self.__coeff('v2s', weighted)
    def coeff_v2va(self, weighted: Optional[bool]=False) -> np.ndarray:
        return self.__coeff('v2va', weighted)
    def coeff_v2vb(self, weighted: Optional[bool]=False) -> np.ndarray:
        return self.__coeff('v2vb', weighted)
    
    def to_weighted(self, weighted: bool) -> SHConv:
        if self.weighted == weighted:
            return self
        else:
            return SHConv(self.cod_type, self.coeff_s2s(weighted), self.coeff_s2v(weighted), self.coeff_v2s(weighted),
                          self.coeff_v2va(weighted), self.coeff_v2vb(weighted), weighted)

    def __repr__(self) -> str:
        if self.cod_type == CodType.SCALAR:
            str_polar = "(SCALAR)"
        else:
            str_polar = f"| p: {int(self.cod_type)}, {int(self.cod_type)}"
        str2 = lambda tup: str(tup)[1:-1]
        res = f"{self.__class__.__name__}[c: {str2(self.shape_chan)} | L: {self.level} {str_polar}][\n"
        res += f"  level = {self.level},\n"
        res += f"  cod_type = {self.cod_type!r},\n"
        res += f"  weighted = {self.weighted},\n"
        for label, coeff in self.get_dict().items():
            if coeff is not None:
                res += f"  {label}[{str2(coeff.shape)}] {coeff.dtype},\n"
        res += "]"
        return res
    
    def __matmul__(self, x: SHCoeff) -> SHCoeff:
        '''
        NOTE current version assumes `SHType.REAL` for scalars and `SHType.COMP` for linear polarization.
        '''
        coeff = x.coeff # [*c, N, p]
        assert self.level == x.level
        assert self.cod_type == x.cod_type
        assert self.sh_type == x.sh_type
        
        if isinstance(x, SHVec):
            l, m = level2lms(self.level, DomType.UNI, unstack=True) # [N]
            idx_negm = lms2idx(np.stack([l, -m], axis=-1), DomType.UNI)  # [N]
            shape = np.broadcast_shapes(self.shape_chan, x.shape_chan) + coeff.shape[x.ndim_chan:]
            res = np.zeros(shape, dtype=coeff.dtype) # [*c, N, p]
            
            U1_s2v, U1_v2s = get_U(m, m) # [N]
            U2_s2v, U2_v2s = get_U(-m, m) # [N]

            # ---------- Main ----------
            if self.cod_type == CodType.SCALAR:
                res[:] = self.s2s[..., l] * coeff

            elif self.cod_type == CodType.POLAR2:
                coeff_vcomp = vec2comp(coeff)
                conv_resh = self.v2va[..., l]
                res[...] += comp2vec(conv_resh * coeff_vcomp)
                
                coeff_vcomp = vec2comp(coeff[..., idx_negm, :])
                conv_resh = self.v2vb[..., l]
                phase = (-1) ** (m % 2)
                res[...] += comp2vec(phase * conv_resh * coeff_vcomp.conj())

            elif self.cod_type == CodType.POLAR3:
                '''
                NOTE CAUTION
                    * `=` and `+=`
                    * Do not compute twice for `m == 0`. See `np.where(...)`.
                '''
                ## Scalar-to-scalar
                res[..., 0] = self.s2s[..., l] * coeff[..., 0]
                
                mask_m0 = m == 0
                ## Scalar-to-Stokes
                conv_resh = self.s2v[..., l]
                res[..., 1:3] = comp2vec(conv_resh * U1_s2v * coeff[..., 0])
                res[..., 1:3] += np.where(np.expand_dims(mask_m0, -1),
                                        0.0,
                                        comp2vec(conv_resh * U2_s2v * coeff[..., idx_negm, 0]))
                
                ## Stokes-to-scalar
                conv_resh = self.v2s[..., l]
                res[..., 0] += np.sum(comp2vec((conv_resh * U1_v2s)) * coeff[..., 1:3], -1)
                res[..., 0] += np.where(mask_m0,
                                        0.0,
                                        np.sum(comp2vec((conv_resh * U2_v2s)) * coeff[..., idx_negm, 1:3], -1))
                
                ## Stokes-to-Stokes
                coeff_vcomp = vec2comp(coeff[..., 1:3])
                conv_resh = self.v2va[..., l]
                res[..., 1:3] += comp2vec(conv_resh * coeff_vcomp)
                
                coeff_vcomp = vec2comp(coeff[..., idx_negm, 1:3])
                conv_resh = self.v2vb[..., l]
                phase = (-1) ** (m % 2)
                res[..., 1:3] += comp2vec(phase * conv_resh * coeff_vcomp.conj())

            else: # self.cod_type == CodType.POLAR4:
                """
                TODO: Testing and merging with the previous branch
                """
                idx_s0 = cidx_scalar(self.cod_type)
                idx_s2 = cidx_lpolar(self.cod_type)
                def expd1(arr):
                    return np.expand_dims(arr, -1)
                # res[*c, N, p]
                # self.s2s[*c, level, 2, 2]
                # self.s2v[*c, level, 2]
                # self.v2s[*c, level, 2]
                # self.v2va[*c, level]
                # self.v2vb[*c, level]

                ## Scalar-to-scalar
                res[..., idx_s0] = matmul_vec1d(self.s2s[..., l, :, :], coeff[..., idx_s0])
                
                mask_m0 = m == 0
                ## Scalar-to-Stokes
                conv_resh = self.s2v[..., l, :]
                res[..., idx_s2] = comp2vec(U1_s2v * (conv_resh * coeff[..., idx_s0]).sum(-1))
                res[..., idx_s2] += np.where(expd1(mask_m0),
                                             0.0,
                                             comp2vec(U2_s2v * (conv_resh * coeff[..., idx_negm, idx_s0]).sum(-1)))
                
                ## Stokes-to-scalar
                conv_resh = self.v2s[..., l, :]
                res[..., idx_s0] += matmul_vec1d(comp2vec(conv_resh * expd1(U1_v2s)), coeff[..., idx_s2])
                res[..., idx_s0] += np.where(expd1(mask_m0),
                                             0.0,
                                             matmul_vec1d(comp2vec(conv_resh * expd1(U2_v2s)), coeff[..., idx_negm, idx_s2]))
                
                ## Stokes-to-Stokes
                coeff_vcomp = vec2comp(coeff[..., idx_s2])
                conv_resh = self.v2va[..., l]
                res[..., idx_s2] += comp2vec(conv_resh * coeff_vcomp)
                
                coeff_vcomp = vec2comp(coeff[..., idx_negm, idx_s2])
                conv_resh = self.v2vb[..., l]
                phase = (-1) ** (m % 2)
                res[..., idx_s2] += comp2vec(phase * conv_resh * coeff_vcomp.conj())
            
            # ---------- Weight ----------
            if self.weighted == False:
                axes = (...) if self.cod_type == CodType.SCALAR else (..., None)
                res *= np.sqrt(4*np.pi/(2*l+1))[axes]

            return SHVec(res, self.cod_type, self.sh_type)
        elif isinstance(x, SHMat):
            raise NotImplementedError()
        elif isinstance(x, SHConv):
            raise NotImplementedError()
        else:
            raise TypeError(f"Invalid argument type: {type(x)=}")
    
    def allclose(self, shc: SHConv, rtol: Optional[float]=1e-05, atol: Optional[float]=1e-08) -> bool:
        if not isinstance(shc, SHConv):
            raise TypeError(f"Invalid type: {type(shc)=}. The type must be `SHConv`.")
        if self.cod_type != shc.cod_type:
            raise ValueError(f"Attributes `cod_type` for `self` and `shv` must be equal, but currently:\n"
                             f"{self.cod_type = }, {shc.cod_type = }")
        res = True
        arg_dict = shc.to_weighted(self.weighted).get_dict()
        for key, coeff in self.get_dict().items():
            if coeff is not None:
                res = res and np.allclose(coeff, arg_dict[key])
        return res
    
    def copy(self) -> SHConv:
        return SHConv(self.cod_type, self.s2s.copy(), self.s2v.copy(), self.v2s.copy(),
                      self.v2va.copy(), self.v2vb.copy(), weighted=self.weighted)

    def cut(self, level: Optional[int] = None, cod_type: Optional[CodType] = None) -> SHConv:
        # ---------- Cut level ----------
        att_dict = self.get_dict()
        if level is None:
            res_dict = att_dict
        else:
            res_dict = dict()
            for key, val in att_dict.items():
                if (self.cod_type == CodType.POLAR4) and (key == 's2s'):
                    res_dict[key] = val[..., :level, :, :].copy()
                elif (self.cod_type == CodType.POLAR4) and ('s' in key):
                    res_dict[key] = val[..., :level, :].copy()
                else:
                    res_dict[key] = val[..., :level].copy()
        
        # ---------- Cut cod_type ----------
        if cod_type is None:
            cod_type = self.cod_type
        else:
            cod_type = CodType(cod_type)
            if cod_type != self.cod_type:
                if (cod_type > self.cod_type) or (self.cod_type == CodType.POLAR2):
                    raise ValueError(f"Invalid argument {cod_type = } for the attribute {self.cod_type = }.")
                if cod_type == CodType.POLAR2:
                    res_dict['s2s'] = None
                elif self.cod_type == CodType.POLAR4:
                    res_dict['s2s'] = res_dict['s2s'][..., 0, 0]
                if cod_type <= CodType.POLAR2:
                    res_dict['s2v'] = None
                    res_dict['v2s'] = None
                if cod_type == CodType.SCALAR:
                    res_dict['v2va'] = None
                    res_dict['v2vb'] = None
                if cod_type == CodType.POLAR3:
                    res_dict['s2v'] = res_dict['s2v'][..., 0]
                    res_dict['v2s'] = res_dict['v2s'][..., 0]
        return SHConv.from_dict(cod_type, res_dict, weighted=self.weighted)

    def cut_low(self, level_to: int) -> SHConv:
        att_dict = self.get_dict()
        for key, val in att_dict.items():
            val_res = val.copy()
            ## TODO generalize to CodType.POLAR4
            val_res[..., :level_to] = 0
            att_dict[key] = val_res
        param_dict = dict()
        for key, val in att_dict.items():
            param_dict["coeff_" + key] = val
        return SHConv(self.cod_type, **param_dict, weighted=self.weighted)

    def to_SHMat(self, dom_type: Optional[DomType] = DomType.UNI, level_upto: Optional[int]=None
                ) -> SHMat: # [*c, N_UNI, N_UNI, cod_type, cod_type]
        if level_upto is None:
            level_upto = self.level
        dom_type = DomType(dom_type)
        if dom_type in [DomType.ISOBI2, DomType.BI]:
            dom_type_temp = dom_type
        else:
            dom_type_temp = DomType.ISOBI2
        
        N = level2num(level_upto, dom_type_temp)
        if self.cod_type == CodType.SCALAR:
            SHmat = np.zeros(self.shape_chan + (N,))
        else:
            SHmat = np.zeros(self.shape_chan + (N, self.cod_type, self.cod_type))
        
        li, lo, mi, mo = level2lms(level_upto, dom_type_temp, unstack=True)
        mask_common = (li == lo) & (np.abs(mi) == np.abs(mo)) # [N]
        mask_diag = mask_common & (mi == mo) # [N]
        mask_anti = mask_common & (mi == -mo) # [N]

        slices_chan = (np.s_[:],) * self.ndim_chan
        if self.cod_type != CodType.POLAR2:
            copym00_s2s = self.coeff_s2s(True)[*slices_chan, li] # [*c, N, ()|(2, 2)]
        if self.cod_type >= CodType.POLAR3:
            nones = (None,) * (4 - int(self.cod_type))
            copym00_s2v = self.coeff_s2v(True)[*slices_chan, li, *nones] # [*c, N, 1|2]
            copym00_v2s = self.coeff_v2s(True)[*slices_chan, li, *nones] # [*c, N, 1|2]
        if self.cod_type >= CodType.POLAR2:
            copym00_v2va = self.coeff_v2va(True)[*slices_chan, li] # [*c, N]
            copym00_v2vb = self.coeff_v2vb(True)[*slices_chan, li] # [*c, N]
        
        # ---------- Scalar to scalar ----------
        if self.cod_type != CodType.POLAR2:
            nones = (None,)*self.ndim_p_dict['s2s']
            s2s_right = np.where(mask_diag[..., *nones], copym00_s2s, 0) # [*c, N, ()|(2, 2)]
            if self.cod_type == CodType.SCALAR:
                SHmat[...] = s2s_right
            elif self.cod_type == CodType.POLAR3:
                SHmat[...,0,0] = np.where(mask_diag, copym00_s2s, 0)
            else:
                SHmat[...,::3,::3] = np.where(mask_diag[..., *nones], copym00_s2s, 0)
        
        if self.cod_type >= CodType.POLAR3:
            U_s2v, U_v2s = get_U(mi, mo) # [N]
            mask_common_Npp = mask_common[:, None, None] # [N, 1, 1]

            # ---------- Scalar to Stokes ----------
            SHmat[...,1:3,::3] = np.where(mask_common_Npp,
                                          comp2vec(U_s2v[:,None] * copym00_s2v, -2), 0)
            # ---------- Stokes to scalar ----------
            SHmat[...,::3,1:3] = np.where(mask_common_Npp,
                                          comp2vec(U_v2s[:,None] * copym00_v2s, -1), 0)
        
        # ---------- Stokes to Stokes ----------
        if self.cod_type >= CodType.POLAR2:
            phase = (-1) ** (mi % 2)
            mat1 = np.where(np.expand_dims(mask_diag, (-1,-2)),
                            comp2mat(copym00_v2va), 0)
            mat2 = np.where(np.expand_dims(mask_anti, (-1,-2)),
                            comp2mat(copym00_v2vb*phase)@J_conj, 0)
            s_ = cidx_lpolar(self.cod_type)
            SHmat[...,s_,s_] = mat1 + mat2
        return SHMat(SHmat, dom_type_temp, self.cod_type, self.sh_type).to_domtype(dom_type)

    def get_dict(self) -> dict[str, np.ndarray]:
        return {'s2s': self.s2s, 's2v': self.s2v, 'v2s': self.v2s,
                'v2va': self.v2va, 'v2vb': self.v2vb}
    
    def save(self, filename: Union[str, Path]):
        filename = Path(filename)
        assert filename.suffix == ".npy", f"Invalid file extension for {filename=}."

        with open(filename, 'wb') as f:
            np.save(f, self.weighted)
            np.save(f, self.cod_type)
            np.save(f, self.s2s)
            np.save(f, self.s2v)
            np.save(f, self.v2s)
            np.save(f, self.v2va)
            np.save(f, self.v2vb)

    def tabulate(self, id_ch_show: int) -> str:
        att_dict = self.get_dict()
        keys = list(att_dict.keys())
        if self.shape_chan == ():
            coeff_mats = [val for val in att_dict.values()]
        else:
            coeff_mats = [val[id_ch_show, ...] for val in att_dict.values()]
        
        headers = ["l"] + [key for key in keys]
        table = []
        for l, row in enumerate(zip(*coeff_mats)):
            row2 = [l, row[0]] + [format_complex(x, " %.2e") for x in row[1:]]
            table.append(row2)
        return tabulate(table, headers=headers, floatfmt=".2e")

    
########################################
### Helper functions
########################################

# def SH_indexing(SH_ordered: ArrayLike, # [*,N1], N1 == N(L, dom_type)
def SH_indexing(SH_ordered: ArrayLike, # [*,N1], N1 == N(L, dom_type)
                dom_type:   DomType,
                *lms:       ArrayLike, # [N2] for each
               )         -> ArrayLike: # [*,N2]
    assert isinstance(dom_type, DomType)
    return SH_ordered[..., lms2idx(np.stack(lms, axis=-1), dom_type)]
        
        
# def SHcoeff_ISOBI_doubling(coeff:   ArrayLike | Tuple[ArrayLike,ArrayLike], # [*c, N1(L)]
def ISOBI_doubling(coeff:   ArrayLike | Tuple[ArrayLike,ArrayLike], # [*c, N1(L)]
                   spin2:   Tuple[bool,bool],
                   sh_type: SHType
                  )      -> ArrayLike | Tuple[ArrayLike,ArrayLike]: # [*c, N2(L), *p]
    '''
    Convert input N1(L) := level2numcoeff(L, DomType.ISOBI) SH coefficient to
    N2(L) := level2numcoeff(L, DomType.ISOBI2) of them.
    
    params:
            coeff:    ArrayLike[*c, N1(L)],         if spin2 != (T,T)
                      (ArrayLike["], ArrayLike["]), if spin2 == (T,T)
            spin2:    (bool, bool), incident and outgoing resp.
            sh_type:  SHType
        return:
            coeff:    ArrayLike[*c, N2(L), *p],     if spin2 != (T,T)
            (c1, c2): (ArrayLike["], ArrayLike["]), if spin2 == (T,T)
                - *p == ()  if spin2 == (F,F)
                        2   if spin2 in [(T,F), (F,T)]
                        2,2 if spin2 == (T,T)
                
    NOTE See also `class:SphereGrid.SHio_iso_upto` and
         `2022.08.30. Factorizing 3-parameter BSDF.md`,
         and `2022.11.22. Isotropic Polarimetric BSDF.md`.
    '''
    assert isinstance(sh_type, SHType)
    if spin2 != (True, True):
        N1 = coeff.shape[-1]
    else:
        N1 = coeff[0].shape[-1]
        assert N1 == coeff[1].shape[-1]
        
    level = num2level_assert(N1, DomType.ISOBI)
    li_N2, lo_N2, mi_N2, mo_N2 = level2lms(level, DomType.ISOBI2, unstack=True)
    li_N1, lo_N1, m_N1 = level2lms(level, DomType.ISOBI, unstack=True)
    
    if spin2 != (True, True):
        mask_pmo = mo_N2 >= 0
        mask_pmi = mi_N2 >= 0
        indexing = lambda li,lo,m: SH_indexing(coeff, DomType.ISOBI, li, lo, m)
            
    if spin2 == (False,False):
        ## Scalar to scalar
        if sh_type == SHType.COMP:
            raise NotImplementedError()
        else:
            ## (mi, mo) == (m, |m|) for DomType.ISOBI
            C_mi = indexing(li_N2, lo_N2, mi_N2)
            C_mineg = indexing(li_N2, lo_N2, -mi_N2)
            coeff_res = np.select(
                [mask_pmo, mask_pmi],
                [C_mi, -C_mineg], C_mineg)
        return coeff_res
    
    elif spin2 == (True, False):
        ## Vector to scalar
        if sh_type == SHType.COMP:
            raise NotImplementedError()
        else:
            ## (mi, mo) == (m, |m|) for DomType.ISOBI
            C_mi = indexing(li_N2, lo_N2, mi_N2)
            coeff_res = np.select(
                [mask_pmo, mask_pmi],
                [C_mi, -1j*C_mi], 1j*C_mi)
        return comp2vec(coeff_res)
    
    elif spin2 == (False, True):
        ## Scalar to vector
        if sh_type == SHType.COMP:
            raise NotImplementedError()
        else:
            ## (mi, mo) == (|m|, m) for DomType.ISOBI
            C_mo = indexing(li_N2, lo_N2, mo_N2)
            coeff_res = np.select(
                [mask_pmi, mask_pmo],
                [C_mo, -1j*C_mo], 1j*C_mo)
        return comp2vec(coeff_res)
            
    elif spin2 == (True, True):
        ## Vector to vector
        ## (mi, mo) == (m, m) for Y1, and == (m, -m) for Y2
        ## NOTE independent of `sh_type`
        coeff1, coeff2 = coeff
        mask_m_equal = mi_N2 == mo_N2
        mask_m_negeq = mi_N2 == -mo_N2
        C1_mi = SH_indexing(coeff1, DomType.ISOBI, li_N2, lo_N2, mi_N2)
        C2_mi = SH_indexing(coeff2, DomType.ISOBI, li_N2, lo_N2, mi_N2)
        
        coeff1_res = np.where(mask_m_equal, C1_mi, 0)
        coeff2_res = np.where(mask_m_negeq, C2_mi, 0)
        
        return comp2mat(coeff1_res) + comp2mat(coeff2_res) @ J_conj
    
    else:
        raise ValueError(f"The argument 'spin2' shoulda 2-tuple of booleans, but {spin2} is given.")


def _toDBI(coeff:    ArrayLike, # [*c, N] | [*c, N, po, pi]
           dom_type: DomType,# Any     | DomType.ISOBI/DomType.ISOBI2/DomType.BI
           cod_type: CodType, # CodType.SCALAR | CPOLARp
          ) ->       ArrayLike: # [*c, N] | [*c, N, po, pi]
    level, axis_N = assert_dctype(coeff, dom_type, cod_type)
    
    if dom_type == DomType.BI:
        return coeff
    
    N_res = level2num(level, DomType.BI)
    shape_res = list(coeff.shape)
    shape_res[axis_N] = N_res
    res = np.zeros(shape_res, dtype=coeff.dtype)
    coeff_view = np.moveaxis(coeff.view(), axis_N, -1)
    res_view = np.moveaxis(res.view(), axis_N, -1)
    
    if dom_type in [DomType.ISOBI, DomType.ISOBI2]:
        lms_from = level2lms(level, dom_type) # N
        if dom_type == DomType.ISOBI:
            li, lo, m = npunstack(lms_from)
            idx_toDBI = lms2idx(np.stack([li, lo, m, m], axis=-1), DomType.BI) # N
        else:
            idx_toDBI = lms2idx(lms_from, DomType.BI) # N
        res_view[..., idx_toDBI] = coeff_view
    else:
        raise ValueError(f"{dom_type = } should be one of [DomType.ISOBI, DomType.ISOBI2].")
    return res

def _fromDBI(coeff:       ArrayLike, # [*c, N] | [*c, N, po, pi]
             dom_type_to: DomType,# Any     | DomType.ISOBI/DomType.ISOBI2/DomType.BI
             cod_type:    CodType  # CodType.SCALAR | CPOLARp
             # n_out:       int = 1    # 1 or 2 (+ error info)
            ) ->          ArrayLike: # [*c, N] | [*c, N, po, pi]
    '''
    NOTE This function contains a numerical assertion for `coeff`.
    '''
    level, axis_N = assert_dctype(coeff, DomType.BI, cod_type)
    
    if dom_type_to == DomType.BI:
        return coeff
    
    N_res = level2num(level, dom_type_to)
    shape_res = list(coeff.shape)
    shape_res[axis_N] = N_res
    
    if dom_type_to in [DomType.ISOBI, DomType.ISOBI2]:
        ## NOTE See "Order consistency among `DomType.ISOBI`, `DomType.ISOBI2`, and `DomType.BI`"
        ##      in `test_SH_indices.ipynb`
        lms_from = level2lms(level, DomType.BI) # N
        mi_from, mo_from = npunstack(lms_from[:,2:], axis=-1)
        if dom_type_to == DomType.ISOBI:
            mask = mi_from == mo_from
        else: # dom_type_to == DomType.ISOBI2
            mask = np.abs(mi_from) == np.abs(mo_from)
        
        if cod_type == CodType.SCALAR:
            res = coeff[..., mask]
            err = coeff[..., ~mask]
        else:
            res = coeff[..., mask, :, :]
            err = coeff[..., ~mask, :, :]
        assert np.allclose(err, 0)
    else:
        raise ValueError(f"{dom_type_to = } should be one of [DomType.ISOBI, DomType.ISOBI2].")
    return res


def _DBI2mat(coeff:    ArrayLike, # [*c, N_DomType.BI]  | [*c, N_DomType.BI, po, pi]
             cod_type: CodType  # CodType.SCALAR      | CPOLARp
            ) ->       ArrayLike: # [*c, No, Ni] | [*c, No, Ni, po, pi]
    level, axis_N = assert_dctype(coeff, DomType.BI, cod_type)
    N_UNI = level2num(level, DomType.UNI)
    
    li,lo,mi,mo = level2lms(level, DomType.BI, unstack=True)
    idx_i = lms2idx(np.stack([li, mi], axis=-1), DomType.UNI)
    idx_o = lms2idx(np.stack([lo, mo], axis=-1), DomType.UNI)
    idx_DBI2DUNIsq = idx_o * N_UNI + idx_i
    shape_res = list(coeff.shape)
    shape_res[axis_N] = N_UNI
    shape_res.insert(axis_N, N_UNI)
    res = np.zeros(shape_res, dtype=coeff.dtype)
    coeff_v = np.moveaxis(coeff.view(), axis_N, -1)
    res_v = np.moveaxis(res.view().reshape(coeff.shape), axis_N, -1)
    res_v[..., idx_DBI2DUNIsq] = coeff_v
    
    return res

def _mat2DBI(coeff:    ArrayLike, # [*c, No, Ni] | [*c, N_o, N_i, po, pi]
             cod_type: CodType  # CodType.SCALAR      | CPOLARp
            ) ->       ArrayLike: # [*c, N_BI]   | [*c, N_BI, po, pi]
    if cod_type == CodType.SCALAR:
        axis_N = -1
    elif cod_type in [CodType.POLAR2, CodType.POLAR3, CodType.POLAR4]:
        axis_N = -3
    else:
        raise TypeError(f"{cod_type =} should be CodomainType.")
    N_UNI = coeff.shape[axis_N]
    assert N_UNI == coeff.shape[axis_N-1]
    level = num2level_assert(N_UNI, DomType.UNI)
    N_BI = level2num(level, DomType.BI)
    
    li,lo,mi,mo = level2lms(level, DomType.BI, unstack=True)
    idx_i = lms2idx(np.stack([li, mi], axis=-1), DomType.UNI)
    idx_o = lms2idx(np.stack([lo, mo], axis=-1), DomType.UNI)
    if cod_type == CodType.SCALAR:
        return coeff[..., idx_o, idx_i]
    else:
        return coeff[..., idx_o, idx_i, :, :]

def _SHcoeff_apply_rotation(coeff:    ArrayLike,
                            rotation: Union[ArrayLike, quat.array],
                            sh_type:  SHType = SHType.COMP,
                            D_cache:  ArrayLike = None
                           ) ->       ArrayLike:
    '''
    params:
        SHcoeff[*c, N]: ArrayLike, `N == level2num(level, DomType.UNI)`
        rotation[3]:         ArrayLike, rotation transform, used for scalars
    return:
        SHcoeff_rot[*c, N]: ArrayLike
    '''
    N = coeff.shape[-1]
    level = num2level_assert(N, DomType.UNI)
    if D_cache is None:
        if isinstance(rotation, quat.array):
            R = rotation
        else:
            R = quat.array.from_axis_angle(rotation)
        wigner = spherical.Wigner(level-1)
        D = wigner.D(R).conj()
    else:
        D = D_cache

    if sh_type == SHType.REAL:
        coeff = SHVec(coeff, CodType.SCALAR, SHType.REAL).to_shtype(SHType.COMP).coeff
    
    coeff_rot = np.zeros(coeff.shape, dtype=np.complex128)#SHcoeff.dtype)
    # TODO Can `dtype=np.complex128` above be generalized?
    iD1 = 0
    for l in range(level):
        im1 = level2num(l, DomType.UNI)
        im2 = level2num(l+1, DomType.UNI)
        nm = im2 - im1
        iD2 = iD1 + nm*nm
        Dl = D[iD1:iD2].reshape(nm, nm)
        coeff_rot[...,im1:im2] = matmul_vec1d(Dl, coeff[...,im1:im2])
        iD1 = iD2
        
    if sh_type == SHType.REAL:
        coeff_rot = SHVec(coeff_rot, CodType.SCALAR, SHType.COMP).to_shtype(SHType.REAL).coeff
    return coeff_rot

def _SHVec_to_TPmat(shv: SHVec, level_to: int, spin: Optional[int] = 0) -> np.ndarray:
    """
    SH vector to triple product matrix
    NOTE Assume all assertions have been already done before calling it.
    """
    if spin not in [0, 2]:
        raise ValueError(f"`spin` must be 0 or 2, but {spin=} is given.")
    shape_chan = shv.coeff.shape[:-1]
    N_to = level2num(level_to, DomType.UNI)
    res_dtype = shv.coeff.dtype if spin == 0 else np.complex128
    res = np.zeros(shape_chan + (N_to, N_to), dtype=res_dtype)
    w3jCalc = spherical.Wigner3jCalculator(level_to-1, level_to-1)

    for idx_o in range(N_to):
        lo, mo = idx2lms(idx_o, DomType.UNI)
        phase_o = (-1) ** (int(spin+mo) % 2)
        const_o = math.sqrt((2*lo+1)/4/math.pi) * phase_o
        
        for idx_i in range(N_to):
            li, mi = idx2lms(idx_i, DomType.UNI)
            const_io = const_o * math.sqrt(2*li+1)
            i1 = abs(li-lo); i2 = li+lo+1
            l_prime = np.arange(i1, i2)
            res_temp = np.sqrt(2*l_prime + 1)
            res_temp *= w3jCalc.calculate(li, lo, -spin, spin)[i1:i2]
            res_temp *= w3jCalc.calculate(li, lo, mi, -mo)[i1:i2]
            f_idx = lms2idx(np.stack(np.broadcast_arrays(l_prime, mo-mi), -1), DomType.UNI)
            res[..., idx_o, idx_i] = const_io * np.sum(res_temp * shv.coeff[..., f_idx], axis=-1)
    return res

class SHTransform:
    """
    Spatical transforms including rotation and reflection on SH coefficient vectors and matrices
    """
    pass

class SHReflection(SHTransform):
    def __matmul__(self, arg: Union[SHVec, SHMat]):
        if isinstance(arg, SHVec):
            level = arg.level
            cod_type = arg.cod_type
            coeff = arg.coeff
            l, m = level2lms(level, DomType.UNI, unstack=True)
            
            phase_l = (-1) ** (l % 2)
            phase_lm = (-1) ** ((l + m) % 2)
            idx_org2res = lms2idx(np.stack([l,-m], -1), DomType.UNI)
            
            if cod_type == CodType.SCALAR:
                res = phase_lm*coeff
            else:
                assert cod_type == CodType.POLAR3
                res = np.zeros(coeff.shape, coeff.dtype)
                res[..., 0] = phase_lm*coeff[..., 0]
                res[..., 1] = phase_l * coeff[..., idx_org2res, 1]
                res[..., 2] = -phase_l * coeff[..., idx_org2res, 2]
            return SHVec(res, cod_type, arg.sh_type)
        elif isinstance(arg, SHMat):
            '''
            Return (coeff mat. of reflection operator) @ coeff
            TODO: Handle weired numpy array shape
            '''
            level = arg.level
            cod_type = arg.cod_type
            coeff = arg.coeff
            if arg.dom_type not in [DomType.ISOBI2, DomType.BI]:
                raise NotImplementedError()
            assert coeff.dtype == np.float64
            li,lo,mi,mo = level2lms(level, arg.dom_type, unstack=True)
            
            phase_l = (-1) ** (lo % 2)
            phase_lm = (-1) ** ((lo + mo) % 2)
            idx_org2res = lms2idx(np.stack([li,lo,mi,-mo], -1), arg.dom_type)
            res = np.zeros(coeff.shape, dtype=coeff.dtype)
            
            # s_s = cidx_scalar(cod_type)
            # s_v = cidx_lpolar(cod_type)
            if cod_type == CodType.SCALAR:
                res[...] = phase_lm * coeff
            else:
                assert cod_type == CodType.POLAR3
                res[..., 0, :] = np.expand_dims(phase_lm, -1) * coeff[..., :, 0, :]
                res[..., 1, :] = np.expand_dims(phase_l, -1) * coeff[..., idx_org2res, 1, :]
                res[..., 2, :] = -np.expand_dims(phase_l, -1) * coeff[..., idx_org2res, 2, :]
            return SHMat(res, arg.dom_type, arg.cod_type, arg.sh_type)
        else:
            return arg.__rmatmul__(self)
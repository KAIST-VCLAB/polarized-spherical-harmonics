from __future__ import annotations
from time import time
from tqdm import tqdm
from typing import Type, List, Union, Optional, Callable, Literal
from pathlib import Path
import math
import numpy as np
from numpy.typing import ArrayLike
import scipy.io

from polarsh.array import *
from polarsh.sphere import *
from polarsh.SH import *
from polarsh.polar import *
from polarsh.image import *
from .Grid import *
from .StokesField import *
from polarsh import SHMat, SHConv, util
from polarsh.SHCoeff import ISOBI_doubling


##################################################
### py:class:MuellerField
##################################################
class MuellerField:
    def __init__(self,
                 SphFF, #: SphereFrameField, # SphFF.shape: *
                 M:     ArrayLike,         # *x4x4|*xcx4x4 Mueller matrices
                ):
        self.SphFF = SphFF
        self.SphGrid = SphFF.SphGrid
        assert isinstance(self.SphGrid.dom_type, DomType) and self.SphGrid.dom_type != DomType.UNI # DomType.BI or DomType.ISOBI
        self.shape_grid = self.SphGrid.shape
        assert self.shape_grid == SphFF.shape
        self.axes_grid = tuple(range(len(self.shape_grid)))
        self.cod_type = CodType(M.shape[-1])
        assert M.shape[-2] == M.shape[-1], f"M.shape ({M.shape}) should be end with two same numbers."
        assert M.shape[:len(self.shape_grid)] == self.shape_grid, f"The front part of M.shape ({M.shape}) should be equal to self.shape_grid ({self.shape_grid})"
        self.shape_chan = M.shape[len(self.shape_grid):-2]
        self.axes_chan = tuple(range(len(self.shape_grid), M.ndim-2))
        self.ndim_grid = len(self.shape_grid)
        self.ndim_chan = len(self.shape_chan)

        self.M = M
        self._mask_validM = None
    
    @classmethod
    def from_SB20(cls,
                  datafile: Union[str, Path],
                  infofile: Union[str, Path],
                  *,
                  dsamp:    int = 1, # subsampling interval for parameter range
                  quiet:    bool = True
                 ) ->       MuellerField:
        '''
        MuellerField instance from [Seung-Hwan Baek et al. 2020] pBRDF dataset
        reference: https://vclab.kaist.ac.kr/siggraph2020/pbrdfdataset/kaistdataset.html
        '''
        if not quiet:
            print(f"\n##############################")
            print(f"### [Function start] MuellerField_from_SB20 with datafile: {util.str_trunc(datafile)} and infofile: {util.str_trunc(infofile)}.\n")
            t = time()

        data_mat = scipy.io.loadmat(str(datafile))
        info_mat = scipy.io.loadmat(str(infofile))

        ## pbrdf_rgb[phi_d,theta_d,theta_h,color,4,4]
        mueller_dsamp = data_mat['pbrdf_T_f'][::dsamp, ::dsamp, ::dsamp, :,:,:]
        pbrdf_rgb = reflectance_spec2rgb(mueller_dsamp, -3)
        if not quiet:
            print(f"Shape of pBRDF: {pbrdf_rgb.shape}")
            print(f"dtype of pBRDF: {pbrdf_rgb.dtype}")
            M_valid = pbrdf_rgb[~np.isnan(pbrdf_rgb)]
            v_max, v_min, v_absmin = M_valid.max(), M_valid.min(), np.abs(M_valid).min()
            print("\n# pBRDF value statistics:")
            print(f"Max = {v_max:>10}\tMin = {v_min:>10}\tAbsMin = {v_absmin:>10}")
        
        phi_d = data_mat['phi_d'].squeeze()[::dsamp] # 361 / dsamp
        theta_d = data_mat['theta_d'].squeeze()[::dsamp] # 91 / dsamp
        theta_h = data_mat['theta_h'].squeeze()[::dsamp] # 91 / dsamp
        ## 361*91*91*3*3
        Fi = np.swapaxes(info_mat['xyzi_T'][::dsamp, ::dsamp, ::dsamp, :,:], -1, -2)
        Fo = np.swapaxes(info_mat['xyzo_T'][::dsamp, ::dsamp, ::dsamp, :,:], -1, -2)

        ## [IMPORTANT] Convert float32 to float64. 
        if not quiet:
            print(f"\n# Primarily dtype of phi_d...Fo: %s, %s, %s, %s, %s"
                % (phi_d.dtype, theta_d.dtype, theta_h.dtype, Fi.dtype, Fo.dtype))
            print("They will be converted to float64")
        phi_d = phi_d.astype(np.float64); theta_d = theta_d.astype(np.float64); theta_h = theta_h.astype(np.float64)
        Fi = Fi.astype(np.float64); Fo = Fo.astype(np.float64)

        ## TODO solve this hard-coding problem raised by 0/0 computation from spin-2 SH
        theta_h_prev = theta_h
        theta_d_prev = theta_d
        mask_th_zero = theta_h==0
        count_th_zero = np.sum(mask_th_zero)
        if np.sum(mask_th_zero) > 0:
            if not quiet:
                print(f"\n# Adding epslion to {count_th_zero} zero-valued theta_h.")
            theta_h = np.where(mask_th_zero, 3.6e-8, theta_h)
        mask_td_same = np.any(colvec(theta_d)==rowvec(theta_h), axis=1).reshape((-1,))
        count_td_same = np.sum(mask_td_same)
        if np.sum(mask_td_same) > 0:
            if not quiet:
                print(f"\n# Subtracting epslion to {count_td_same} theta_d, which is equal to theta_h.")
            theta_d = np.where(mask_td_same, theta_d - 1.8e-8, theta_d)

        SphGrid = SphereGrid.from_meshgrid(DomType.ISOBI, AngType.RAD, phi_d, theta_d, theta_h)
        
        ## Make Fi/Fo has no NaN wherever M has no NaN
        ## NOTE See `analysis_SB20_NaN.ipynb`.
        SphFFcan = SphGrid.ThetaPhiFrameField()
        Fi[:,0,...] = SphFFcan.Fi[:,0,...]
        Fo[:,0,...] = SphFFcan.Fo[:,0,...]
        
        SphFF = SphereFrameField(SphGrid, Fi, Fo, allow_nan=True, quiet=quiet)
        MulGrid = cls(SphFF, pbrdf_rgb)

        if not quiet:
            print(f"\n### [Function end] time: {time()-t:.6f} seconds.")
            print("#"*30)

        return MulGrid

    @classmethod
    def from_pplastic(cls,
                      SphFF:                SphereFrameField,
                      diffuse_reflectance:  ArrayLike, # [3]
                      specular_reflectance: ArrayLike, #[3]
                      eta:                  float, # int_ior / ext_ior
                      alpha:                float, # roughness
                      backend:              Optional[Literal['polarsh', 'mitsuba']] = 'polarsh'
                     ) ->                   MuellerField:
        # ---------- Assertions ----------
        diffuse_reflectance = np.broadcast_to(diffuse_reflectance, (3,))
        specular_reflectance = np.broadcast_to(specular_reflectance, (3,))
        assert diffuse_reflectance.shape == (3,)
        assert specular_reflectance.shape == (3,)
        assert np.isscalar(eta) and np.isscalar(alpha)

        assert isinstance(SphFF, SphereFrameField), f"SphFF should be an instance of SphereFrameField, but currently: {type(SphFF)}"
        sphG = SphFF.SphGrid
        assert sphG.dom_type in [DomType.ISOBI, DomType.ISOBI2, DomType.BI]

        assert hasattr(sphG, 'phid_grid') and hasattr(sphG, 'thetad_grid') and hasattr(sphG, 'thetah_grid')
        veci, veco = sphG.veci_m2s, sphG.veco
        assert not np.isnan(veci).any()
        assert not np.isnan(veco).any()
        assert backend in ['polarsh', 'mitsuba']

        mask_out_hemi = (veci[...,2] <= 0) | (veco[..., 2] <= 0)

        if backend == 'polarsh':
            # ---------- Reflectance & transmittance ----------
            vech = halfvec(veci, veco, sphG.thetai_m2s, sphG.phii_m2s)
            cos_theta_i = veci[...,2]
            cos_theta_o = veco[...,2]
            cos_theta_d = np.cos(*radian_regularize(sphG.ang_type, sphG.thetad_grid))
            cos_theta_h = np.cos(*radian_regularize(sphG.ang_type, sphG.thetah_grid))
            R_wrt_spu = fresnel_reflection_wrt_spu(cos_theta_d, eta)
            Ti_wrt_spu = fresnel_refraction_wrt_spu(cos_theta_i, eta)
            To_wrt_spu = fresnel_refraction_wrt_spu(cos_theta_o, eta)
            TDT = To_wrt_spu[..., :, 0:1] * Ti_wrt_spu[..., 0:1, :]
            TDT = TDT[..., None, :, :] * np.expand_dims(diffuse_reflectance, (-1,-2))
            TDT /= np.pi
            del Ti_wrt_spu, To_wrt_spu
            
            # ---------- Finish specular transmittance ----------
            D = beckmann_distribution(cos_theta_h, alpha)
            G = beckmann_G(veci, veco, vech, alpha)
            value = D * G / (4.0 * cos_theta_i * cos_theta_o)
            R_wrt_spu *= value[..., None, None]
            R_wrt_spu = R_wrt_spu[..., None, :, :] * np.expand_dims(specular_reflectance, (-1,-2))

            # ---------- Build resulting `MuellerField` instance ----------
            Fi_diff = vec2Fspu(sphG.veci_s2m)
            Fo_diff = vec2Fspu(veco)

            MGrid_diff = cls(SphereFrameField(sphG, Fi_diff, Fo_diff), TDT)
            MGrid_spec = cls(sphG.RusinkiewiczFrameField(), R_wrt_spu)
            MGrid =  MGrid_diff.to_SphFF(SphFF) + MGrid_spec.to_SphFF(SphFF)
            MGrid.M[mask_out_hemi] = 0.0
        else: # backend == 'mitsuba'
            import mitsuba as mi
            mi.set_variant('cuda_rgb_polarized_double', 'llvm_rgb_polarized_double',
                           'cuda_rgb_polarized',        'llvm_rgb_polarized')
            bsdf_dict = {'type':'pplastic',
                            'diffuse_reflectance':{
                                'type': 'rgb',
                                'value': diffuse_reflectance
                            },
                            'specular_reflectance':{
                                'type': 'rgb',
                                'value': specular_reflectance
                            },
                            'int_ior': eta,
                            'ext_ior': 1.0,
                            'alpha': alpha
                        }
            bsdf = mi.load_dict(bsdf_dict)
            shape_vec = veci.shape
            veci = veci.reshape(-1, 3)
            veco = veco.reshape(-1, 3)

            ctx = mi.BSDFContext(mi.TransportMode.Importance)
            si = mi.SurfaceInteraction3f()
            si.wi = veci
            bsdf_mi = bsdf.eval(ctx, si, veco)
            bsdf_np = bsdf_mi.numpy().astype(float)
            bsdf_np /= np.expand_dims(np.abs(veco[..., 2]), (-1,-2,-3))
            bsdf_np.shape = (*shape_vec[:-1], 3, 4, 4)
            
            bsdf_np[mask_out_hemi] = 0.0
            MGrid = cls(sphG.TD17FrameField(), bsdf_np).to_SphFF(SphFF)
        return MGrid

    @classmethod
    def zeros_like(cls,
                   muelF: MuellerField
                  ) ->    MuellerField:
        return cls(muelF.SphFF, np.zeros_like(muelF.M))
    
    def __add__(self, arg: MuellerField) -> MuellerField:
        return MuellerField(self.SphFF, self.M + arg.to_SphFF(self.SphFF).M)
    def __sub__(self, arg: MuellerField) -> MuellerField:
        return MuellerField(self.SphFF, self.M - arg.to_SphFF(self.SphFF).M)

    def __repr__(self) -> str:
        def str2(tup: tuple) -> str:
            return str(tup)[1:-1]
        
        result = "MuellerField[\n"
        result += f"  dom_type = {repr(self.SphGrid.dom_type)},\n"
        result += f"  M.shape = [g:{str2(self.shape_grid)} | c:{str2(self.shape_chan)} | p:{str2(self.M.shape[-2:])}],\n"
        result += "]"
        return result
    
    def to_SphFF(self,
                 SphFF_to:  SphereFrameField,
                 allow_nan: bool = False
                ) ->        MuellerField:
        ## Convert self.M w.r.t. given SphFF
        assert self.SphGrid is SphFF_to.SphGrid, "SphGrid attributes in self and SphFF_to should be identical."
        if self.SphGrid.dom_type == DomType.BI:
            self_Fi = self.SphFF.Fi
            self_Fo = self.SphFF.Fo
        else: # self.SphGrid.dom_type in [DomType.ISOBI, DomType.ISOBI2]
            ## NOTE Align (global) azimuth for self and SphFF_to,
            ## since just implementation detail may produces
            ## azimuthally rotated Fi and Fo due to absence of phi_h
            dphi, R, err = align_azimuth(
                           (self.SphFF.zi, self.SphFF.zo),
                           (SphFF_to.zi, SphFF_to.zo))
            assert_error_bound(err, name="Out-of-azimuth for self and SphFF_to", allow_nan=allow_nan)
            self_Fi = R@self.SphFF.Fi
            self_Fo = R@self.SphFF.Fo
        if self.SphFF is SphFF_to:
            return self
        else:
            M_to = mueller_frame_convert(self.M,
                        (self_Fi, self_Fo),
                        (SphFF_to.Fi, SphFF_to.Fo),
                        allow_nan=allow_nan)
            return MuellerField(SphFF_to, M_to)
    
    def mask_validM(self,
                    keepdims: bool = False # False | True
                   )       -> ArrayLike:   # [*g]  | [*g,*c,p,p]
        if self._mask_validM is None:
            self._mask_validM_keepdims = ~np.isnan(self.M)
            self._mask_validM = self._mask_validM_keepdims.all(self.axes_chan + (-1,-2))
        if keepdims:
            return self._mask_validM_keepdims
        else:
            return self._mask_validM
    
    def M_weighted(self,
                   cos_i:     bool = False,
                   allow_nan: bool = False
                  ) ->        np.ndarray:
        '''
        if allow_nan == True, NaN in `self.M` will be considered as zero.
        '''
        axes_app = self.axes_chan + (-1,-2)
        res = self.M * np.expand_dims(self.SphGrid.weight(), axes_app)
        if cos_i == True:
            res *= np.expand_dims(np.abs(np.cos(self.SphGrid.thetai_m2s)), axes_app)
        if allow_nan:
            res = np.where(self.mask_validM(keepdims=True), res, 0)
        return res
    
    def cidx_scalar(self) -> Tuple[int,...]:
        return cidx_scalar(self.cod_type)
    def cidx_lpolar(self) -> Tuple[int,...]:
        return cidx_lpolar(self.cod_type)
    
    ## NOTE Not used
    # def dot_domain(self,  # *g := shape_grid, *c1 := shape_chan
    #                MGrid, # MuellerField
    #                chan_tdot: bool = False, # False | True
    #                allow_nan: bool = False
    #               ) -> ArrayLike: # *gx(*c1|*c2)xpxp | *gx*c1x*c2xpxp
    #     assert self.SphGrid is MGrid.SphGrid, "SphGrid for the two MuellerField instances should be identical."
        
    #     ## Frame alignment
    #     if self.SphFF is MGrid.SphFF:
    #         M_arg = MGrid.M
    #     else:
    #         MGrid_aligned = MGrid.to_SphFF(self.SphFF, allow_nan=allow_nan)
    #         M_arg = MGrid_aligned.M
        
    #     ## Deal with allow_nan
    #     if allow_nan: sum_curr = np.nansum
    #     else: sum_curr = np.sum

    #     ## Deal with channel tensor/elementwise product
    #     if chan_tdot:
    #         ax_ch1 = self.axes_chan
    #         ax_ch2 = tuple_add(MGrid.axes_chan, len(self.axes_chan))
    #         M_arg = np.expand_dims(M_arg, ax_ch1)
    #         M_self = np.expand_dims(self.M, ax_ch2)
    #         weight = np.expand_dims(self.SphGrid.weight(), ax_ch1+ax_ch2)

    #         ## To save memory, execute for loop for each Mueller component
    #         shape_muel = M_self.shape[-2:]
    #         DOT = np.zeros(self.shape_chan + MGrid.shape_chan + shape_muel)
    #         for pi in range(shape_muel[0]):
    #             for pj in range(shape_muel[1]):
    #                 prod = np.conj(M_self[...,pi,pj])*M_arg[...,pi,pj]*weight
    #                 DOT[...,pi,pj] = sum_curr(prod, axis=self.axes_grid)
    #         return DOT

    #     else:
    #         ## TODO deal with proper broadcasting for shape_chans
    #         raise NotImplementedError()
    #         prod = np.conj(M_self)*M_arg*weight
    #         DOT = sum_curr(prod, axis=self.axes_grid)
    #         assert DOT.shape[-2:] == self.M.shape[-2:]
    #         return DOT
    
    def SHCoeff(self,
                  level:     int,
                  sh_type:   SHType = SHType.REAL,
                  allow_nan: bool = False,
                  quiet:     bool = True
                 )        -> SHMat: # [*c, N(L), po, pi]
        """
        sh_type: SHType.REAL: real scalar SH for s0 and s3, spin-2 SH for s1 and s2 (recommended)
                 SHType.COMP: complex scalar SH for s0 and s3, spin-2 SH for s1 and s2 (only for intermediate computation)
        """
        if not quiet:
            t = time()
            print("\n## [Function start] MuellerField.SHCoeff")
        sh_type = SHType(sh_type)
        
        dom_type = self.SphGrid.dom_type
        if dom_type == DomType.ISOBI:
            dom_type = DomType.ISOBI2
        cod_type = self.cod_type
        idx_sc = self.cidx_scalar()
        idx_lp = self.cidx_lpolar()
        n_grid = math.prod(self.shape_grid) # NOTE np.int is more weak to overflow
        N_lm = level2num(level, dom_type)
        
        CanMF = self.SphGrid.ThetaPhiFrameField()
        MGrid_can = self.to_SphFF(CanMF, allow_nan=allow_nan)
        shape_re = (n_grid,) + self.shape_chan + (cod_type, cod_type)
        M_self = MGrid_can.M_weighted(cos_i=True, allow_nan=allow_nan).view().reshape(shape_re)
        ## M_self[Ng, *c, p, p]
        if sh_type == SHType.COMP:
            res_dtype = np.complex128
        else:
            assert sh_type == SHType.REAL
            res_dtype = np.float64
        M_res = np.zeros(self.shape_chan + (N_lm, cod_type, cod_type), dtype=res_dtype)
        
        full_memory = n_grid * N_lm * 4 * 16
        n_pass = math.ceil(full_memory / MEMORY_PER_PASS)
        if not quiet and n_pass > 1:
            iter = tqdm(range(n_pass), desc="Seperate passes due to memory")
        else:
            iter = range(n_pass)
            
        for i_pass in iter:
            i = n_grid*i_pass // n_pass
            j = n_grid*(i_pass+1) // n_pass
            kargs = {"sh_type":sh_type, "cfg_pass":(i_pass, n_pass)}
            
            ''' Scalar-to-scalar components '''
            spin2 = (False, False)
            # idx1, idx2 = np.ix_(idx_sc, idx_sc)
            M_s2s = M_self[i:j, ..., idx_sc, idx_sc] # [Ng',*c,ps,ps]
            SHgrid = self.SphGrid.SHio_iso_upto(level, spin2, **kargs) # [Ng',N1(L)]
            temp = np.tensordot(M_s2s, SHgrid.conj(),
                                axes=(0,0)) # [*c,ps,ps,N1(L)]
            del M_s2s, SHgrid
            ## DomType.ISOBI -> DomType.ISOBI2
            temp = ISOBI_doubling(temp, spin2, sh_type=sh_type) # [*c,ps,ps,N2(L)]
            temp = np.moveaxis(temp, -1, -3) # [*c,N2(L),ps,ps]
            M_res[..., idx_sc, idx_sc] += temp
            del temp
            
            ''' Vector-to-scalar components '''
            spin2 = (True, False)
            # idx1, idx2 = np.ix_(idx_sc, idx_lp)
            M_v2s = vec2comp(M_self[i:j, ..., idx_sc, idx_lp], axis=-1) # [Ng',*c,ps,(pl)]
            SHgrid = self.SphGrid.SHio_iso_upto(level, spin2, **kargs) # [Ng',N1(L)]
            temp = np.tensordot(M_v2s, SHgrid.conj(),
                                axes=(0,0)) # [*c,ps,N1(L)]
            del M_v2s, SHgrid
            ## DomType.ISOBI complex -> DomType.ISOBI2 vector
            temp = ISOBI_doubling(temp, spin2, sh_type=sh_type) # [*c,ps,N2(L),pl]
            temp = np.moveaxis(temp, -3, -2) # [*c,N2(L),ps,pl]
            M_res[..., idx_sc, idx_lp] += temp
            del temp
            
            ''' Scalar-to-vector components '''
            spin2 = (False, True)
            # idx1, idx2 = np.ix_(idx_lp, idx_sc)
            M_s2v = vec2comp(M_self[i:j, ..., idx_lp, idx_sc], axis=-2) # [Ng',*c,(pl),ps]
            SHgrid = self.SphGrid.SHio_iso_upto(level, spin2, **kargs) # [Ng',N1(L)]
            temp = np.tensordot(M_s2v, SHgrid.conj(),
                                axes=(0,0)) # [*c,ps,N1(L)]
            del M_s2v, SHgrid
            ## DomType.ISOBI complex -> DomType.ISOBI2 vector
            temp = ISOBI_doubling(temp, spin2, sh_type=sh_type) # [*c,ps,N2(L),pl]
            temp = np.moveaxis(temp, -3, -1) # [*c,N2(L),pl,ps]
            M_res[..., idx_lp, idx_sc] += temp
            del temp
            
            ''' Vector-to-vector components '''
            spin2 = (True, True)
            # idx1, idx2 = np.ix_(idx_lp, idx_lp)
            M1_v2v, M2_v2v = mat2comppair(M_self[i:j, ..., idx_lp, idx_lp],
                                          axes=(-2,-1)) # [Ng',*c,(pl),(pl)] for each
            SHgrid1, SHgrid2 = self.SphGrid.SHio_iso_upto(
                               level, spin2, **kargs) # [Ng',N1(L)] for each
            temp1 = np.tensordot(M1_v2v, SHgrid1.conj(),
                                axes=(0,0)) # [*c,N1(L)]
            del M1_v2v, SHgrid1
            temp2 = np.tensordot(M2_v2v, SHgrid2.conj(),
                                axes=(0,0)) # [*c,N1(L)]
            del M2_v2v, SHgrid2
            ## DomType.ISOBI -> DomType.ISOBI2
            temp = ISOBI_doubling((temp1, temp2),
                                          spin2, sh_type=sh_type) # [*c,N2(L),pl,pl]
            del temp1, temp2
            M_res[..., idx_lp, idx_lp] += temp
            del temp
        
        shm_res = SHMat(M_res, dom_type, cod_type, sh_type)
        if not quiet:
            print(f"## [Function end] time: {time()-t} seconds.")
        return shm_res
    


def min_codom(a: CodType|ArrayLike|StokesField|MuellerField,
              b: CodType|ArrayLike|StokesField|MuellerField
             )-> CodType:
    if type(a) == np.ndarray:
        a = a.shape[-1]
    elif type(a) in [StokesField, MuellerField]:
        a = a.cod_type
    if type(b) == np.ndarray:
        b = b.shape[-1]
    if type(b) in [StokesField, MuellerField]:
        b = b.cod_type
    assert isinstance(a, CodType)
    assert isinstance(b, CodType)
    return min(a, b)

def max_codom(A: CodType|StokesField|MuellerField,
              B: CodType|StokesField|MuellerField
             )-> CodType:
    if type(A) in [StokesField, MuellerField]:
        A = A.cod_type
    if type(B) in [StokesField, MuellerField]:
        B = B.cod_type
    assert isinstance(A, CodType)
    assert isinstance(A, CodType)
    return max(A, B)

def truncate_codom_to(A:        ArrayLike|StokesField,
                      cod_type: CodType
                     )       -> ArrayLike|StokesField:
    '''
    Convert Stokes vector array into a smaller codomain type
    
    params:
        A: ArrayLike[*,p] | StokesField having .Stk[*,p]
        cod_type: CodType
    return:
        res: ArrayLike[*]          | StokesField having .Stk[*]
             ArrayLike[*,cod_type] | StokesField having .Stk[*,cod_type],
        res with * if `cod_type == CodType.SCALAR`, *xcod_type otherwise.

    NOTE This method returns a reference, which requires to be careful when modifying the values
    '''
    #* Preprocess
    assert type(A) in [np.ndarray, StokesField]
    cod_type = CodType(cod_type)
    if type(A) == np.ndarray:
        A_Stk = A
        A_cod = CodType(A.shape[-1])
    else: #* type(A) == StokesField
        A_Stk = A.Stk
        A_cod = A.cod_type
    assert isinstance(A_cod, CodType)
    
    #* Main
    assert cod_type <= A_cod, f"A (currently having codomain type {A_cod}) can be truncated " \
                              f"into smaller codomain type, but currently cod_type=={cod_type}."
    if A_cod == CodType.POLAR2: # 2
        assert cod_type == CodType.POLAR2, "CodType.POLAR2 cannot be truncated into CodType.SCALAR."
        res_Stk = A_Stk
    else:
        if cod_type == CodType.SCALAR:
            res_Stk = A_Stk[...,0]
        elif cod_type == CodType.POLAR2:
            res_Stk = A_Stk[...,1:3]
        else: #* cod_type in [CodType.POLAR3, CodType.POLAR4]
            res_Stk = A_Stk[...,0:cod_type]

    #* Postprocess
    if type(A) == np.ndarray:
        return res_Stk
    else: #* type(A) == StokesField
        return StokesField(A.SphFF, res_Stk)
    
def truncate_codom(A:        ArrayLike|StokesField,
                   B:        ArrayLike|StokesField
                  )       -> Tuple[ArrayLike|StokesField]:
    cod_type = min_codom(A, B)
    A = truncate_codom_to(A, cod_type)
    B = truncate_codom_to(B, cod_type)
    return A, B


##################################################
### py:class:StokesConvKernel
##################################################
class StokesConvKernel:
    def __init__(self, func: Callable[[float], ArrayLike]):
        """
        func(x[*]) -> [*, *c, p, p], radian only
        """
        M0 = func(0)
        self.shape_chan = M0.shape[:-2]
        self.ndim_chan = len(self.shape_chan)
        assert M0.shape[-2] == M0.shape[-1]
        self.cod_type = CodType(M0.shape[-1])
        assert int(self.cod_type) >= int(CodType.POLAR2)

        assert np.allclose(M0[..., self.cidx_lpolar(), self.cidx_scalar()], 0)
        assert np.allclose(M0[..., self.cidx_scalar(), self.cidx_lpolar()], 0)
        iso, conj = mat2comppair(M0[..., self.cidx_lpolar(), self.cidx_lpolar()])
        assert np.allclose(conj, 0)

        Mpi = func(np.pi)
        assert M0.shape == Mpi.shape

        assert np.allclose(Mpi[..., self.cidx_lpolar(), self.cidx_scalar()], 0)
        assert np.allclose(Mpi[..., self.cidx_scalar(), self.cidx_lpolar()], 0)
        iso, conj = mat2comppair(Mpi[..., self.cidx_lpolar(), self.cidx_lpolar()])
        assert np.allclose(iso, 0)

        self.func = func

    def __repr__(self):
        return "StokesConvKernel[\n" + \
              f"  cod_type = {repr(self.cod_type)},\n" + \
              f"  shape_chan = {self.shape_chan},\n" + \
               "]"
    
    def __call__(self, theta: ArrayLike) -> np.ndarray:
        theta = np.asarray(theta)
        shape_input = theta.shape
        res = self.func(theta)
        assert res.shape == shape_input + self.shape_chan + (int(self.cod_type), int(self.cod_type)), f"{res.shape=}, {shape_input=}, {self=}"
        return res

    def cidx_scalar(self):
        return cidx_scalar(self.cod_type)
    def cidx_lpolar(self):
        return cidx_lpolar(self.cod_type)
    
    def at(self, theta):
        return self(theta)

    def apply_delta(self,                        # [*c1, p]
                    F:         ArrayLike,        # [3, 3]
                    Stk:       ArrayLike,        # [*c2, p]
                    sphMF_out: Union[SphereGrid, SphereFrameField] # [*g]
                   ) ->        StokesField:       # [*g, (*c1|*c2), p], StokesField(sphMF_out, Stk_res)
        """
        `res.shape_chan = np.broadcast_shapes(self.shape_chan, Stk.shape[:-1])`
        """
        p = int(self.cod_type)
        F = np.asarray(F)
        Stk = np.asarray(Stk)
        assert np.allclose(orthogonal_error(F), 0), f"Invalid input: {orthogonal_error(F)=}\n\t{F}"
        assert Stk.shape[-1] == p, f"Invalid shape: {self.cod_type=}, {Stk.shape=}"
        if isinstance(sphMF_out, SphereGrid):
            sphMF_out = sphMF_out.ThetaPhiFrameField()
        assert isinstance(sphMF_out, SphereFrameField)

        # ---------- Main 1/2 ----------
        # [*g]
        sphMF_out_rotinv = sphMF_out.apply_rotation(np.linalg.inv(F))
        sphMF_rotinv_can = sphMF_out_rotinv.SphGrid.ThetaPhiFrameField()
        sphG_rotinv = sphMF_rotinv_can.SphGrid
        rotinv_theta, rotinv_phi = radian_regularize(sphG_rotinv.ang_type, sphG_rotinv.theta_grid, sphG_rotinv.phi_grid)

        # ---------- Assertion: array shapes ----------
        # [*g, *c1, p, p]
        M = self(rotinv_theta)
        ndim_grid = len(sphMF_out.shape)
        assert M.shape[ndim_grid:-2] == self.shape_chan, \
            f"Invalid shape change: {self.shape_chan=}, {sphMF_out.shape=}, {M.shape=}."
        assert M.shape[-2:] == (p, p), \
            f"Invalid shape change: {self.cod_type=}, {M.shape=}."
        
        # ---------- Array broadcast ----------
        shape_chan = np.broadcast_shapes(self.shape_chan, Stk.shape[:-1]) # [*c] = [(*c1|*c2)]
        ndim_bdct = len(shape_chan) - (M.ndim - ndim_grid - 2)
        M = np.expand_dims(M, tuple(range(ndim_grid, ndim_grid + ndim_bdct)))
        M = np.broadcast_to(M, sphMF_out.shape + shape_chan + (p, p)) # [*c, p, p]
        Stk = np.broadcast_to(Stk, shape_chan + (p,)) # [*c, p]

        # ---------- Main 2/2 ----------
        M_frame_conv = stokes_frame_convert(np.eye(3), rotz(rotinv_phi)) # [*g, p, p]
        Stk_res = np.moveaxis(
            np.tensordot(M_frame_conv, Stk, axes=(-1,-1)), # [*g, p, *c]
            -Stk.ndim, -1
        ) # [*g, *c, p]
        Stk_res = matmul_vec1d(M, Stk_res) # [*g, *c, p]
        stkG_rotinv = StokesField(sphMF_rotinv_can, Stk_res).to_SphFF(sphMF_out_rotinv)
        return StokesField(sphMF_out, stkG_rotinv.Stk)

    def apply(self,
              stkG:  StokesField,
              quiet: Optional[bool] = True
             ) ->   StokesField:
        sphG = stkG.SphGrid
        sphMF = stkG.SphFF
        assert sphG.dom_type == DomType.UNI

        first = True
        iterator = np.ndenumerate(sphG.theta_grid)
        if not quiet:
            iterator = tqdm(iterator)
        for idx_tup, _ in iterator:
            curr = self.apply_delta(sphMF.F[idx_tup], stkG.Stk[idx_tup], sphMF).wrt_SphFF(sphMF)
            curr *= sphG.weight()[idx_tup]
            if first:
                Stk_res = curr
            else:
                Stk_res += curr
            first = False
        return StokesField(sphMF, Stk_res)

    def SHCoeff(self,
                level:     int,
                n_sample:  Optional[int] = None
               ) ->        SHConv:
        if n_sample is None:
            n_sample = 4*(level**2)
        theta = np.linspace(0, np.pi, n_sample) # [n_sample]
        dtheta = interval_length(theta, 0, np.pi)
        weight = 2*np.pi*np.sin(theta)*dtheta
        
        M = self(theta) # [n_sample, *c, p, p]

        # [n_sample, N(level)]
        s0_SHval = SH_upto(theta, 0, level, False, sh_type=SHType.COMP)
        s2_SHval = SH_upto(theta, 0, level, True, sh_type=SHType.COMP)

        l, m = level2lms(level, DomType.UNI, unstack=True)
        Z = np.zeros((n_sample, 1))
        def extend_m2(arr):
            return np.concatenate([Z, Z, arr], axis=-1)
        
        # [n_sample, level]
        assert np.allclose(s0_SHval[:, m == 0].imag, 0)
        s0m0_SHval = s0_SHval[:, m == 0].real
        s0mn2_SHval = extend_m2(s0_SHval[:, m == -2])
        s2m0_SHval = s2_SHval[:, m == 0]
        s2mn2_SHval = extend_m2(s2_SHval[:, m == -2])
        s2mp2_SHval = extend_m2(s2_SHval[:, m == 2])
        assert {(n_sample, level)} == {x.shape for x in
                                       [s0m0_SHval, s0mn2_SHval, s2m0_SHval, s2mn2_SHval, s2mp2_SHval]}

        """
        coeff_s2s:  ArrayLike, # [*c, level] | [*c, level, 2, 2]
        coeff_s2v:  ArrayLike, # [*c, level] | [*c, level, 2] complex
        coeff_v2s:  ArrayLike, # [*c, level] | [*c, level, 2] complex
        coeff_v2va: ArrayLike, # [*c, level] complex
        coeff_v2vb: ArrayLike,  # [*c, level] complex
        """
        def get_coeff(SHval, M_part, p_axes = None):
            """
            params:
                SHval[n_sample, level]
                M_part[n_sample, *c, (2, 2)]
                p_axes: (-1,) or (-2, -1)
            return:
                [*c, level, (2, 2)]
            """
            if p_axes is None:
                res = np.einsum('nl, n..., n -> ...l', SHval.conj(), M_part, weight)
            elif p_axes in [-1, (-1,)]:
                res = np.einsum('nl, n...p, n -> ...lp', SHval.conj(), M_part, weight)
                if self.cod_type == CodType.POLAR3:
                    res = res.squeeze(-1)
            elif p_axes in [(-1, -2), (-2, -1)]:
                res = np.einsum('nl, n...pq, n -> ...lpq', SHval.conj(), M_part, weight)
                if self.cod_type == CodType.POLAR3:
                    res = res.squeeze((-1, -2))
            return res

        coeff_s2s = get_coeff(s0m0_SHval, M[..., self.cidx_scalar(), self.cidx_scalar()], (-2,-1))
        coeff_s2v = get_coeff(s2m0_SHval, vec2comp(M[..., self.cidx_lpolar(), self.cidx_scalar()], -2), -1)
        coeff_v2s = get_coeff(s0mn2_SHval, vec2comp(M[..., self.cidx_scalar(), self.cidx_lpolar()]), -1)
        M_v2va, M_v2vb = mat2comppair(M[..., self.cidx_lpolar(), self.cidx_lpolar()])
        coeff_v2va = get_coeff(s2mn2_SHval, M_v2va)
        coeff_v2vb = get_coeff(s2mp2_SHval, M_v2vb)
    
        return SHConv(self.cod_type,
                      coeff_s2s = coeff_s2s,
                      coeff_s2v = coeff_s2v,
                      coeff_v2s = coeff_v2s,
                      coeff_v2va = coeff_v2va,
                      coeff_v2vb = coeff_v2vb,
                      weighted = False)

from __future__ import annotations
from time import time
from typing import Type, List, Optional, Sequence, Callable
import math
from tqdm import tqdm
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import SphericalVoronoi
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial.transform import Rotation as scipyRotation
import quaternionic as quat

import vispy
import vispy.scene
import vispy.geometry

from polarsh.array import *
from polarsh.sphere import *
from polarsh.SH import *
from polarsh.SHCoeff import SHCoeff
from polarsh.image import *
from polarsh import SHVec, SHMat, SHConv
from ..SHCoeff import SH_indexing
import polarsh.plot as plot


class SphereGrid:
    """
    The reason of naming "Spherical" instead of "Sphere" is
      to indicate "spherical coordinates" rather than "sphere" itself,
      which is a geometric object invariant over coordinates representation.
    NOTE: Two conventions for incident directions:
             (1) m2s: material to source,
                      more common in computer graphics when we does not consider polarization
             (2) s2m: source to material == direction of light propagation,
                      important for polarimetric states
    NOTE: Independently to use of incident direction convention,
             the half-way vectors always indicates the half-way of "vec_i_m2s" and vec_o.
    NOTE: For DomType.ISOBI(2), phii_m2s == phi_d and phio == 0
    """

    def __init__(self,
                 dom_type: DomType,
                 ang_type: AngType,
                 *params:  Sequence[ArrayLike]):
        """
        Argument `params` (sequence of broadcastable arrays) depends the value of `dom_type`:
        (1) `SphereGrid(DomType.UNI, ang_type, theta, phi)`
        (2) `SphereGrid(DomType.ISOBI, ang_type, phi_d, theta_d, theta_h)`
        (3) `SphereGrid(DomType.UNI, ang_type, theta_i, phi_i, theta_o, phi_o)` NOTE [NOT IMPLEMENTED]
        """
        assert isinstance(dom_type, DomType), f"dom_type (currently {dom_type}) should be a DomType."
        self.dom_type = dom_type if dom_type != DomType.ISOBI2 else DomType.ISOBI
        assert isinstance(ang_type, AngType), f"ang_type (currently {ang_type}) should be a AngType."
        self.ang_type = ang_type

        self.grid_list = np.broadcast_arrays(*params)
        self.shape = self.grid_list[0].shape
        self.axes = tuple(range(len(self.shape)))
        self.ndim = len(self.shape)

        # ---------- Cached attributes ----------
        """
        Attributes below will have non-None value after self.weight(),
        self.ThetaPhiFrameField(), self.GeodesicFrameField(), self.RusinkiewiczFrameField(),
        self.CubemapFrameField(), self.TD17FrameField() are called, respectively.
        """
        self._weight: Union[None, SphereFrameField] = None
        self._canMF: Union[None, SphereFrameField] = None
        self._geoMF: Union[None, SphereFrameField] = None
        self._rusMF: Union[None, SphereFrameField] = None
        self._cubeMF: Union[None, SphereFrameField] = None
        self._TD17MF: Union[None, SphereFrameField] = None

        """
        `self._unrotated is not None` only if `self` was constructed by `SphereGrid.apply_rotaion()`
        Conversely, the result of `self.apply_rotation()` will be stored in `self.__cached_rotated`
        """
        self._unrotated: Union[None, SphereGrid] = None
        self.__cache_rotated: dict[Tuple, SphereGrid] = {}
        # ----------------------------------------

        self.VisObj_list = [] # for `ScalarField.visualization()` and `StokesField.visualization()`
        if dom_type == DomType.UNI:
            self.theta_grid = self.grid_list[0] # [*]
            self.phi_grid = self.grid_list[1]   # [*]
            self.vec = sph2vec(self.theta_grid, self.phi_grid, ang_type=ang_type)
        elif dom_type == DomType.BI:
            raise NotImplementedError()
        else: # DomType.ISOBI
            self.phid_grid = self.grid_list[0]   # [*]
            self.thetad_grid = self.grid_list[1] # [*]
            self.thetah_grid = self.grid_list[2] # [*]

            sph_params = rus2sph(self.phid_grid, self.thetad_grid, self.thetah_grid,
                                 ang_type=ang_type)
            assert len(sph_params) == 3
            self.thetai_m2s, self.thetao, self.phii_m2s = sph_params
            self.phio = np.zeros(self.phii_m2s.shape)

            self.veci_m2s = sph2vec(self.thetai_m2s, self.phii_m2s, ang_type=ang_type)
            self.veco = sph2vec(self.thetao, self.phio, ang_type=ang_type)

            ## Convert incident m2s to s2m
            pi = np.pi if ang_type == AngType.RAD else 180.0
            self.thetai_s2m = pi - self.thetai_m2s
            self.phii_s2m = self.phii_m2s + pi
            self.veci_s2m = -self.veci_m2s
            
    
    @classmethod
    def from_meshgrid(cls,
                      dom_type: DomType,
                      ang_type: AngType,
                      *params:  Sequence[ArrayLike]
                     ) ->       SphereGrid:
        """
        Argument `params` (sequence of 1D arrays) depends the value of `dom_type`:
        (1) `SphereGrid(DomType.UNI, ang_type, theta, phi)`
        (2) `SphereGrid(DomType.ISOBI, ang_type, phi_d, theta_d, theta_h)`
        (3) `SphereGrid(DomType.UNI, ang_type, theta_i, phi_i, theta_o, phi_o)` NOTE [NOT IMPLEMENTED]
        """
        params = list(params)
        for i in range(len(params)):
            params[i] = np.asarray(params[i])
            if params[i].ndim != 1:
                raise ValueError(f"Given shape of params[{i}] is {params[i].shape}, but it should have `ndim == 1`.")
        params_grid = np.meshgrid(*params, indexing='ij')
        SphGrid = cls(dom_type, ang_type, *params_grid)

        # ---------- Compute weights ----------        
        if dom_type == DomType.UNI:
            th_rad, ph_rad = radian_regularize(ang_type, *params_grid)
            th_range, ph_range = radian_regularize(ang_type, *params)

            dth = interval_length(th_range, 0, np.pi) # TODO: generalize to cyclic shifting
            dph = interval_length(ph_range, 0, 2*np.pi)
            dth, dph = np.meshgrid(dth, dph, indexing='ij')
            Jacob = np.sin(th_rad)
            SphGrid._weight = Jacob*dth*dph
        elif dom_type == DomType.BI:
            raise NotImplementedError()
        else: # dom_type in [DomType.ISOBI, DomType.ISOBI2]
            ## NOTE It computes J(subspace measure from R^6 / phi_d, theta_d, theta_h),
            ##      2pi from int_0^2pi{d phi_h} should be computed outside of here,
            ##      such as `SphereGrid.SHio_iso_upto`
            pd_rad, td_rad, th_rad = radian_regularize(ang_type, *params_grid)
            pd_range, td_range, th_range = radian_regularize(ang_type, *params)

            dpd = interval_length(pd_range, -np.pi, np.pi) # TODO: generalize to cyclic shifting
            dtd = interval_length(td_range, 0, np.pi/2)
            dth = interval_length(th_range, 0, np.pi/2) # NOTE currently BRDF only, TODO: generalize to BSDF
            dpd, dtd, dth = np.meshgrid(dpd, dtd, dth, indexing='ij')
            Jacob = 4*np.sin(th_rad)*np.sin(td_rad)*np.cos(td_rad)
            SphGrid._weight = Jacob*dpd*dtd*dth
        return SphGrid
    

    @property
    def x(self) -> np.ndarray:
        return self.vec[..., 0]
    @property
    def y(self) -> np.ndarray:
        return self.vec[..., 1]
    @property
    def z(self) -> np.ndarray:
        return self.vec[..., 2]
    
    def __repr__(self) -> str:
        result = "SphereGrid[\n"
        result += f"  dom_type = {repr(self.dom_type)},\n"
        result += f"  shape = [{str(self.shape)[1:-1]}],\n"
        result += "]"
        return result

    def weight(self) -> np.ndarray: # *
        """
        DomType.UNI:   scipy.spatial.SphericalVoronoi
        DomType.BI:    Not implemented yet
        DomType.ISOBI: mesh grid assumption

        NOTE See MuellerField.dot_domain too
        """
        if self._weight is None:
            if self.dom_type == DomType.UNI:
                sphVor = SphericalVoronoi(self.vec.reshape(-1, 3))
                W = sphVor.calculate_areas().reshape(self.shape)
            elif self.dom_type == DomType.BI:
                raise NotImplementedError()
            else: # self.dom_type in [DomType.ISOBI, DomType.ISOBI2]
                raise NotImplementedError()
            self._weight = W
        return self._weight

    def copy_weight(self, SphGrid):
        assert self.shape == SphGrid.shape, f"Two SphereGrid instances should have the same shape, but currently: {self.shape}, {SphGrid.shape}"
        if SphGrid._weight is not None:
            self._weight = SphGrid.weight()

    def apply_rotation(self, rotation: RotationLike) -> SphereGrid:
        """
        The result will be cached
        """
        R = rotation2quat(rotation)
        rotvec = rotation if np.asarray(rotation).shape == (3,) else R.to_rotation_vector
        rotvec_tup = tuple(rotation) if np.asarray(rotation).shape == (3,) else tuple(R.to_rotation_vector)
        rotation = R.to_rotation_matrix
        
        if not rotvec_tup in self.__cache_rotated:
            if self.dom_type == DomType.UNI:
                vec_f = matmul_vec1d(rotation, self.vec)
                theta_f, phi_f = vec2sph(vec_f, ang_type=self.ang_type)
                SphGrid = SphereGrid(self.dom_type, self.ang_type, theta_f, phi_f)
                SphGrid.copy_weight(self)
                for visobj in self.VisObj_list:
                    SphGrid.VisObj_list.append(visobj.apply_rotation(rotation))
                SphGrid._unrotated = self
            else:
                raise NotImplementedError()
            self.__cache_rotated[rotvec_tup] = SphGrid
        return self.__cache_rotated[rotvec_tup]


    def ThetaPhiFrameField(self, quiet: bool=True) -> SphereFrameField:
        if self._canMF is None: 
            if self.dom_type == DomType.UNI:
                F = sph2Ftp(self.theta_grid, self.phi_grid, vec=self.vec, ang_type=self.ang_type)
                self._canMF = SphereFrameField(self, F, quiet=quiet)
            else: # self.dom_type == DomType.BI or DomType.ISOBI:
                Fi = sph2Ftp(self.thetai_s2m, self.phii_s2m, vec=self.veci_s2m, ang_type=self.ang_type)
                Fo = sph2Ftp(self.thetao, self.phio, vec=self.veco, ang_type=self.ang_type)
                self._canMF = SphereFrameField(self, Fi, Fo, quiet=quiet)
        return self._canMF

    def GeodesicFrameField(self, quiet: bool=True) -> SphereFrameField:
        '''
        Moving frame obtained using parallel transport from the zenith
        along geodesics
        '''
        if self._geoMF is None: 
            if self.dom_type == DomType.UNI:
                F = sph2Fgeo(self.theta_grid, self.phi_grid)
                self._geoMF = SphereFrameField(self, F, quiet=quiet)
            else: # self.dom_type == DomType.BI or DomType.ISOBI:
                raise NotImplementedError()
        return self._geoMF
    
    def RusinkiewiczFrameField(self, quiet: bool=True) -> SphereFrameField:
        '''
        Moving_frame
        [SB 2018]
        '''
        if not (hasattr(self, 'phid_grid') and hasattr(self, 'thetad_grid') and hasattr(self, 'thetah_grid')):
            raise ValueError(f"Fields `phid_grid`, `thetad_grid`, and `thetah_gird` must be exist.")
        if self._rusMF is None:
            Fo = rotz(-np.pi/2)
            Fi = roty(np.pi) @ Fo
            Fi = roty(self.thetad_grid) @ Fi
            Fo = roty(-self.thetad_grid) @ Fo
            Rhd = roty(self.thetah_grid) @ rotz(self.phid_grid)
            if hasattr(self, 'phih_grid'):
                Rhd = rotz(self.phih_grid) @ Rhd
            Fi = Rhd @ Fi
            Fo = Rhd @ Fo
            self._rusMF = SphereFrameField(self, Fi, Fo, quiet=quiet)
        return self._rusMF

    def PerspectiveFrameField(self, up: ArrayLike, quiet: Optional[bool]=True) -> SphereFrameField:
        '''
        See the docstring of `SphereFrameField_args_from_persp` in this file.
        NOTE this moving frame yields singular points on every point in a great circle. Only useful for partial solid angles.
        '''
        if self.dom_type != DomType.UNI:
            raise NotImplementedError()
        up = np.asarray(up).squeeze()
        if up.shape != (3,):
            raise ValueError(f"Invaild shape of `up` ({up.shape}). It must be (3,).")
        Fx = normalize(np.cross(up, self.vec))
        Fy = normalize(np.cross(self.vec, Fx))
        
        F = np.stack([Fx, Fy, self.vec], -1)
        return SphereFrameField(self, F, quiet=quiet)
    
    def CubemapFrameField(self, quiet: Optional[bool]=True) -> SphereFrameField:
        """
        TODO: Add an argument `to_world`.
        """
        if self._cubeMF is None:
            if self.dom_type != DomType.UNI:
                raise NotImplementedError()
            F = vec2Fcube(self.vec)
            self._cubeMF = SphereFrameField(self, F, quiet=quiet)
        return self._cubeMF
    
    def TD17FrameField(self) -> SphereFrameField:
        """
        A numerically stable moving frame.

        Reference:
        * "Building an Orthonormal Basis, Revisited" by
            Tom Duff, James Burgess, Per Christensen, Christophe Hery,
            Andrew Kensler, Max Liani, and Ryusuke Villemin
            (JCGT Vol 6, No 1, 2017)
        * func:`coordinate_system` in
            https://github.com/mitsuba-renderer/mitsuba3/blob/master/include/mitsuba/core/vector.h
        """
        if self._TD17MF is None:
            if self.dom_type == DomType.UNI:
                F = vec2FTD17(self.vec)
                self._TD17MF = SphereFrameField(self, F)
            else:
                Fi = vec2FTD17(self.veci_s2m)
                Fo = vec2FTD17(self.veco)
                self._TD17MF = SphereFrameField(self, Fi, Fo)
        return self._TD17MF

    def SH_upto(self,          # *g := self.shape
                level:    int, # =: L
                spin2:    bool,
                sh_type:  SHType = SHType.COMP,
                rotated:  Optional[ArrayLike] = None,
                cfg_pass: Optional[Tuple[int,int]] = None, # (i_pass: int, n_pass: int)
               ) ->       ArrayLike: # [*g,N(L)] | [|*g|,N(L)]
        if self.dom_type == DomType.UNI:
            theta, phi = slice_pass([self.theta_grid, self.phi_grid], cfg_pass)
            return SH_upto(theta, phi, level, spin2, sh_type, ang_type=self.ang_type, rotated=rotated)
        else:
            raise NotImplementedError()
    
    def SHi_upto(self,         # *g := self.shape
                 level:   int, # =: L
                 spin2:   bool,
                 sh_type: SHType = SHType.COMP,
                 cfg_pass:Tuple[int,int] = None # (i_pass: int, n_pass: int)
                )      -> ArrayLike: # [*g,N] | [|*g|,N] where N:=N(L, DUNI)
        '''
        NOTE Use s2m convention.
        '''
        assert self.dom_type != DomType.UNI
        theta, phi = slice_pass([self.thetai_s2m, self.phii_s2m], cfg_pass)
        return SH_upto(theta, phi, level, spin2, sh_type, ang_type=self.ang_type)
    
    def SHo_upto(self,         # *g := self.shape
                 level:   int, # =: L
                 spin2:   bool,
                 sh_type: SHType = SHType.COMP,
                 cfg_pass:Tuple[int,int] = None # (i_pass: int, n_pass: int)
                )      -> ArrayLike: # [*g,N] | [|*g|,N] where N:=N(L, DUNI)
        assert self.dom_type != DomType.UNI
        theta, phi = slice_pass([self.thetao, self.phio], cfg_pass)
        return SH_upto(theta, phi, level, spin2, sh_type, ang_type=self.ang_type)
    
    def SHi_phii0_upto(self,         # *g := self.shape
                       level:   int, # =: L
                       spin2:   bool,
                       sh_type: SHType = SHType.COMP,
                       cfg_pass:Tuple[int,int] = None # (i_pass: int, n_pass: int)
                      )      -> ArrayLike: # [*g,N] | [|*g|,N] where N:=N(L, DUNI)
        '''
        NOTE Use s2m convention.
        '''
        assert self.dom_type in [DomType.ISOBI, DomType.ISOBI2]
        theta = slice_pass([self.thetai_s2m], cfg_pass)[0]
        return SH_upto(theta, 0, level, spin2, sh_type, ang_type=self.ang_type)
    
    def SHo_phii0_upto(self,         # *g := self.shape
                       level:   int, # =: L
                       spin2:   bool,
                       sh_type: SHType = SHType.COMP,
                       cfg_pass:Tuple[int,int] = None # (i_pass: int, n_pass: int)
                      )      -> ArrayLike: # [*g,N] | [|*g|,N] where N:=N(L, DUNI)
        assert self.dom_type in [DomType.ISOBI, DomType.ISOBI2]
        theta, phi = slice_pass([self.thetao, -self.phii_s2m], cfg_pass)
        return SH_upto(theta, phi, level, spin2, sh_type, ang_type=self.ang_type)
    
    def SHio_iso_upto(self,         # *g := self.shape
                      level:   int, # =: L
                      spin2:   Tuple[bool,bool],
                      sh_type: SHType = SHType.COMP,
                      cfg_pass:Tuple[int,int] = None # (i_pass: int, n_pass: int)
                     )      -> ArrayLike | Tuple[ArrayLike, ArrayLike]: # [*g,N] | [|*g|,N]
        '''
        Return Y so that s2SH coefficient of a Mueller matrix M is:
            int <Y, M> d(phi_d, theta_d, theta_h)
        Complex spin2-SH w.r.t. the canonical moving frame.
        
        params:
            level:    int (=: L)
            spin2:    (bool, bool), incident and outgoing resp.
            sh_type:  SHType
            cfg_pass: None | (int, int) (=: i_pass, n_pass)
        return:
            Y:        ArrayLike[*r,N],                    if spin2 != (True,True)
            (Y1, Y2): (ArrayLike[*r,N], ArrayLike[*r,N]), if spin2 == (True,True)
                - *r == *g if cfg_pass is None, and *r == |*g| otherwise.
                - N := N(L, DomType.ISOBI) NOTE always `DomType.ISOBI` even if `self.dom_type == DomType.ISOBI2`.
                - When a 2x2 target Mueller matrix M is decomposed into
                  M = R^{2x2}a + R^{2x2}b @ [[1,0],[0,-1]],
                  Y1 and Y2 will be inner producted to a and b, resp.
        
        NOTE Refer to `2022.08.30. Factorizing 3-parameter BSDF.md`.
        '''
        assert self.dom_type in [DomType.ISOBI, DomType.ISOBI2]
        lms = level2lms(level, DomType.ISOBI)
        li, lo, m = lms[:,0], lms[:,1], lms[:,2]
        if spin2 != (False, True):
            Yi = self.SHi_upto(level, spin2[0], sh_type, cfg_pass)
            Yo = self.SHo_upto(level, spin2[1], sh_type, cfg_pass)
        else: # spin2 == (False, True)
            Yi = self.SHi_phii0_upto(level, spin2[0], sh_type, cfg_pass)
            Yo = self.SHo_phii0_upto(level, spin2[1], sh_type, cfg_pass)
        
        Yi_idxing = lambda l, m: SH_indexing(Yi, DomType.UNI, l, m)
        Yo_idxing = lambda l, m: SH_indexing(Yo, DomType.UNI, l, m)
        
        if spin2 == (False, False):
            ## Scalar to scalar
            if sh_type == SHType.COMP:
                ## (mi, mo) == (m, m)
                return 2*np.pi * Yi_idxing(li, m).conj() * Yo_idxing(lo, m)
            else:
                ## (mi, mo) == (m, |m|)
                Y = np.pi * Yi_idxing(li, m).conj() * Yo_idxing(lo, abs(m))
                Y[..., m == 0] *= 2
                return Y
        
        elif spin2 == (True, False):
            ## Vector to scalar
            if sh_type == SHType.COMP: raise NotImplementedError()
            ## (mi, mo) == (m, |m|)
            Y = np.pi * Yi_idxing(li, m) * Yo_idxing(lo, abs(m))
            Y[..., m == 0] *= 2
            return Y
            
        elif spin2 == (False, True):
            ## Scalar to vector
            if sh_type == SHType.COMP: raise NotImplementedError()
            ## (mi, mo) == (|m|, m)
            Y = np.pi * Yi_idxing(li, abs(m)) * Yo_idxing(lo, m)
            Y[..., m == 0] *= 2
            return Y
        
        else:
            ## Vector to vector
            assert spin2 == (True, True)
            ## NOTE independent of `sh_type`
            ## (mi, mo) == (m, m) for Y1, and == (m, -m) for Y2
            Y1 = 2*np.pi * Yi_idxing(li, m).conj() * Yo_idxing(lo, m)
            Y2 = 2*np.pi * Yi_idxing(li, m)        * Yo_idxing(lo, -m)
            return Y1, Y2

    
    def dsamp_index_exp(self, dsamp: int):
        """
        `np.index_exp` for downsampling the sphere grid.
        The primary usage is visualization.
        This method will be overrided.
        """
        if self._unrotated is None:
            idx_exp = ()
            for i in self.axes:
                idx_exp += np.index_exp[::dsamp]
        else:
            idx_exp = self._unrotated.dsamp_index_exp(dsamp)
        return idx_exp
    
    def visualize(self,
                  marker_size: Optional[float] = 3.0,
                  marker_dsamp: Optional[int] = 1,
                  title: Optional[str] = None,
                  help: Optional[bool] = None):
        canvas, view = plot.gen_vispy_canvas()
        sphere = vispy.scene.visuals.Sphere(radius=0.999999, parent=view.scene)
        axes, event_a = plot.gen_vispy_axes(view.scene)

        idx_exp = self.dsamp_index_exp(marker_dsamp)
        grid = vispy.scene.visuals.Markers(parent=view.scene)
        grid.set_data(self.vec[idx_exp].reshape(-1, 3),
                      edge_width=0,
                      face_color=(0., 0., 0., 0.8),
                      size=marker_size)

        @canvas.events.key_press.connect
        def on_key_press(event):
            if event.key == "a":
                event_a()
        
        if help is None:
            help = plot._vis_options_vispy['help']
        if help:
            print("[Keyboard interface]\n"
                "A: hide/show global axes")
        
        if title is not None:
            plot.vispy_attach_title(view, title)
        return canvas

class SphereGridEquirect(SphereGrid):
    __obj_cache = {}

    def __new__(cls, h: int, w: int, *args, **kargs):
        if (h, w) in cls.__obj_cache:
            return cls.__obj_cache[(h, w)]
        else:
            obj = super().__new__(cls)
            return obj
    
    def __init__(self, h: int, w: int):
        if (h, w) in self.__obj_cache:
            assert self is self.__obj_cache[(h, w)]
            return
        else:
            self.__obj_cache[(h, w)] = self
        
        self.h = h
        self.w = w
        self.theta_range, self.phi_range = linspace_tp(h, w)
        
        params_grid = np.meshgrid(self.theta_range, self.phi_range, indexing='ij')
        super().__init__(DomType.UNI, AngType.RAD, *params_grid)
        self.__compute_weight()
        self.VisObj_list.append(plot.visUVSphere)
    
    def __compute_weight(self):
        if not self._weight is None:
            raise RuntimeError("The weight is already computed.")
        dth = interval_length(self.theta_range, 0, np.pi) # TODO: generalize to cyclic shifting
        dph = interval_length(self.phi_range, 0, 2*np.pi)
        dth, dph = np.meshgrid(dth, dph, indexing='ij')
        Jacob = np.sin(self.theta_grid)
        self._weight = Jacob*dth*dph
    
    @classmethod
    def clear_cache(cls):
        cls.__obj_cache.clear()
    
        
class SphereGridMirrorball(SphereGrid):
    __obj_cache = {}

    def __new__(cls, size: int, *args, **kargs):
        if size in cls.__obj_cache:
            return cls.__obj_cache[size]
        else:
            obj = super().__new__(cls)
            return obj
    
    def __init__(self, size: int):
        '''
        To avoid NaN values, pixels outside of the ball mask have
        the same value as the nearest position inside the mask.
        '''
        if size in self.__obj_cache:
            assert self is self.__obj_cache[size]
            return
        else:
            self.__obj_cache[size] = self
        
        y, x = np.mgrid[1:-1:size*1j, -1:1:size*1j]
        r_sq = x**2 + y**2
        r_sq = np.where(r_sq > 1, 1, r_sq)
        z = np.sqrt(1-r_sq)
        theta_mir, phi_mir = vec2sph(np.stack([x, y, z], -1))
        
        super().__init__(DomType.UNI, AngType.RAD, theta_mir*2, phi_mir)
        self.size = size
    
    @classmethod
    def clear_cache(cls):
        cls.__obj_cache.clear()

class SphereGridPersp(SphereGrid):
    def __init__(self,
                 h: int,
                 w: int,
                 fov_x_deg: float,
                 to_world: ArrayLike = np.eye(3) # Transform
                ):
        theta, phi, _ = SphereFrameField_args_from_persp(h, w, fov_x_deg, to_world)
        super().__init__(DomType.UNI, AngType.RAD, theta, phi)
        self.h = h
        self.w = w
        self.to_world = to_world

        self.fov_x_deg = fov_x_deg
        self.fov_x_rad = np.deg2rad(fov_x_deg)
        self.fov_x_tan = np.tan(self.fov_x_rad/2)*2
        self.fov_y_tan = self.fov_x_tan * h/w
        self.fov_y_rad = np.arctan(self.fov_y_tan/2)*2
        self.fov_y_deg = np.rad2deg(self.fov_y_rad)

        self.__compute_weight()
        self.VisObj_list = self.__compute_VisObj_list()
    
    def __compute_weight(self):
        if not self._weight is None:
            raise RuntimeError("The weight is already computed.")
        '''
        NOTE # Derivation of equations
        sphere = {r in R3 | |r|=1}
        image plane = {(x,y,1)}
        projection: P(x,y)=(x,y,1)/|x,y,1| in sphere
        See the appropriate markdown document
        '''
        ### Cumulative sqrt(det J.T@J)
        CumDetJ = lambda x,y: np.arctan(x*y/np.sqrt(1+x*x+y*y))
        
        h, w = self.h, self.w
        i = colvec(np.arange(h))
        j = rowvec(np.arange(w))
        i,j = np.broadcast_arrays(i,j)

        x0 = 2/w*j - 1; x1 = 2/w*(j+1) - 1
        y0 = 2/h*i - 1; y1 = 2/h*(i+1) - 1
        x0 *= self.fov_x_tan/2; x1 *= self.fov_x_tan/2
        y0 *= self.fov_y_tan/2; y1 *= self.fov_y_tan/2

        W = CumDetJ(x1,y1)+CumDetJ(x0,y0)-CumDetJ(x0,y1)-CumDetJ(x1,y0)
        self._weight = np.broadcast_to(W, self.shape)
    
    def __compute_VisObj_list(self) -> List[plot.VisObj]:
        """
        Used in `ScalarField.visualize()`
        """
        def imageplane2sphere(vertices, to_world = np.eye(3), depth = 1):
            # vertices[N, 3]
            vert_res = vertices + np.array([0, 0, depth])
            vert_res = matmul_vec1d(to_world, vert_res)
            return normalize(vert_res, -1)
        create_plane = vispy.geometry.create_plane

        tan_x, tan_y = self.fov_x_tan, self.fov_y_tan
        z = int(np.ceil(self.w * 2 / self.fov_x_tan))
        vertices_hw, indices_hw, _ = create_plane(tan_x, tan_y, self.w, self.h)
        vertices_zw, indices_zw, _ = create_plane(tan_x, 2, self.w, z)
        vertices_hz, indices_hz, _ = create_plane(2, tan_y, z, self.h)
        
        delta_tf_list = [np.eye(3), roty(np.pi),
                            rotx(np.pi/2), rotx(-np.pi/2),
                            roty(np.pi/2), roty(-np.pi/2)]

        vertices_list = [vertices_hw, vertices_hw, vertices_zw, vertices_zw, vertices_hz, vertices_hz]
        indices_list = [indices_hw, indices_hw, indices_zw, indices_zw, indices_hz, indices_hz]
        depth_list = np.repeat([1, tan_y/2, tan_x/2], 2)

        return [plot.VisObj(imageplane2sphere(vertices['position'], self.to_world @ tf, depth),
                                              indices,
                                              vertices['texcoord'])
                for tf, vertices, indices, depth
                in zip(delta_tf_list, vertices_list, indices_list, depth_list)]

class SphereGridCube(SphereGrid):
    __obj_cache = {}

    def __new__(cls, edge: int, *args, **kargs):
        if edge in cls.__obj_cache:
            return cls.__obj_cache[edge]
        else:
            obj = super().__new__(cls)
            return obj
    
    def __init__(self, edge: int):
        '''
        env_tfs:
            Up
        Left Front Right Back
            Down
        world x,y,z == Right, Front, Up
        '''
        ## 'env_tf' is a transform which
        ##          maps frame generated by 'SphereFrameField_from_persp'
        ##          to env. frame describe above
        if edge in self.__obj_cache:
            assert self is self.__obj_cache[edge]
            return
        else:
            self.__obj_cache[edge] = self
        
        thetas = []; phis = [] #; Fs = []
        for row in env_tfs:
            for env_tf in row:
                if env_tf is None:
                    continue
                theta, phi, F = SphereFrameField_args_from_persp(edge, edge, 90, env_tf)
                thetas.append(theta); phis.append(phi) #; Fs.append(F)
        
        super().__init__(DomType.UNI, AngType.RAD, np.stack(thetas), np.stack(phis))
        self.edge = edge
        self.__compute_weight()
        self.VisObj_list.append(plot.visCubeSphere)
    
    def dsamp_index_exp(self, dsamp: int):
        """
        `np.index_exp` for downsampling the sphere grid.
        The primary usage is visualization.
        """
        return np.index_exp[:, ::dsamp, ::dsamp]
    
    def __compute_weight(self): # Store the weight attr. only if it is not computed yet.
        if not self._weight is None:
            raise RuntimeError("The weight is already computed.")
        '''
        NOTE # Derivation of equations
        sphere = {r in R3 | |r|=1}
        image plane = {(x,y,1)}
        projection: P(x,y)=(x,y,1)/|x,y,1| in sphere
        See the appropriate markdown document
        '''
        ### Cumulative sqrt(det J.T@J)
        CumDetJ = lambda x,y: np.arctan(x*y/np.sqrt(1+x*x+y*y))
        
        h, w = self.edge, self.edge
        i = colvec(np.arange(h))
        j = rowvec(np.arange(w))
        i,j = np.broadcast_arrays(i,j)
        x0 = 2/w*j - 1; x1 = 2/w*(j+1) - 1
        y0 = 2/h*i - 1; y1 = 2/h*(i+1) - 1
        W = CumDetJ(x1,y1)+CumDetJ(x0,y0)-CumDetJ(x0,y1)-CumDetJ(x1,y0)
        self._weight = np.broadcast_to(W, self.shape)
    
    @classmethod
    def clear_cache(cls):
        cls.__obj_cache.clear()
    


class SphereGridFibonacci(SphereGrid):
    __obj_cache = {}

    def __new__(cls, n_samples: int, *args, **kargs):
        if n_samples in cls.__obj_cache:
            return cls.__obj_cache[n_samples]
        else:
            obj = super().__new__(cls)
            return obj

    def __init__(self, n_samples: int):
        if n_samples in self.__obj_cache:
            assert self is self.__obj_cache[n_samples]
            return
        else:
            self.__obj_cache[n_samples] = self
        
        theta, phi = fibonacci_sphere(n_samples, out_type='sph')
        super().__init__(DomType.UNI, AngType.RAD, theta, phi)
        self.n_samples = n_samples
    
    @classmethod
    def clear_cache(cls):
        cls.__obj_cache.clear()


def cat_adaptive(param_list: List):
    typ = consistent_type(param_list)
    
    if typ == type(None):
        return None
    elif typ == np.ndarray:
        return np.stack(param_list)
    elif typ == SphereGrid:
        return SphereGrid_cat(param_list)
    else:
        raise TypeError(f"Invalid type {typ}.")


def SphereGrid_cat(SGrids: List) -> SphereGrid:
    for i,SGrid in enumerate(SGrids):
        if i == 0:
            dom_type = SGrid.dom_type
            ang_type = SGrid.ang_type
            params_list = [SGrid.grid_list]
        else:
            assert dom_type == SGrid.dom_type, "Domain types of arguments must be same."
            params_curr = SGrid.grid_list
            params_curr = angle_convert(SGrid.ang_type, ang_type, *params_curr)
            params_list.append(params_curr)
    ## params_list[idx_array][idx_param], idx_param = THETA|PHI, ...
    
    params_res = [np.stack(param_list) for param_list in zip(*params_list)]
    return SphereGrid(dom_type, ang_type, *params_res)

class SphereFrameField:
    '''
    SphGrid: SphereGrid
    shape:   Tuple == SphGrid.shape =: *
    Fi:      ArrayLike[*,3,3]
    Fo:      ArrayLike[*,3,3]

    -----
    NOTE
    Q. Why isn't it a subclass of `SphereGrid`?:
    A. Determining two sets of floating-point numbers is known as not simple.
       Current implementation enable comparing sphere grids of moving frames by reference, i.e.
       `smf1.SphGrid is smf2.SphGrid`
    '''
    def __init__(self,
                 SphGrid:   SphereGrid, # SphGrid.params[i].shape: *
                 Fi:        ArrayLike,  # *x3x3
                 Fo:        ArrayLike = None,  # *x3x3
                 allow_nan: bool = False,
                 quiet:     bool = True):
        #* F is a frame, linear map (local coord. vec.) -> (geom. vec. == world coord. vec.)
        assert isinstance(SphGrid, SphereGrid), f"SphGrid should be an instance of SphereGrid, but currently: {type(SphGrid)}"
        self.SphGrid: SphereGrid = SphGrid
        self.shape: Tuple = SphGrid.shape
        self.ndim = len(self.shape)

        param0 = np.expand_dims(SphGrid.grid_list[0], axis=(-1,-2))
        if Fo is None:
            assert SphGrid.dom_type == DomType.UNI
            _param0, Fi = np.broadcast_arrays(param0, Fi)
        else:
            assert SphGrid.dom_type != DomType.UNI
            _param0, Fi, Fo = np.broadcast_arrays(param0, Fi, Fo)
        assert param0.shape[:-2] == _param0.shape[:-2], f"Broadcasing a grid parameter (currently {param0.shape}) and F(s) should not change the shape of the grid param, but it changed to {_param0.shape}."
        
        if Fo is None:
            assert_frame(Fi, name="F", allow_nan=allow_nan, quiet=quiet)
            self.F: np.ndarray = Fi

            #* Direction assertion
            err3 = np.linalg.norm(SphGrid.vec - self.z, axis=-1)
            assert_error_bound(err3, name="Direction error of z-axis of the frame", allow_nan=allow_nan, quiet=quiet)
        else:
            assert_frame(Fi, name="Fi", allow_nan=allow_nan, quiet=quiet)
            assert_frame(Fo, name="Fo", allow_nan=allow_nan, quiet=quiet)
            self.Fi: np.ndarray = Fi
            self.Fo: np.ndarray = Fo

            #* Direction assertion
            _, _, err3 = align_azimuth((SphGrid.veci_s2m, SphGrid.veco), (self.zi, self.zo))
            assert_error_bound(err3, name="Out-of-azimuthal error of z-axes of frames", allow_nan=allow_nan, quiet=quiet)
    
    @classmethod
    def from_equirect(cls, h: int, w: int):
        sphG = SphereGridEquirect(h, w)
        return sphG.ThetaPhiFrameField()
    
    @classmethod
    def from_mirrorball(cls, size: int):
        sphG = SphereGridMirrorball(size)
        return sphG.GeodesicFrameField()

    @classmethod
    def from_persp(cls,
                   h: int,
                   w: int,
                   fov_x_deg: float,
                   to_world: ArrayLike = np.eye(3) # Transform
                  ): # s2m SphFF
        sphG = SphereGridPersp(h, w, fov_x_deg, to_world)
        return sphG.PerspectiveFrameField(to_world[:, 1])
    
    @classmethod
    def from_cube(cls, edge: int):
        sphG = SphereGridCube(edge)
        return sphG.CubemapFrameField()

    @classmethod
    def from_frames(cls,
                    Fi:        ArrayLike,        # [*,3,3]
                    Fo:        Optional[ArrayLike] = None, # [*,3,3]
                    allow_nan: bool = False,
                    quiet:     bool = True
                   ) ->        SphereFrameField:
        '''
        Generate `SphereFrameField` instance from frames.
        params:
            Fi:        ArrayLike[*,3,3]
            Fo:        ArrayLike[*,3,3], optional
            allow_nan: bool, optional
            quiet:     bool, optional
        return:
            smf:       SphereFrameField
        '''
        theta_i, phi_i = vec2sph(Fi[...,2])
        param_list = [theta_i, phi_i]
        if Fo is None:
            dom_type = DomType.UNI
        else:
            dom_type = DomType.BI
            theta_o, phi_o = vec2sph(Fo[...,2])
            param_list += [theta_o, phi_o]
        sgrid = SphereGrid(dom_type, AngType.RAD, *param_list)
        return cls(sgrid, Fi, Fo, allow_nan=allow_nan, quiet=quiet)

    def __repr__(self):
        result = "SphereFrameField[\n"
        result += f"  dom_type = {repr(self.SphGrid.dom_type)},\n"
        result += f"  shape = [{str(self.shape)[1:-1]}],\n"
        result += "]"
        return result
    
    def apply_rotation(self, rotation: RotationLike):
        SphGrid_f = self.SphGrid.apply_rotation(rotation)
        if self.SphGrid.dom_type == DomType.UNI:
            rotation = rotation2quat(rotation).to_rotation_matrix
            F_f = rotation @ self.F
            SphFF_f = SphereFrameField(SphGrid_f, F_f)
        else:
            raise NotImplementedError()
        return SphFF_f

    @property
    def dom_type(self):
        return self.SphGrid.dom_type

    #* *x3 geometric vector of each local frame axes
    @property
    def x(self):
        assert self.dom_type == DomType.UNI
        return self.F[...,:,0]
    @property
    def y(self):
        assert self.dom_type == DomType.UNI
        return self.F[...,:,1]
    @property
    def z(self):
        assert self.dom_type == DomType.UNI
        return self.F[...,:,2]
    @property
    def xi(self):
        assert self.dom_type in [DomType.ISOBI, DomType.BI]
        return self.Fi[...,:,0]
    @property
    def yi(self):
        assert self.dom_type in [DomType.ISOBI, DomType.BI]
        return self.Fi[...,:,1]
    @property
    def zi(self):
        assert self.dom_type in [DomType.ISOBI, DomType.BI]
        return self.Fi[...,:,2]
    @property
    def xo(self):
        assert self.dom_type in [DomType.ISOBI, DomType.BI]
        return self.Fo[...,:,0]
    @property
    def yo(self):
        assert self.dom_type in [DomType.ISOBI, DomType.BI]
        return self.Fo[...,:,1]
    @property
    def zo(self):
        assert self.dom_type in [DomType.ISOBI, DomType.BI]
        return self.Fo[...,:,2]
        
    def visualize(self,
                  marker_size:  Optional[float] = 3.0,
                  marker_dsamp: Optional[int]   = 1,
                  arrow_size:   Optional[float] = None, # 0.05
                  arrow_dsamp:  Optional[int]   = None,
                  title:        Optional[str] = None,
                  help:         Optional[bool]  = None):
        # ---------- Parameters ----------
        if arrow_size is None:
            arrow_size = plot._vis_options_vispy['arrow_size']
        if arrow_dsamp is None:
            arrow_dsamp = plot._vis_options_vispy['arrow_dsamp']
        if help is None:
            help = plot._vis_options_vispy['help']

        # ---------- Main ----------
        canvas, view = plot.gen_vispy_canvas()
        sphere = vispy.scene.visuals.Sphere(radius=0.999999, #0.999999,
                                            color=(0.99, 0.99, 0.99, 1),
                                            parent=view.scene)
        axes, event_a = plot.gen_vispy_axes(view.scene)

        idx_marker = self.SphGrid.dsamp_index_exp(marker_dsamp)
        grid = vispy.scene.visuals.Markers(parent=view.scene)
        grid.set_data(self.SphGrid.vec[idx_marker].reshape(-1, 3),
                      edge_width=0,
                      face_color=(0., 0., 0., 0.8),
                      size=marker_size)
        
        idx_arrow = self.SphGrid.dsamp_index_exp(arrow_dsamp)
        inst_pos = self.z[idx_arrow].reshape(-1, 3)
        inst_tf  = arrow_size * self.F[idx_arrow].reshape(-1, 3, 3)
        
        def gen_instmesh(color, inst_tf):
            return vispy.scene.visuals.InstancedMesh(
                plot.visArrow.vertices,
                plot.visArrow.faces,
                color=color,
                instance_positions=inst_pos,
                instance_transforms=inst_tf,
                parent=view.scene
            )
        
        arrowX = gen_instmesh('red', inst_tf @ roty(np.pi/2))
        arrowY = gen_instmesh('green', inst_tf @ rotx(-np.pi/2))
        arrowZ = gen_instmesh('blue', inst_tf)
        XYZ_visible = [2]

        @canvas.events.key_press.connect
        def on_key_press(event):
            match event.key:
                case "a":
                    event_a()
                case "g":
                    grid.visible = not grid.visible
                case "f":
                    XYZ_visible[0] = (XYZ_visible[0] + 1) % 3
                    arrowY.visible = (XYZ_visible[0] >= 1)
                    arrowZ.visible = (XYZ_visible[0] >= 2)

        if help:
            print("[Keyboard interface]\n"
                "A: hide/show global axes\n"
                "G: hide/show sphere grids\n"
                "F: show x(, y(, z)) axes for the frame field")
        
        if title is not None:
            # plot.vispy_attach_title(canvas, title)
            plot.vispy_attach_title(view, title)
        return canvas
    
        

class ScalarField:
    def __init__(self,
                 sphG: SphereGrid, # sphG.shape: *g
                 fval: ArrayLike,  # [*g,*c] values of the scalar field 
                ):
        self.SphGrid = sphG
        assert self.SphGrid.dom_type == DomType.UNI
        self.shape_grid = self.SphGrid.shape
        self.axes_grid = tuple(range(len(self.shape_grid)))
        assert fval.shape[:len(self.shape_grid)] == self.shape_grid, f"The front part of fval.shape ({fval.shape}) should be equal to self.shape_grid ({self.shape_grid})"
        self.shape_chan = fval.shape[len(self.shape_grid):]
        self.axes_chan = tuple(range(len(self.shape_grid), fval.ndim))
        self.ndim_grid = len(self.shape_grid)
        self.ndim_chan = len(self.shape_chan)

        self.fval = fval
    
    @classmethod
    def from_image(cls,
                   filename:  Union[str, Path], # exr file
                   fov_x_deg: float
                  )        -> ScalarField:
        filename = str(filename)
        img = imread(filename)
        h,w,c = img.shape
        sphG = SphereGridPersp(h, w, fov_x_deg)
        return cls(sphG, img)

    @classmethod
    def from_cubeimage(cls,
                       filename: Union[str, Path] # exr file
                      )       -> ScalarField:
        filename = str(filename)
        img = imread(filename)
        h,w,c = img.shape
        edge = cubemap_hw2edge(h, w)
        sphG = SphereGridCube(edge)
        return cls(sphG, envmap_stack(img))
    
    @classmethod
    def from_equirectimage(cls,
                           filename: Union[str, Path] # exr file
                          )       -> ScalarField:
        filename = str(filename)
        img = imread(filename)
        h,w,c = img.shape
        sphG = SphereGridEquirect(h, w)
        return cls(sphG, img)

    @classmethod
    def from_SH_upto(cls,
                     level:   int,
                     sphG:    Union[SphereGrid, SphereFrameField],
                     sh_type: SHType,
                     rotated: Optional[ArrayLike] = None
                    ) ->      ScalarField:
        """
        Construct ScalarField for spin-2 SH basis functions ($\overset{\leftrightarrow}{Y}_{lm1}$)
        where each (l,m)-index is considered as a channel of the ScalarField instance.
        """
        if isinstance(sphG, SphereFrameField):
            sphG = sphG.SphGrid
        assert isinstance(sphG, SphereGrid), f"`sphG` should be an instance of `SphereGrid` or `SphereFrameField`, but {type(sphG)=} is given."
        assert sphG.dom_type == DomType.UNI, f"Invalid domain type of `SphereGrid`: {sphG.dom_type=}"

        fval = sphG.SH_upto(level, False, sh_type, rotated=rotated)
        return cls(sphG, fval)
    
    @classmethod
    def from_SHCoeff(cls,
                     shv:   SHVec,
                     sphG:  Union[SphereGrid, SphereFrameField],
                     quiet: Optional[bool] = True
                    ) ->    ScalarField:
        if not quiet:
            t = time()
            print("\n## [Function start] ScalarField.from_SHcoeff")
        assert shv.cod_type == CodType.SCALAR
        if isinstance(sphG, SphereFrameField):
            sphG = sphG.SphGrid
        assert isinstance(sphG, SphereGrid)
        assert sphG.dom_type == DomType.UNI
        ## Assert SHcoeff[...,1:3] is virtually real

        ## Deal with sh_type
        if shv.sh_type == SHType.COMP: dtype = np.complex128
        else:                          dtype = np.float64
        
        n_grid = math.prod(sphG.shape) # NOTE np.int is weaker to overflow
        shape_chan = shv.coeff.shape[:-1]
        fval = np.zeros(sphG.shape + shape_chan, dtype = dtype)
        fval_short = fval.view().reshape(n_grid, *shape_chan)
        
        full_memory = n_grid * shv.N * 16
        n_pass = math.ceil(full_memory / MEMORY_PER_PASS)
        if not quiet and n_pass > 1:
            iter = tqdm(range(n_pass), desc="Seperate passes due to memory")
        else:
            iter = range(n_pass)
        
        for i_pass in iter:
            ## SHgrid[Ng', N(L)]
            ## SHcoeff[*c, N(L)]
            i = n_grid*i_pass // n_pass
            j = n_grid*(i_pass+1) // n_pass
            kargs = {"sh_type":shv.sh_type, "cfg_pass":(i_pass, n_pass)}
            
            ## Scalar components
            if shv.cod_type == CodType.POLAR3:
                sidx = 0
                s_tdot_ax = (-1, -1)
            else:
                sidx = [0,3]
                s_tdot_ax = (-1, -2)
            SHgrid = sphG.SH_upto(shv.level, False, **kargs)
            
            # temp[Ng', *c]
            temp = np.tensordot(SHgrid, shv.coeff, axes = (-1, -1))
            del SHgrid
            fval_short[i:j, ...] = temp
            del temp
        
        scalF = cls(sphG, fval)
        if not quiet:
            print(f"## [Function end] time: {time()-t} seconds.")
        return scalF
    
    @classmethod
    def zeros_like(cls,
                   scalF: ScalarField
                  ) ->    ScalarField:
        return cls(scalF.SphGrid, np.zeros_like(scalF.fval))
    
    def __repr__(self):
        def str2(tup: tuple) -> str:
            return str(tup)[1:-1]
        
        result = "ScalarField[\n"
        result += f"  dom_type = {repr(self.SphGrid.dom_type)},\n"
        result += f"  fval.shape = [g:{str2(self.shape_grid)} | c:{str2(self.shape_chan)}],\n"
        result += "]"
        return result
    
    def __add__(self, scalF: ScalarField) -> ScalarField:
        """
        TODO: Broadcasting for `shape_chan`'s??
        """
        assert isinstance(scalF, ScalarField), f"Invalid argument type: {type(scalF) = }"
        assert self.SphGrid is scalF.SphGrid, f"`SphGrid` attributes of `self` and `scalF` must be identical," + \
                                              f" but {self.SphGrid=} and {scalF.SphGrid=} are given."
        return ScalarField(self.SphGrid, self.fval + scalF.fval)
    def __sub__(self, scalF: ScalarField) -> ScalarField:
        """
        TODO: Broadcasting for `shape_chan`'s??
        """
        assert isinstance(scalF, ScalarField), f"Invalid argument type: {type(scalF) = }"
        assert self.SphGrid is scalF.SphGrid, f"`SphGrid` attributes of `self` and `scalF` must be identical," + \
                                              f" but {self.SphGrid=} and {scalF.SphGrid=} are given."
        return ScalarField(self.SphGrid, self.fval - scalF.fval)
    def __mul__(self, x: Union[float, ScalarField]) -> ScalarField:
        if isinstance(x, ScalarField):
            assert self.SphGrid is x.SphGrid, f"`SphGrid` attributes of `self` and `x` must be identical," + \
                                              f" but {self.SphGrid=} and {x.SphGrid=} are given."
            return ScalarField(self.SphGrid, self.fval * x.fval)
        else:
            assert np.isscalar(x)
            assert np.isrealobj(x)
            return ScalarField(self.SphGrid, self.fval * x)
    def __div__(self, x: Union[float, ScalarField]) -> ScalarField:
        if isinstance(x, ScalarField):
            assert self.SphGrid is x.SphGrid, f"`SphGrid` attributes of `self` and `x` must be identical," + \
                                              f" but {self.SphGrid=} and {x.SphGrid=} are given."
            return ScalarField(self.SphGrid, self.fval / x.fval)
        else:
            assert np.isscalar(x)
            assert np.isrealobj(x)
            return ScalarField(self.SphGrid, self.fval / x)
    def __iadd__(self, scalF: ScalarField) -> ScalarField:
        assert isinstance(scalF, ScalarField), f"Invalid argument type: {type(scalF) = }"
        assert self.SphGrid is scalF.SphGrid, f"`SphGrid` attributes of `self` and `scalF` must be identical," + \
                                              f" but {self.SphGrid=} and {scalF.SphGrid=} are given."
        self.fval += scalF.fval
        return self
    def __isub__(self, scalF: ScalarField) -> ScalarField:
        assert isinstance(scalF, ScalarField), f"Invalid argument type: {type(scalF) = }"
        assert self.SphGrid is scalF.SphGrid, f"`SphGrid` attributes of `self` and `scalF` must be identical," + \
                                              f" but {self.SphGrid=} and {scalF.SphGrid=} are given."
        self.fval -= scalF.fval
        return self
    def conj(self) -> ScalarField:
        return ScalarField(self.SphGrid, self.fval.conj())
    
    def allclose(self, scalF: ScalarField, rtol: Optional[float]=1e-05, atol: Optional[float]=1e-08) -> bool:
        if not isinstance(scalF, ScalarField):
            raise TypeError(f"Invalid type: {type(ScalarField)=}. The type must be `ScalarField`.")
        if not self.SphGrid is scalF.SphGrid:
            raise ValueError(f"`.SphGrid: SphereGrid` attributes of two operands must be the identical instance.")
        return np.allclose(self.fval, scalF.fval, rtol=rtol, atol=atol)
    
    @property
    def chan(self) -> _ScalarField_channel_indexer:
        return _ScalarField_channel_indexer(self)

    def save_envmap(self, filename: Union[str,Path]): # filename must contain one '%d'
        ## TODO Make more safe. The current version naively assumes self is for an env. map.
        filename = str(filename)
        fval_env = envmap_unfold(self.fval)
        imwrite(filename, fval_env)
    

    def fval_weighted(self):
        return self.fval * np.expand_dims(self.SphGrid.weight(), self.axes_chan)


    def multiply_grid(self, mat: ArrayLike) -> ScalarField:
        mat = np.asarray(mat)
        return ScalarField(self.SphGrid, self.fval * np.expand_dims(mat, self.axes_chan ))
    def multiply_chan(self, mat: ArrayLike) -> ScalarField:
        mat = np.asarray(mat)
        return ScalarField(self.SphGrid, self.fval * mat)

    def apply_rotation(self, rotation: RotationLike) -> ScalarField:
        SphGrid_f = self.SphGrid.apply_rotation(rotation)
        return ScalarField(SphGrid_f, self.fval)
    
    
    def inner(self,                 # *g, *c1 := self.shape_grid, self.shape_chan
              scalF:   ScalarField, # *g, *c2 := scalF.shape_grid, scalF.shape_chan
              chan_tdot: Optional[bool] = False, # False | True
             )        -> ArrayLike: # [*c1|*c2, p] | [*c1, *c2, p]
        '''
        Compute the inner product <self, scalF> as scalar fields on S^2.
        
        params:
            self.fval[*g, *c1]:  ArrayLike
            scalF.fval[*g, *c2]: ArrayLike
            chan_tdot:           bool
        return:
            INNER[*c]: ArrayLike,
        
        where *c == `np.broadcast_shape(*c1, *c2)` if `chan_tdot` == False
                 == `*c1 + *c2`                    if      "      == True
        '''
        assert self.SphGrid is scalF.SphGrid

        if chan_tdot:
            # INNER[*c1, *c2]
            return np.tensordot(self.fval_weighted().conj(), scalF.fval, axes=(self.axes_grid, self.axes_grid))
        else:
            # INNER[*c1|*c2]    (broadcast `*c1` and `*c2`)
            prebc = prebc_chan(self.fval.shape, scalF.fval.shape,
                           len(self.shape_grid), chan_tdot)
            _, ax_exp_self, ax_exp_arg, _ = prebc

            fval_selfw = np.expand_dims(self.fval_weighted(), ax_exp_self)
            fval_arg = np.expand_dims(scalF.fval, ax_exp_arg)
            return np.sum(fval_selfw.conj() * fval_arg, axis=self.axes_grid)


    def integral(self) -> Union[np.float64, np.complex128]:
        return np.sum(self.fval_weighted(), self.axes_grid)


    def __color_for_plot(self,
                         chan_id: Optional[Union[int, Sequence[int]]] = None,
                         scale: Optional[float] = 1.0,
                         gamma: Optional[float] = None,
                         cmap:  Optional[Union[str, plot.Colormap, Callable]] = plot._vis_options['cmap']
                        ) ->      np.ndarray:
        color_cfg = dict(scale=scale, cmap=cmap)
        if gamma is not None:
            color_cfg['gamma'] = gamma
        gamma_for_rgb = 1/2.2 if gamma is None else gamma

        if chan_id is None:
            if self.shape_chan == (3,):
                return np.clip(self.fval[..., :], 0, 1)**gamma_for_rgb
            elif np.prod(self.shape_chan) == 1:
                return plot.apply_colormap_sgn(self.fval, axis=None, **color_cfg)
            else:
                raise ValueError(f"Only supports channel shape (), (1,), and (3,) for `ScalarField` instance, but {self.shape_chan} given.")
            
        
        elif isinstance(chan_id, int) or (hasattr(chan_id, 'shape') and chan_id.shape == ()):
            return plot.apply_colormap_sgn(self.fval[..., chan_id], axis=None, **color_cfg)
        
        elif hasattr(chan_id, "__len__"):
            if len(chan_id) == 2:
                return plot.apply_colormap2d(self.fval[..., chan_id], **color_cfg)
            elif len(chan_id) == 3:
                return np.clip(self.fval[..., chan_id], 0, 1)**gamma_for_rgb
            else:
                raise ValueError(f"`chan_id` must be an integer or a tuple of two or three integers, but ({type(chan_id)}) {chan_id=} given.")
        else:
            raise ValueError(f"`chan_id` must be an integer or a tuple of two or three integers, but ({type(chan_id)}) {chan_id=} given.")
    
    def gen_vispy_objs(self,
                       parent,
                       chan_id:      Optional[Union[int, Sequence[int]]] = None,
                       marker_size:  Optional[float] = 5.0,
                       marker_dsamp: Optional[int] = 1,
                       scale:        Optional[float] = 1.0,
                       gamma:        Optional[float] = None,
                       cmap:         Optional[Union[str, plot.Colormap, Callable]] = plot._vis_options['cmap']):
        # -------------------- Base sphere --------------------
        sphere = vispy.scene.visuals.Sphere(radius=0.99,
                                            color=(0.99, 0.99, 0.99, 1.0),
                                            parent=parent)
        
        # -------------------- Colored markers --------------------
        color = self.__color_for_plot(chan_id, scale, gamma, cmap)

        idx_exp = self.SphGrid.dsamp_index_exp(marker_dsamp)
        grid = vispy.scene.visuals.Markers(parent=parent)

        if color.shape[-1] == 3:
            alpha = np.broadcast_to(0.8, self.shape_grid)[idx_exp].reshape(-1, 1)
            color_texture = np.concatenate([color[idx_exp].reshape(-1, 3), alpha],
                                           -1)
        elif color.shape[-1] == 4:
            color_texture = color[idx_exp].reshape(-1, 4)
        else:
            raise ValueError(f"Unvalid shape: {color.shape = }. The last axis must be 3 or 4.")
        np.clip(color_texture, 0, 1, out=color_texture)
        
        grid.set_data(self.SphGrid.vec[idx_exp].reshape(-1, 3),
                      edge_width=0,
                      face_color=color_texture,
                      size=marker_size)
        
        # -------------------- (Textured) meshes --------------------
        flag_texture = bool(self.SphGrid.VisObj_list)
        sphere.visible = not flag_texture
        grid.visible = not flag_texture

        if flag_texture:
            meshes = []
            visobj_list = self.SphGrid.VisObj_list
            for i, visobj in enumerate(visobj_list):
                mesh = vispy.scene.visuals.Mesh(visobj.vertices,
                                                visobj.faces,
                                                color='white' if i==0 else (0.99, 0.99, 0.99, 1),
                                                parent=parent)
                meshes.append(mesh)
            
            edge_perhaps = self.shape_grid[1]
            if self.shape_grid == (6, edge_perhaps, edge_perhaps): # isinstance(self.SphGrid, SphereGridCube), generalized for rotated
                img = envmap_unfold(color)
            else:
                img = color
            img = img.astype(np.float32)
            texture = vispy.visuals.filters.TextureFilter(img, visobj_list[0].texcoords)
            meshes[0].attach(texture)

        # -------------------- Keyboard event --------------------
        if flag_texture:
            def func_toggle_mode():
                sphere.visible = not sphere.visible
                grid.visible = not grid.visible
                for mesh in meshes:
                    mesh.visible = not mesh.visible
        else:
            func_toggle_mode = None
        
        # -------------------- End --------------------
        if not flag_texture:
            return sphere, grid, func_toggle_mode
        else:
            return sphere, grid, meshes, func_toggle_mode

    def visualize(self,
                  chan_id:      Optional[Union[int, Sequence[int]]] = None,
                  marker_size:  Optional[float] = 5.0,
                  marker_dsamp: Optional[int] = 1,
                  scale:        Optional[float] = 1.0,
                  gamma:        Optional[float] = None,
                  cmap:         Optional[Union[str, plot.Colormap, Callable]] = plot._vis_options['cmap'],
                  title:        Optional[str] = None,
                  help:         Optional[bool] = None):
        canvas, view = plot.gen_vispy_canvas()
        axes, event_a = plot.gen_vispy_axes(view.scene)

        sphere, grid, *wrap_meshes, event_v = self.gen_vispy_objs(view.scene,
                                                            marker_size=marker_size,
                                                            marker_dsamp=marker_dsamp,
                                                            chan_id=chan_id,
                                                            scale=scale,
                                                            gamma=gamma,
                                                            cmap=cmap)
        
        @canvas.events.key_press.connect
        def on_key_press(event):
            if event.key == "a":
                event_a()
            elif event.key == "v":
                if event_v is not None:
                    event_v()

        if title is not None:
            plot.vispy_attach_title(view, title)

        if help is None:
            help = plot._vis_options_vispy['help']
        if help:
            print("[Keyboard interface]\n" + \
                ("" if (event_v is None) else "V: texture/point cloud mode\n") + \
                "A: hide/show global axes")
            
        return canvas

    
    '''
    ########################################################################
    # >>                            Resampling                          << #
    ########################################################################
    '''
    def resample_equirect(self,             # [*g, *c]
                          h:    int,
                          w:    int
                         ) ->   np.ndarray: # [h, w, *c]
        if self.SphGrid.dom_type != DomType.UNI:
            raise ValueError(f"Invalid domain type: {self.dom_type=}")

        fval = self.fval
        ndim_grid = len(self.shape_grid)
        assert fval.shape[:ndim_grid] == self.shape_grid
        shape_chan = fval.shape[ndim_grid:]
        fval2D = fval.reshape(-1, int(np.prod(shape_chan)))

        vec_rect2D = np.stack([self.SphGrid.theta_grid.ravel(), self.SphGrid.phi_grid.ravel()], axis=-1)
        if self.SphGrid.ang_type == AngType.DEG:
            vec_rect2D = np.deg2rad(vec_rect2D)
        
        # Avoid interpolating around poles, but allow for phi=0 and 2*pi
        azim_thres = np.pi/2
        
        I1, = np.where((vec_rect2D[:,1] > -azim_thres) & (vec_rect2D[:,1] < 0))
        I2, = np.where((vec_rect2D[:,1] > 0) & (vec_rect2D[:,1] < azim_thres))
        vec_tail1 = vec_rect2D[I1,:]
        vec_tail2 = vec_rect2D[I2,:]
        vec_tail2[:,1] += 2*np.pi
        vec_rect2D[:,1][vec_rect2D[:,1] < 0] += 2*np.pi
        vec_rect2D = np.concatenate([vec_rect2D, vec_tail1, vec_tail2])
        # NOTE `np.prod( () ) == 1.0`` for the empty tuple, so we need type casinting `int`
        
        fval2D = np.concatenate([fval2D, fval2D[I1,:], fval2D[I2,:]])
        
        # Consistent external reference:
        # [https://github.com/haruishi43/equilib/blob/8d492b7181a82a6690a04e485ca10d794147fa61/equilib/cube2equi/numpy.py#L160]
        theta_lin, phi_lin = linspace_tp(h, w)
        x_img, y_img = np.meshgrid(theta_lin, phi_lin, indexing='ij')
        interp_lin = LinearNDInterpolator(vec_rect2D, fval2D)
        fval_interp = interp_lin(x_img, y_img)
        # interp_NN = NearestNDInterpolator

        return fval_interp.reshape((h, w) + shape_chan)
    

    '''
    ########################################################################
    # >>                          SH Coefficient                        << #
    ########################################################################
    '''
    def SHCoeff(self, # .fval[*g, *c], where *g := shape_grid, *c := shape_chan
                level:   int,
                sh_type: SHType = SHType.COMP,
                quiet:   bool = True
               )      -> SHVec: # [*c, N(L)]
        ## NOTE via iteration rather than vectorization
        ##      for O(|grid| + N(L)) memory
        if not quiet:
            t = time()
            print("\n## [Function start] ScalarField.SHCoeff")
        sh_type = SHType(sh_type)

        ## SphFF.F: shape_grid x 3 x 3
        ## Stk:     shape_grid x shape_chan x 4
        
        n_grid = math.prod(self.shape_grid) # NOTE np.int is more weak to overflow
        N_lm = level2num(level, DomType.UNI)
        shape_re = (n_grid,) + self.shape_chan
        fval_self = self.fval_weighted().view().reshape(shape_re) # [Ng,*c]
        
        full_memory = n_grid * N_lm * 16
        n_pass = math.ceil(full_memory / MEMORY_PER_PASS)
        
        if sh_type == SHType.COMP:
            res_dtype = np.complex128
        else:
            res_dtype = np.float64
        res = np.zeros(self.shape_chan + (N_lm,), dtype=res_dtype)
        
        iter = range(n_pass)
        if not quiet and n_pass > 1:
            iter = tqdm(iter, desc="Seperate passes due to memory")
            
        for i_pass in iter:
            i = n_grid*i_pass // n_pass
            j = n_grid*(i_pass+1) // n_pass
            kargs = {"sh_type":sh_type, "cfg_pass":(i_pass, n_pass)}
            
            ## Scalar components
            fval_slice = fval_self[i:j, ...] # [Ng', *c]
            SHgrid = self.SphGrid.SH_upto(level, False, **kargs) # [Ng', N(L)]
            res[:] += np.tensordot(fval_slice, SHgrid.conj(),
                                axes=(0,0)) # [*c, N(L)]
            del fval_slice, SHgrid

        res_shv = SHVec(res, CodType.SCALAR, sh_type)
        if not quiet:
            print(f"## [Function end] time: {time()-t} seconds.")
        return res_shv

class _ScalarField_channel_indexer:
    def __init__(self, obj: ScalarField):
        self.obj = obj
    def __getitem__(self, key):
        key_grid = tuple(slice(None, None, None) for _ in self.obj.shape_grid)
        if not isinstance(key, tuple):
            key = (key, )
        key_res = key_grid + key
        return ScalarField(self.obj.SphGrid, self.obj.fval[key_res])
    
    def __add__(self, arr: ArrayLike) -> ScalarField:
        return ScalarField(self.obj.SphGrid, self.broadcast_self_with(arr) + self.broadcast_array(arr))
    def __sub__(self, arr: ArrayLike) -> ScalarField:
        return ScalarField(self.obj.SphGrid, self.broadcast_self_with(arr) - self.broadcast_array(arr))
    def __mul__(self, arr: ArrayLike) -> ScalarField:
        return ScalarField(self.obj.SphGrid, self.broadcast_self_with(arr) * self.broadcast_array(arr))
    def __div__(self, arr: ArrayLike) -> ScalarField:
        return ScalarField(self.obj.SphGrid, self.broadcast_self_with(arr) / self.broadcast_array(arr))
    
    def __iadd__(self, arr: ArrayLike) -> ScalarField:
        self.obj.fval += self.broadcast_array(arr)
        return self # or `self.obj`? I am not sure which one is prefarable.
    def __isub__(self, arr: ArrayLike) -> ScalarField:
        self.obj.fval -= self.broadcast_array(arr)
        return self # or `self.obj`? I am not sure which one is prefarable.
    def __imul__(self, arr: ArrayLike) -> ScalarField:
        self.obj.fval *= self.broadcast_array(arr)
        return self # or `self.obj`? I am not sure which one is prefarable.
    def __idiv__(self, arr: ArrayLike) -> ScalarField:
        self.obj.fval /= self.broadcast_array(arr)
        return self # or `self.obj`? I am not sure which one is prefarable.

    def broadcast_array(self, arr: ArrayLike) -> np.ndarray:
        arr = np.asarray(arr)
        return np.broadcast_to(arr, np.broadcast_shapes(arr.shape, self.obj.shape_chan))
    def broadcast_self_with(self, arr: ArrayLike) -> np.ndarray:
        arr = np.asarray(arr)
        shape_chan_res = np.broadcast_shapes(arr.shape, self.obj.shape_chan)
        ndim_grid = len(self.obj.shape_grid)
        ndim_chan_diff = len(shape_chan_res) - len(self.obj.shape_chan)
        ax_expdims = tuple(range(ndim_grid, ndim_grid + ndim_chan_diff))
        return np.broadcast_to(np.expand_dims(self.obj.fval, ax_expdims),
                               self.obj.shape_grid + shape_chan_res)
    def rgb2gray(self):
        if self.obj.shape_chan != (3,):
            raise ValueError(f"Invalid channel shape for the `ScalarField` instance: {self.obj.shape_chan}")
        return ScalarField(self.obj.SphGrid, plot.rgb2gray(self.obj.fval, -1))
    

def assert_frame(F:         ArrayLike, # *x3x3
                 name:      str = "F",
                 allow_nan: bool = False,
                 quiet:     bool = True
                )        -> None: # For SphereFrameField.__init__
    ## Shape assertion
    assert F.shape[-2:] == (3,3), f"Shape of {name} should end with (3,3), but currently: {F.shape}"
    err = orthogonal_error(F)
    ## Orthogonality assertion
    assert_error_bound(err, name=f"Orthogonality error of {name}", allow_nan=allow_nan, quiet=quiet)

def SphereFrameField_cat(SMFs: List) -> SphereFrameField:
    dom_type = consistent_value([SMF.SphGrid.dom_type for SMF in SMFs])
    if dom_type == DomType.UNI:
        extractors = [lambda SMF:SMF.SphGrid, lambda SMF:SMF.F]
    else:
        extractors = [lambda SMF:SMF.SphGrid, lambda SMF:SMF.Fi, lambda SMF:SMF.Fo]

    ## params_res == [SphGrid, F] or [SphGrid, Fi, Fo]
    params_res = [cat_adaptive([ext(SMF) for SMF in SMFs])
                     for ext in extractors]
    return SphereFrameField(*params_res)

def SphereFrameField_args_from_persp(h: int,
                                     w: int,
                                     fov_x_deg: float,
                                     to_world: Optional[ArrayLike] = np.eye(3), # Transform
                                     pixel_center: Optional[bool] = True
                                    ): # s2m SphFF
    '''
    Mitsuba perspecctive moving frame
      ^ y
      | (xy image plane)
      o -- > x
     /
    z
    source: https://github.com/mitsuba-renderer/mitsuba3/blob/master/src/integrators/stokes.cpp#L104
    '''
    tanx_half = np.tan(fov_x_deg/2*np.pi/180)
    # tany_half = tanx_half * h/w
    if pixel_center:
        pix_size = tanx_half*2/w
    else:
        pix_size = tanx_half*2/(w-1)

    pplane_x = (np.arange(w).reshape(1,-1) - (w-1)/2)*pix_size
    pplane_y = (np.arange(h-1,-1,-1).reshape(-1,1) - (h-1)/2)*pix_size
    pplane_z = np.full((h,w), -1) # projective plane
    pplane_x, pplane_y, pplane_z = np.broadcast_arrays(pplane_x, pplane_y, pplane_z)
    pplane = np.stack([pplane_x, pplane_y, pplane_z], axis=-1)
    pplane = normalize(pplane)
    pplane = matmul_vec1d(to_world, pplane)
    
    local_y = to_world @ unit_y # `== R_tf[...,:,1]`
    ## NOTE source-to-material incident radiance
    Fx = normalize(np.cross(local_y, -pplane))
    Fy = normalize(np.cross(-pplane, Fx))
    F = np.stack([Fx, Fy, -pplane], -1)

    theta, phi = vec2sph(-pplane)
    return theta, phi, F


##################################################
### py:class:ScalarConvKernel
##################################################
class ScalarConvKernel:
    def __init__(self, func: Callable[[float], ArrayLike]):
        """
        func(x[*]) -> [*, *c], radian only
        """
        fval = np.asarray(func(0))
        self.shape_chan = fval.shape

        self.func = func

    def __repr__(self):
        return "ScalarConvKernel[\n" + \
              f"  shape_chan = {self.shape_chan},\n" + \
               "]"
    
    def __call__(self, theta: ArrayLike) -> np.ndarray:
        return self.func(theta)
    
    def at(self, theta):
        return self(theta)

    def apply_delta(self,                  # [*c]
                    vec:      ArrayLike,   # [3]
                    sphG_out: SphereGrid   # [*g]
                   ) ->       ScalarField: # [*g, *c], ScalarField(sphG_out, fval_res)
        """
        `res.shape_chan = self.shape_chan`
        """
        vec = np.asarray(vec)
        assert vec.shape == (3,)
        vec = normalize(vec)
        assert isinstance(sphG_out, SphereGrid), f"Invalid type: {type(sphG_out) = }"

        # ---------- Main 1/2 ----------
        theta = np.arccos(np.clip(sphG_out.vec @ vec, -1, 1))
        return ScalarField(sphG_out, self(theta))

    def apply(self,               # [*c1]
              scalF: ScalarField, # [*g, *c2]
              quiet: Optional[bool] = True
             ) ->    ScalarField: # [*g, (*c1|*c2)]
        sphG = scalF.SphGrid
        assert sphG.dom_type == DomType.UNI

        fval_weighted = scalF.fval_weighted()

        first = True
        iterator = np.ndenumerate(sphG.theta_grid)
        if not quiet:
            iterator = tqdm(iterator)
        for idx_tup, _ in iterator:
            curr = self.apply_delta(sphG.vec[idx_tup], sphG)
            curr = curr.chan * fval_weighted[idx_tup]
            if first:
                sphF_res = curr
            else:
                sphF_res += curr
            first = False
        return sphF_res

    def SHCoeff(self,
                level:     int,
                n_sample:  Optional[int] = None
               ) ->        SHConv:
        if n_sample is None:
            n_sample = 4*(level**2)
        theta = np.linspace(0, np.pi, n_sample) # [n_sample]
        dtheta = interval_length(theta, 0, np.pi)
        weight = 2*np.pi*np.sin(theta)*dtheta
        
        k_val = self(theta) # [n_sample, *c]

        # [n_sample, N(level)]
        SHval = SH_upto(theta, 0, level, False, sh_type=SHType.REAL)
        l, m = level2lms(level, DomType.UNI, unstack=True)
        
        # [n_sample, level]
        s0m0_SHval = SHval[:, m == 0]

        """
        coeff_s2s:  ArrayLike, # [*c, level]
        """
        coeff_s2s = np.einsum('nl, n..., n -> ...l', s0m0_SHval.conj(), k_val, weight).astype(np.float64)
    
        return SHConv(CodType.SCALAR,
                      coeff_s2s = coeff_s2s,
                      coeff_s2v = None,
                      coeff_v2s = None,
                      coeff_v2va = None,
                      coeff_v2vb = None,
                      weighted = False)
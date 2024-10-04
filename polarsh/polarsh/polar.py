from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
from .array import assert_error_bound
from .sphere import align_azimuth, orthogonal_error
from .SH import CodType

##################################################
### Stokes-Mueller calculus
##################################################
def stokes_frame_convert(F_from:    ArrayLike, # *x3x3
                         F_to:      ArrayLike, # *x3x3
                         cod_type:  CodType = CodType.POLAR4,
                         allow_nan: Optional[bool] = False,
                         quiet:     Optional[bool] = True
                        )        -> ArrayLike: # *x4x4 (for default cod_type)
    '''
    Input: Frames, linear map from coord. vec. to geom. vec., i.e.
           Frame[:,0|1|2] indicates axis vectors of the frame.
    Return: frame conversion Mueller matrix M such that:
            M @ (Stokes coord. w.r.t. F_from) == (Stokes coord. w.r.t. F_to)
    Constraint: F_from[...,:,2] == F_to[...,:,2]
    '''
    if not quiet: print("\n# [Function start] stokes_frame_convert")
    assert F_from.shape[-2:] == (3,3), f"F_from.shape ({F_from.shape}) should end with (3,3)."
    assert F_to.shape[-2:] == (3,3), f"F_to.shape ({F_to.shape}) should end with (3,3)."
    assert cod_type in list(CodType)[1:], f"Invalid value of cod_type ({cod_type})."

    F_fromT = np.swapaxes(F_from,-1,-2)
    F_from2 = F_fromT@F_from
    F_to2 = F_fromT@F_to

    dphi, R, err = align_azimuth((F_to2[...,:,0],F_to2[...,:,1]), (F_from2[...,:,0],F_from2[...,:,1]))
    assert_error_bound(err, name="Out-of-azimuthal for two frames", allow_nan=allow_nan, quiet=quiet)

    shape_M = list(F_to2.shape)
    if cod_type == CodType.POLAR2:
        R2 = R[...,:2,:2]
        M = R2 @ R2
    elif cod_type == CodType.POLAR3:
        ## NOTE ".copy()" below is necessary since
        ## broadcasted numpy.ndarray has a flag WRITABLE == False
        R3 = np.broadcast_to(np.eye(3), shape_M).copy()
        R3[...,1:,1:] = R[...,:2,:2]
        M = R3 @ R3
    else: # cod_type == CodType.POLAR4
        shape_M[-2:] = [4,4]
        M = np.broadcast_to(np.eye(4), shape_M).copy()
        M[...,1:,1:] = R @ R

    if not quiet:
        err_M = orthogonal_error(M)
        assert_error_bound(err_M, name="Orthogonality for Mueller", allow_nan=allow_nan, quiet=True)
    return M

def mueller_frame_convert(M_from:    ArrayLike, # *gx*cx4x4
                          F_from:    ArrayLike, # *gx3x3 or tuple of 2
                          F_to:      ArrayLike, # *gx3x3 or tuple of 2
                          allow_nan: bool = False,
                          quiet:     bool = True
                         )        -> ArrayLike: # *gx*cx4x4
    ## *g: grid shape
    ## *c: channel shape                     
    if isinstance(F_from, np.ndarray):
        assert isinstance(F_to, np.ndarray)
        M_conv = stokes_frame_convert(F_from, F_to, allow_nan=allow_nan, quiet=quiet)
        ch_axes = tuple(range(-2 - M_from.ndim+F_from.ndim,-2))
        M_conv = np.expand_dims(M_conv, ch_axes)
        M_to = M_conv @ M_from @ np.swapaxes(M_conv,-1,-2)
    elif isinstance(F_from, (tuple, list)):
        assert isinstance(F_to, (tuple, list))
        Fi_from, Fo_from = F_from
        Fi_to, Fo_to = F_to
        Mi = stokes_frame_convert(Fi_to, Fi_from, allow_nan=allow_nan, quiet=quiet)
        Mo = stokes_frame_convert(Fo_from, Fo_to, allow_nan=allow_nan, quiet=quiet)
        ch_axes = tuple(range(-2 - M_from.ndim+Fi_from.ndim,-2))
        M_to = np.expand_dims(Mo, ch_axes) @ M_from @ np.expand_dims(Mi, ch_axes)
    else:
        raise TypeError(f"Invalid type of F_from: {type(F_from)}.")
    return M_to

##################################################
### Reflectance color convention
##################################################
def reflectance_spec2rgb(refl: ArrayLike, axis: Optional[int] = -1) -> np.ndarray:
    """
    Constraint:
        `refl.shape[axis] == 5`, which indicates spectral reflectance at 450, 500, 550, 600, and 650 nm.
    Reference:
        `polar-harmonics-code/python/test/color_transform.ipynb`
        `spec2rgb.m` and `XYZ2sRGBlinear` from [Seung-Hwan Baek et al. 2020]
    """
    # ---------- Constant (standards) ----------
    d65 = [117.008003000000, 109.353996000000, 104.045998000000, 90.0062030000000, 80.0268020000000]
    spec2xyz = np.array([[0.336200000000, 0.038000000000, 1.772110000000],
                         [0.004900000000, 0.323000000000, 0.272000000000],
                         [0.433449900000, 0.994950100000, 0.008749999000],
                         [1.062200000000, 0.631000000000, 0.000800000000],
                         [0.283500000000, 0.107000000000, 0.000000000000]]).T
    xyz2rgb = np.array([[3.2406, -1.5372, -0.4986],
                       [-0.9689,  1.8758,  0.0415],
                       [0.0557, -0.2040,  1.0570]])
    spec2rgb = xyz2rgb @ spec2xyz
    # ---------- main ----------
    if refl.shape[axis] != 5:
        raise ValueError(f"The argument `refl` must have five spectral channels alone given `{axis=}`, but currently: {refl.shape = }")
    refl = np.moveaxis(refl, axis, -1)
    res = np.einsum('ij,...j->...i', spec2rgb, refl * d65) / (spec2rgb @ d65)
    return np.moveaxis(res, -1, axis)

##################################################
### BSDF
##################################################
def fresnel_complex(cos_theta_i, eta):
    outside_mask = cos_theta_i >= 0.0
    eta_inv = 1/eta
    eta_it = np.where(outside_mask, eta, eta_inv)
    eta_ti = np.where(outside_mask, eta_inv, eta)
    
    cos_theta_t_sqr = 1 - eta_ti*eta_ti*(1-(cos_theta_i*cos_theta_i)) # Snell's law
    if np.isrealobj(cos_theta_t_sqr):
        cos_theta_t_sqr = cos_theta_t_sqr * (1+0j)
    
    cos_theta_i_abs = np.abs(cos_theta_i)
    cos_theta_t = np.sqrt(cos_theta_t_sqr*(1+0j))

    """
    Use n - ki convention for complex refractive index
    For the signs of imaginary parts, see a discussion here: https://github.com/mitsuba-renderer/mitsuba3/pull/1161
    """
    cos_theta_t = np.sqrt(cos_theta_t_sqr.conj()).conj()

    a_s = (cos_theta_i_abs - eta_it * cos_theta_t) / (cos_theta_i_abs + eta_it * cos_theta_t)
    a_p = (eta_it * cos_theta_i_abs - cos_theta_t) / (eta_it * cos_theta_i_abs + cos_theta_t)

    ## use material-to-source convention for cos_theta_i
    cos_theta_t = np.where((cos_theta_t_sqr >= 0) & (cos_theta_i >= 0), -cos_theta_t, cos_theta_t)
    return a_s, a_p, cos_theta_t, eta_it, eta_ti


def fresnel_reflection_wrt_spu(cos_theta_i, eta):
    a_s, a_p, _, _, _ = fresnel_complex(cos_theta_i, eta)
    a_s, a_p, Z = np.broadcast_arrays(a_s, a_p, 0)
    
    R_s = np.abs(a_s*a_s)
    R_p = np.abs(a_p*a_p)
    A = 0.5 * (R_s + R_p)
    B = 0.5 * (R_s - R_p)
    C = a_s.conj() * a_p

    res = np.stack([[A, B, Z, Z],
                    [B, A, Z, Z],
                    [Z, Z, C.real, -C.imag],
                    [Z, Z, C.imag, C.real]])
    return np.moveaxis(res, (0,1), (-2,-1))


def fresnel_refraction_wrt_spu(cos_theta_i, eta):
    a_s, a_p, cos_theta_t, eta_it, eta_ti = fresnel_complex(cos_theta_i, eta)
    assert np.all((cos_theta_t.imag == 0) | (cos_theta_t.real == 0) | np.isnan(cos_theta_t)), \
           "The method only supports nonconducting media."
    cos_theta_t = cos_theta_t.real

    # (amplitude)reflectance to transmittance
    a_s += 1.0
    a_p += 1.0
    a_p *= eta_ti

    a_s, a_p, Z = np.broadcast_arrays(a_s, a_p, 0)

    # $n\cdot S_i = n\cdot S_t$
    factor = -eta_it * np.where(np.abs(cos_theta_i) > 1e-8,
                                cos_theta_t / cos_theta_i, 0.0)
    
    T_s = np.abs(a_s*a_s)
    T_p  =np.abs(a_p*a_p)
    A = 0.5 * factor * (T_s + T_p)
    B = 0.5 * factor * (T_s - T_p)
    C = factor * np.abs(a_s*a_p)

    res = np.stack([[A, B, Z, Z],
                    [B, A, Z, Z],
                    [Z, Z, C, Z],
                    [Z, Z, Z, C]])
    return np.moveaxis(res, (0,1), (-2,-1))

def beckmann_distribution(cos_theta, alpha):
    alpha_2 = alpha * alpha
    cos_theta_2 = np.square(cos_theta)
    res = np.exp(-(1-cos_theta_2)/alpha_2/cos_theta_2)
    res /= np.pi * alpha_2 * np.square(cos_theta_2)
    return np.where(res * cos_theta > 1e-20, res, 0.0)

def beckmann_smith_g1(w, m, alpha): # [*, 3] -> [*]
    xy_alpha_2 = np.square(alpha * w[..., 0]) + np.square(alpha * w[..., 1])
    tan_theta_alpha_2 = xy_alpha_2 / np.square(w[..., 2])

    # Masked computations to avoid RuntimeWarning due to dividing by zero
    mask = xy_alpha_2 != 0.0
    a = np.reciprocal(np.sqrt(tan_theta_alpha_2), where=mask)
    a_sqr = np.square(a, where=mask)
    res = np.where(a >= 1.6, 1.0,
                   (3.535 * a + 2.181 * a_sqr) /
                    (1.0 + 2.276 * a + 2.577 * a_sqr))
    res[~mask] = 1.0
    # Masked computation has done.

    res[np.sum(w*m, -1) * w[..., 2] <= 0.0] = 0.0
    return res

def beckmann_G(wi_m2s, wo, m, alpha):
    return beckmann_smith_g1(wi_m2s, m, alpha) * beckmann_smith_g1(wo, m, alpha)
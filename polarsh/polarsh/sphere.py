'''
For testing, see "test_sphere.py"
'''
from enum import Enum
from typing import Tuple, List, Union, Optional
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as scipyRotation
import quaternionic as quat

from .array import *


def diff_cyclic(x1: ArrayLike, x2: ArrayLike, period: float) -> np.ndarray:
    '''
    Compute `x1 - x2` in the quotient ring ℝ/`period`ℤ.
    '''
    diff = np.remainder(np.asarray(x1) - np.asarray(x2), period)
    return np.where(diff < period / 2, diff, diff - period)

def cross_mat(vec:  ArrayLike, # [*, 3] (default)
              axis: int = -1
             )   -> ArrayLike: # [*, 3, 3] (default)
    if vec.shape[axis] != 3:
        raise ValueError(
            f"The size along given axis({axis}) should be equal to 3, but currently shape:{vec.shape}.")
    vec = np.swapaxes(vec, axis, -1) # vec[...,3]
    vec = np.expand_dims(vec, -2)  # vec[...,1,3]
    CPM = -np.cross(vec, np.eye(3))

    oaxis = axis
    if axis < 0:
        oaxis -= 1
    CPM = np.moveaxis(np.swapaxes(CPM, -2, oaxis), -1, oaxis+1)
    return CPM

def outer_mat(vec:  ArrayLike, # *xn (default)
              axis: int = -1
             )   -> ArrayLike: # *xnxn (default)
    vec = np.swapaxes(vec, axis, -1) # vec[...,3]
    col = np.expand_dims(vec, -1)    # vec[...,3,1]
    row = np.expand_dims(vec, -2)    # vec[...,1,3]
    mat = col*row

    oaxis = axis
    if axis < 0:
        oaxis -= 1
    mat = np.moveaxis(np.swapaxes(mat, -2, oaxis), -1, oaxis+1)
    return mat

def orthogonal_error(mat: ArrayLike  # [*, n, n]
                    )  -> ArrayLike: # [*]
    matT = np.swapaxes(mat, -1, -2)
    # I = np.eye(3)
    I = np.eye(mat.shape[-1])
    E1 = rms(matT @ mat - I, axis=(-1,-2))
    E2 = rms(mat @ matT - I, axis=(-1,-2))
    return np.maximum(E1, E2)

def align_azimuth(vpair1:   Tuple[ArrayLike, ArrayLike], # *x3 for each
                  vpair2:   Tuple[ArrayLike, ArrayLike], # *x3 for each
                  ang_type: AngType = AngType.RAD # for return dphi
                 )       -> Tuple[ArrayLike, ArrayLike, ArrayLike]: # See below
    ## return = (dphi, R, err)
    ##          dphi[*]: phi(vpair2) - phi(vpair1)
    ##          R[*x3x3]: rotation matrix along z axis by phi, which minimizes err
    ##          err[*]: |R @ vpair1 - vpair2|
    ## NOTE Do not use "items = [[]]*2" due to copying same reference of empty list []
    items = [] # items[h|ha][pair1|pair2]
    vpairs = [vpair1, vpair2]
    for i_item in range(2):
        items.append([])
        for j_pair in range(2):
            if i_item == 0:
                half_vector = (vpairs[j_pair][0] + vpairs[j_pair][1])/2
                items[i_item].append(half_vector)
            else: # if i_item == 1:
                half_vector_alternative = (vpairs[j_pair][0] - vpairs[j_pair][1])/2
                items[i_item].append(half_vector_alternative)

    items_rho2 = []
    items_dphi = []
    for i_item in range(2):
        items_rho2.append([])
        items_dphi.append([])
        pair = items[i_item]
        for j_pair in range(2):
            item = pair[j_pair]
            rho2 = item[...,0]**2 + item[...,1]**2
            dphi = np.arctan2(item[...,1], item[...,0])
            items_rho2[i_item].append(rho2)
            items_dphi[i_item].append(dphi)

    rho2_h = np.minimum( *(items_rho2[0]) )
    rho2_ha = np.minimum( *(items_rho2[1]) )
    dphi = np.where(rho2_h >= rho2_ha,
                    items_dphi[0][1] - items_dphi[0][0],
                    items_dphi[1][1] - items_dphi[1][0])
    R = axisang2rot(unit_z, np.expand_dims(dphi, -1))
    err = 0
    for k in range(2):
        err += meansq( R@(np.expand_dims(vpair1[k], -1)) - np.expand_dims(vpair2[k], -1), axis=(-1,-2))
    assert err.shape + (3,3) == R.shape

    if ang_type == AngType.DEG:
        dphi = np.rad2deg(dphi)
    return dphi, R, err


def rotvec2rot(rotvec: ArrayLike, # *x3 (default), rotation vector (angular velocity) in radian
               axis:   int = -1,   # avel.shape[axis] == 3
               homog:  Optional[bool] = False
              )     -> ArrayLike: # *x3x3 (default), *x3x3x* (general)
    ## Rodrigues's rotation formula
    ## u := normalize(rotvec), theta := norm(rotvec)
    ## return = I cos(theta) + [u]_x sin(theta) + uu^T (1-cos(theta))
    rotvec = np.asarray(rotvec)
    if rotvec.shape[axis] != 3:
        raise ValueError(
            f"The size along given axis({axis}) should be equal to 3, but currently shape:{rotvec.shape}.")
    rotvec = np.swapaxes(rotvec, axis, -1)
    norm = np.expand_dims(np.linalg.norm(rotvec, axis=-1, keepdims=True), axis=-1)
    I = np.eye(3)

    ## NOTE: np.sinc(x) == sin(pi*x)/(pi*x)
    sinon = np.sinc(norm/np.pi) # "sin over norm", sin(theta)/norm
    ## NOTE: (1 - cos x) / x^2 == [2 sin^2(x/2)] / x^2 == 1/2 * sin^2(x/2) / (x/2)^2
    omcoson2 = 1/2 * (np.sinc(norm/2/np.pi)**2) # "one minus cos over norm sq", (1-cos(theta)) / norm^2

    R = I*np.cos(norm) + cross_mat(rotvec)*sinon + outer_mat(rotvec)*omcoson2
    if homog:
        shape = list(R.shape)
        shape[-2:] = (4, 4)
        R_new = np.zeros(shape, R.dtype)
        R_new[..., :3, :3] = R
        R_new[..., 3, 3] = 1
        R = R_new
        
    oaxis = axis
    if axis < 0:
        oaxis -= 1
    R = np.moveaxis(np.swapaxes(R, -2, oaxis), -1, oaxis+1)
    return R

def axisang2rot(unit:  ArrayLike, # *x3      (default), normalized rotation axis
                angle: ArrayLike, # * or *x1 (default), rotation angle in radian
                axis:  int = -1,   # unit.shape[axis] == 3
                homog: Optional[bool] = False
               )     -> ArrayLike: # *x3x3 (default), *x3x3x* (general)
    '''
    When `axis == -1`
    unit[3], angle[*]
    unit[*, 3], angle[]
    TODO polish comment
    '''
    unit = np.asarray(unit); angle = np.asarray(angle)
    if unit.shape[axis] != 3:
        raise ValueError(
            f"The size along given axis({axis}) should be equal to 3, but currently shape:{unit.shape}.")
    unit = np.swapaxes(unit, axis, -1)
    if angle.ndim == 0 or angle.shape[axis] != 1:
        assert angle.ndim == unit.ndim-1 or unit.ndim == 1, \
            f"`angle.ndim` should be equal to `unit.ndim` or `unit.ndim-1`, or `unit.ndim` be equal to 1, but currently: unit.shape={unit.shape} and angle.shape={angle.shape}"
        angle = np.expand_dims(angle, axis=axis)
    angle = np.swapaxes(angle, axis, -1)
            # raise ValueError(
            #     f"The size along given axis({axis}) should be equal to 1, but currently shape:{angle.shape}.")
        
    angle = np.expand_dims(angle, axis=-1)
    ## Now unit[...,3], angle[...,1,1]

    I = np.eye(3)
    R = I*np.cos(angle) + cross_mat(unit)*np.sin(angle) + outer_mat(unit)*(1-np.cos(angle))
    if homog:
        shape = list(R.shape)
        shape[-2:] = (4, 4)
        R_new = np.zeros(shape, R.dtype)
        R_new[..., :3, :3] = R
        R_new[..., 3, 3] = 1
        R = R_new

    oaxis = axis
    if axis < 0:
        oaxis -= 1
    R = np.moveaxis(np.swapaxes(R, -2, oaxis), -1, oaxis+1)
    return R

def rotx(angle: ArrayLike, axis: Optional[int] = -1, homog: Optional[bool] = False) -> ArrayLike:
    return axisang2rot(unit_x, angle, axis=axis, homog=homog)
def roty(angle: ArrayLike, axis: Optional[int] = -1, homog: Optional[bool] = False) -> ArrayLike:
    return axisang2rot(unit_y, angle, axis=axis, homog=homog)
def rotz(angle: ArrayLike, axis: Optional[int] = -1, homog: Optional[bool] = False) -> ArrayLike:
    return axisang2rot(unit_z, angle, axis=axis, homog=homog)

def rot2d(angle: ArrayLike) -> ArrayLike: # angle in radian
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s], [s, c]])
rot2d90 = np.array([[0, -1], [1, 0]])

def normal2rotvec(normal:    ArrayLike, # [*, 3]
                  axis:      int = -1,
                 ) ->        ArrayLike: # [*, 3]
    '''
    Take one rotvec which satisfy:
    `R @ unit_z == normal`
    NOTE This function just pick one result among infinite solutions.
    '''
    theta, phi = vec2sph(normal, axis=axis)
    rotvec = quat.array.from_euler_angles(phi, theta, 0).to_axis_angle
    return np.moveaxis(rotvec, -1, axis)

def normal2rot(normal: ArrayLike, # [*, 3] (default)
               axis:   int = -1   # avel.shape[axis] == 3
              )     -> ArrayLike: # [*, 3, 3] (default), [*, 3, 3, *] (general)
    '''
    Take one 3x3 rotation matrix which satisfy:
    `R @ unit_z == normal`
    NOTE This function just pick one result among infinite solutions.
    '''
    rotvec = normal2rotvec(normal, axis=axis)
    return rotvec2rot(rotvec, axis=-1)

def scipyRotaion2quat(R: scipyRotation) -> quat.array:
    """
    NOTE `scipy` uses [x, y, z, w] convention but `quaternionic` uses [w, x, y, z] for quaternions.
    """
    return quat.array(np.roll(R.as_quat(), 1, axis=-1))

RotationLike = Union[ArrayLike, scipyRotation, quat.array]

def rotation2scipy(rotation: Union[ArrayLike, scipyRotation]) -> scipyRotation:
    if isinstance(rotation, scipyRotation):
        R = rotation
    else:
        rotation = np.asarray(rotation)
        if rotation.shape == (3,):
            R = scipyRotation.from_rotvec(rotation)
        elif rotation.shape == (3, 3):
            R = scipyRotation.from_matrix(rotation)
        else:
            raise ValueError(f"Invalid shape: {rotation.shape = }, {rotation}")
    return R

def rotation2quat(rotation: RotationLike) -> quat.array:
    if isinstance(rotation, quat.array):
        Q = rotation
    elif isinstance(rotation, scipyRotation):
        Q = scipyRotaion2quat(rotation)
    else:
        rotation = np.asarray(rotation)
        if rotation.shape == (3,):
            Q = quat.array.from_rotation_vector(rotation)
        elif rotation.shape == (3, 3):
            Q = quat.array.from_rotation_matrix(rotation)
        else:
            raise ValueError(f"Invalid shape: {rotation.shape = }, {rotation}")
    return Q

'''
Coordinates conversion
'''
def sph2vec(theta:    ArrayLike, # *, zenith
            phi:      ArrayLike, # *, azimuth
            axis:     int = -1,  # stack axis
            ang_type: AngType = AngType.RAD,
           )       -> ArrayLike: # *x3 (default)
    theta, phi = radian_regularize(ang_type, theta, phi)
    sinth = np.sin(theta)
    costh = np.cos(theta)
    X = sinth * np.cos(phi)
    Y = sinth * np.sin(phi)
    Z = costh
    X, Y, Z = np.broadcast_arrays(X, Y, Z)
    return np.stack([X, Y, Z], axis=axis)


def vec2sph(vec:      ArrayLike, # *x3 (default), not assumed normalized
            axis:     int = -1,
            ang_type: AngType = AngType.RAD
           )       -> Tuple[ArrayLike, ArrayLike]: # (theta[*], phi[*])
    vec = np.asarray(vec)
    if vec.shape[axis] != 3:
        raise ValueError(f"The size along given axis({axis}) should be equal to 3, but curretly shape:{vec.shape}.")
    vec = normalize(vec, axis)
    vec = np.swapaxes(vec, axis, -1)
    X = vec[...,0]
    Y = vec[...,1]
    Z = vec[...,2]
    theta = np.arccos(Z)
    phi = np.arctan2(Y, X)
    return radian_deregularize(ang_type, theta, phi)

def xyz2sph(x: ArrayLike, y: ArrayLike, z: ArrayLike, ang_type: AngType = AngType.RAD,
            broadcast: Optional[bool] = False
           ) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    if broadcast:
        x, y, z = np.broadcast_arrays(x, y, z)
    return vec2sph(np.stack([x, y, z], axis=-1), axis=-1, ang_type=ang_type)

def halfvec(vec_i, vec_o, theta_i, phi_i, *, axis=-1):
    vec_h = vec_i+vec_o
    norm = np.linalg.norm(vec_h, axis=axis, keepdims=True)

    ## NOTE When `vec_i == -vec_o` exactly, vec_h could be any vector orthogonal to `vec_i`
    mask = norm == 0.0
    vec_h = np.where(mask,
                     sph2vec(theta_i - np.pi/2, phi_i),
                     np.divide(vec_h, norm, where=~mask))
    
    ## NOTE Handling `vec_i \approx -vec_o` for numerical stability
    ## See `2022.08.30. Factorizing 3-parameter BSDF.md`, "Direct once, correct after"
    vec_oi = vec_o - vec_i
    A = (vec_oi*vec_h).sum(axis, keepdims=True)
    B = -(vec_i*vec_oi).sum(axis, keepdims=True)
    vec_h2 = A * vec_i +  B * vec_h
    norm = np.linalg.norm(vec_h2, axis=-1, keepdims=True)
    mask = norm != 0.0
    vec_h = np.where(mask, np.divide(vec_h2, norm, where=mask),
                           vec_h)
    return vec_h

def sph2rus(theta_i:   ArrayLike,
            theta_o:   ArrayLike,
            phi_i:     ArrayLike,
            phi_o:     ArrayLike = None,
            ang_type:  AngType = AngType.RAD,
           )        -> ArrayLike:
    if phi_o is None:
        flag_phio_given = False
        phi_o = np.zeros(np.shape(phi_i))
    else:
        flag_phio_given = True
    theta_i, phi_i, theta_o, phi_o = radian_regularize(ang_type,
                                    theta_i, phi_i, theta_o, phi_o)
    theta_i = np.asarray(theta_i)
    theta_o = np.asarray(theta_o)
    phi_i = np.asarray(phi_i)
    phi_o = np.asarray(phi_o)
    shape = np.broadcast_shapes(theta_i.shape, theta_o.shape, phi_i.shape, phi_o.shape)
    # vec_i
    sin_theta_i = np.sin(theta_i)
    vec_i = np.stack([sin_theta_i*np.cos(phi_i),
                      sin_theta_i*np.sin(phi_i),
                      np.cos(theta_i)], axis=-1)
    # vec_o, vec_h
    vec_o = sph2vec(theta_o, phi_o)
    vec_h = halfvec(vec_i, vec_o, theta_i, phi_i)
    
    theta_h_rad, phi_h_rad = vec2sph(vec_h)
    vec_d = np.empty_like(vec_i)
    
    theta_imh = theta_i-theta_h_rad
    phi_imh = phi_i-phi_h_rad
    sin2_half_phi_imh = np.broadcast_to(phi_imh / 2, shape)
    if not sin2_half_phi_imh.flags.writeable:
        sin2_half_phi_imh = np.array(sin2_half_phi_imh)
    np.sin(sin2_half_phi_imh, out=sin2_half_phi_imh)
    np.square(sin2_half_phi_imh, out=sin2_half_phi_imh)
    sin2_half_phi_imh *= -2
    sin2_half_phi_imh *= sin_theta_i
    # vec_d[x]
    np.cos(theta_h_rad, out=vec_d[..., 0])
    vec_d[..., 0] *= sin2_half_phi_imh
    vec_d[..., 0] += np.sin(theta_imh)
    # vec_d[y]
    np.sin(phi_imh, out=vec_d[..., 1])
    vec_d[..., 1] *= sin_theta_i
    # vec_d[z]
    np.sin(theta_h_rad, out=vec_d[..., 2])
    vec_d[..., 2] *= sin2_half_phi_imh
    vec_d[..., 2] += np.cos(theta_imh)
    theta_d_rad, phi_d_rad = vec2sph(vec_d)

    if flag_phio_given:
        return radian_deregularize(ang_type, phi_d_rad, theta_d_rad, theta_h_rad, phi_h_rad)
    else:
        return radian_deregularize(ang_type, phi_d_rad, theta_d_rad, theta_h_rad)

def rus2sph(phi_d:    ArrayLike,
            theta_d:  ArrayLike,
            theta_h:  ArrayLike,
            phi_h:    ArrayLike = None,
            ang_type: AngType   = AngType.RAD
           )       -> List[ArrayLike]:
    """
    return: (theta_i, theta_o, phi_i, phi_o) if `phi_h is None`
            (theta_i, theta_o, phi_i-phi_o)  otherwise
    
    Convention: omega_i points material to source
    """
    veci, veco = rus2vec(phi_d, theta_d, theta_h, phi_h, ang_type=ang_type)
    thetai, phii = vec2sph(veci, ang_type=ang_type)
    thetao, phio = vec2sph(veco, ang_type=ang_type)
    if phi_h is None:
        return thetai, thetao, phii - phio
    else:
        return thetai, thetao, phii, phio

def rus2vec(phi_d:    ArrayLike,
            theta_d:  ArrayLike,
            theta_h:  ArrayLike,
            phi_h:    ArrayLike = None,
            axis:     int = -1,  # stack axis
            ang_type: AngType = AngType.RAD,
           )       -> Tuple[ArrayLike, ArrayLike]:
    ## Convention: omega_i points material to source
    if phi_h is None:
        phi_h = 0
    phi_d, theta_d, theta_h, phi_h = np.broadcast_arrays(*radian_regularize(ang_type, phi_d, theta_d, theta_h, phi_h))
    def unsq(a):
        return np.expand_dims(a, axis)
    
    # [Variable reuse] vech -> vech_cosd
    vech_cosd = sph2vec(theta_h, phi_h, axis=axis)
    vech_cosd *= unsq(np.cos(theta_d))

    pihalf = np.pi/2
    # [Variable reuse] pthetah -> vecd
    vecd = sph2vec(theta_h + pihalf, phi_h, axis=axis)
    vecd *= unsq(np.cos(phi_d))
    pphih = sph2vec(pihalf, phi_h + pihalf, axis=axis)
    pphih *= unsq(np.sin(phi_d))
    vecd += pphih
    vecd *= unsq(np.sin(theta_d))

    veci = vech_cosd + vecd
    # veco = vech_cosd - vecd # Variable reuse
    vech_cosd -= vecd
    
    # return veci, veco # Variable reuse
    normalize(veci, axis=axis, out=veci)
    normalize(vech_cosd, axis=axis, out=vech_cosd)
    return veci, vech_cosd

def sph2Ftp(theta:    ArrayLike,                  # [*]
            phi:      ArrayLike,                  # [*]
            vec:      Optional[ArrayLike] = None, # [*, 3]
            ang_type: Optional[AngType] = AngType.RAD
           ) ->       np.ndarray:                 # [*, 3, 3]
    '''
    Return the matrix spherical coordinates frame,
    
    i.e. Fsp[*,:,0] == \\hat theta
         Fsp[*,:,1] == \\hat phi
         Fsp[*,:,2] == vec
    
    Discontinuity rather than NaN
           (using np.atan2(0, 0))
    '''
    if vec is None:
        vec = normalize(sph2vec(theta, phi, ang_type=ang_type))
    pi_half = np.pi/2 if ang_type == AngType.RAD else 90.0
    hat_theta = sph2vec(theta+pi_half, phi, ang_type=ang_type)
    hat_phi = normalize(np.cross(vec, hat_theta))

    return np.stack([hat_theta, hat_phi, vec], axis=-1)

def vec2Ftp(vec: ArrayLike  # [*, 3]
           )  -> ArrayLike: # [*, 3, 3]
    '''
    Return the matrix spherical coordinates frame,
    
    i.e. Fsp[*,:,0] == \\hat theta
         Fsp[*,:,1] == \\hat phi
         Fsp[*,:,2] == vec
    
    Discontinuity rather than NaN
           (using np.atan2(0, 0))
    '''
    theta, phi = vec2sph(vec) # radian (since default)
    hat_theta = sph2vec(theta+np.pi/2, phi)
    hat_phi = np.cross(vec, hat_theta)
    
    x = hat_theta
    y = normalize(hat_phi, axis=-1)
    z = normalize(vec, axis=-1)
    return np.stack([x,y,z], axis=-1)

def vec2Fspu(vec: ArrayLike  # [*, 3]
            ) ->  ArrayLike: # [*, 3, 3]
    '''
    Return s-p-u frame
    (s = normalize(n \times u) where n=[0,0,1] and u=vec)
    '''
    theta, phi = vec2sph(vec) # radian
    hat_theta = sph2vec(theta+np.pi/2, phi)
    hat_phi = np.cross(vec, hat_theta)
    
    x = normalize(hat_phi, axis=-1)
    y = -hat_theta
    z = normalize(vec, axis=-1)
    return np.stack([x, y, z], axis=-1)

def sph2Fgeo(theta: ArrayLike, # [*]
             phi:   ArrayLike  # [*]
            ) ->    ArrayLike: # [*,3,3]
    F = rotz(phi) @ roty(theta) @ rotz(-phi)
    return F

def vec2Fpersp(vec: ArrayLike,  # [*, 3]
               up:  ArrayLike,  # [*, 3]
              ) ->  np.ndarray: # [*, 3, 3]
    Fx = normalize(np.cross(up, vec))
    Fy = normalize(np.cross(vec, Fx))
    F = np.stack([Fx, Fy, vec], -1)
    return F

def vec2Fcube(vec: ArrayLike,  # [*, 3]
             ) ->  np.ndarray: # [*, 3, 3]
    """
    TODO: Add an argument `to_world`.
    """
    x, y, z = npunstack(vec)
    xy_max = np.maximum(np.abs(x), np.abs(y))

    mask1 = z >= xy_max  # Ray propagates from ground to the sky
    mask2 = z <= -xy_max
    up = np.empty_like(vec)
    up[mask1] = [0, 1, 0]
    up[mask2] = [0, -1, 0]
    up[~(mask1 | mask2)] = [0, 0, 1]
    return vec2Fpersp(vec, up)

def vec2FTD17(vec: ArrayLike) -> np.ndarray:
    """
    A numerically stable moving frame.

    Reference:
    * "Building an Orthonormal Basis, Revisited" by
       Tom Duff, James Burgess, Per Christensen, Christophe Hery,
       Andrew Kensler, Max Liani, and Ryusuke Villemin
       (JCGT Vol 6, No 1, 2017)
    * func:`coordinate_system` in
        https://github.com/mitsuba-renderer/mitsuba3/blob/master/include/mitsuba/core/vector.h

    params:
        vec: [*, 3], the last axis corresponds to x, y, and z.
    returns:
        [*, 3, 3]
    """
    x, y, z = npunstack(vec, axis=-1)
    sign = np.where(z >= 0, 1.0, -1.0)
    a = -np.reciprocal(sign + z)
    b = x * y * a
    Fx = np.stack([sign * np.square(x) * a + 1,
                    sign * b,
                    -sign * x], -1)
    Fy = np.stack([b,
                    a * np.square(y) + sign,
                    -y], -1)
    return np.stack([Fx, Fy, vec], -1)


def fibonacci_sphere(n_samples: int,
                     hemi:      str = None, # None | 1 == 'upper' | -1 == 'lower'
                     out_type:  str = 'vec' # 'vec' | 'sph'
                     ) ->       Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    '''
    Return n_samples deterministic samples on the upper hemisphere (||x,y,z|| = 1, z >= 0)
    
    Parameters
        n_samples: int
        hemi:      str
        out_type: 'vec' (default) or 'sph'
    Returns
        vec: np.ndarray[n_samples, 3]                if out_type == 'vec'
        theta, phi: np.ndarray[n_samples] (for each) if out_type == 'sph'
            radians
        vec, theta, phi                              if out_tpye == 'both'

    Source: modified from https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    '''
    ratio = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    I = np.arange(n_samples)

    if hemi is None:
        z = 1 - (I / float(n_samples - 1)) * 2  # goes from 1 to -1
    elif hemi == 1 or hemi.lower() == 'upper':
        z = 1 - (I / float(2*n_samples - 1)) * 2  # goes from 1 to 0
    elif hemi == -1 or hemi.lower() == 'lower':
        z = (I / float(2*n_samples - 1)) * 2  # goes from 1 to 0
    else:
        raise ValueError()
    
    phi = ratio * I  # golden angle increment

    out_type = out_type.lower()
    if out_type in ['vec', 'both']:
        radius = np.sqrt(1 - z * z)  # radius at z    

        x = np.cos(phi) * radius
        y = np.sin(phi) * radius

        points = np.stack([x, y, z], 1)
    if out_type in ['sph', 'both']:
        theta = np.arccos(z)
    
    if out_type == 'vec':
        return points
    elif out_type == 'sph':
        return theta, phi
    elif out_type == 'both':
        return points, theta, phi
    else:
        raise ValueError(f"The argument `out_type` must be either 'vec' or 'sph', but {out_type} is given.")

def linspace_tp(h: int, w: int
               ) -> Tuple[np.ndarray, np.ndarray]: # theta[h], phi[w]
    theta = np.linspace(0, np.pi, h, endpoint=False) + np.pi/2/h
    phi = np.linspace(0, 2*np.pi, w, endpoint=False) + np.pi/w
    return theta, phi
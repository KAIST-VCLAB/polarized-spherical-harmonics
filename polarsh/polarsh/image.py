import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
from pathlib import Path
from typing import Union, Optional
import numpy as np
from numpy.typing import ArrayLike
# import skimage
# import skimage.io

from polarsh.array import common_dtype
from polarsh.sphere import rotx, roty, rotz

def flip_RGB(img: np.ndarray) -> np.ndarray:
    """
    Flip RGB(A) channels to BGR(A), vice versa.
    If given image consists of a signel channel then return the image itself.
    """
    match img.ndim:
        case 2:
            return img
        case 3:
            match img.shape[2]:
                case 3:
                    return img[:, :, ::-1]
                case 4:
                    return np.concatenate([img[:,:,2::-1], img[:,:,3:4]], -1)
                case _:
                    raise ValueError(f"Invalid shape: {img.shape}.")
        case _:
            raise ValueError(f"Invalid ndim: {img.shape}.")

def imwrite(filename: Union[str, Path], img: ArrayLike, normalize: Optional[bool] = False, gm_png: Optional[bool] = True):
    # SYYI appended
    filename = str(filename)
    if img.ndim == 3 and img.shape[-1] >= 6 :
        # ---------- Polarized image concatenated along channel axis ----------
        filename_split = filename.split('.')
        for i in range(img.shape[-1]//3):
            temp_split = filename_split[:]
            temp_split[-2] += f'_ch{i}'
            imwrite('.'.join(temp_split), img[:,:,3*i:3*(i+1)], normalize)
        return
    # SYYI end

    directory = os.path.dirname(filename)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    dtype = img.dtype
    if normalize:
        img_rng = np.max(img) - np.min(img)
        if img_rng > 0:
            img = (img - np.min(img)) / img_rng

    if filename[-4:] == '.exr':
        if len(img.shape) == 2:
            img = img.reshape((img.shape[0], img.shape[1], 1))
        if img.shape[2] == 1:
            img = np.tile(img, (1, 1, 3))
        if dtype.kind in ["u", "i"]:
            img = img / 255
            if gm_png:
                img = img ** (1/2.2)
        return cv2.imwrite(filename, flip_RGB(img).astype(np.float32))
    else: # '.png', etc.
        if dtype.kind == 'f':
            img = np.clip(img, 0.0, 1.0)
            if gm_png:
                img = np.power(img, 1.0/2.2)
            img *= 255
        
        return cv2.imwrite(filename, (flip_RGB(img)).astype(np.uint8))
        # skimage.io.imsave(filename, (img2*255).astype(np.uint8), check_contrast=False)
        ## [NOTE] We sometimes save zero images for s1, s2, s3 components to be consistent for output format.
        ## Thus, we turn off low contrast image warning from `scikit-image`.

def imread(filename: Union[str, Path], gm_png: Optional[bool] = True) -> np.ndarray:
    filename = str(filename)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Such file does not exists: {filename}")
    if (filename[-4:] == '.exr'):
        res_numpy = flip_RGB(cv2.imread(filename, -1))
        if res_numpy is None:
            raise RuntimeError(f"{filename = }")
        return res_numpy
        # return torch.from_numpy(res_numpy)
    else:
        # im = skimage.io.imread(filename)
        im = flip_RGB(cv2.imread(filename, -1))
        if im.ndim == 2:
            im = np.stack([im, im, im], axis=-1)
        elif im.shape[2] == 4:
            im = im[:, :, :3]
        img = im.astype(np.float32) / 255
        if gm_png:
            res_numpy = np.power(img, 2.2)
        else:
            res_numpy = img
        return res_numpy
        # return torch.from_numpy(res_numpy)

def imread_Stk(filename:     Union[str, Path], # must contain "%d"
               n_polar_comp: Optional[int] = 4, # 1, 2, 3, 4
               *,
               base_idx:     Optional[int] = 0,
               gm_png:       Optional[bool] = True
              ) ->           np.ndarray: # [h, w, *c, n_polar_comp]
    """
    Read Stokes component images from file name including "%d".

    As a default, s0 - s3 components correspond to filename `"%d" % 1` to `"%d" % 4`, respectively.

    NOTE: Index starting from 1 rather than 0 comes from that
          when saving rendered image from Mitsuba 3, S0 components duplicated twice.
    """
    filename = str(filename)
    comp_list = []
    for i in range(base_idx, base_idx + n_polar_comp):
        comp_list.append(imread(filename % i, gm_png=gm_png))
    return np.stack(comp_list, -1)

def imwrite_Stk(filename:     Union[str, Path], # must contain "%d"
                stk_image:    np.ndarray, # [h, w, *c, p]
                *,
                base_idx:     Optional[int] = 0,
                normalize:    Optional[bool] = False,
                gm_png:       Optional[bool] = True
               ) ->           np.ndarray: # [h, w, *c, n_polar_comp]
    """
    Write Stokes component images to file name including "%d".
    The last axis for `stk_image` indicates each Stokes component

    As a default, s0 - s3 components correspond to filename `"%d" % 1` to `"%d" % 4`, respectively.

    NOTE: Index starting from 1 rather than 0 comes from that
          when saving rendered image from Mitsuba 3, S0 components duplicated twice.
    """
    n_polar_comp = stk_image.shape[-1]
    for i in range(n_polar_comp):
        imwrite(str(filename) % (i+base_idx), stk_image[..., i], normalize=normalize, gm_png=gm_png)



'''
         Up
    Left Front Right Back
         Down
        +Z
    -X  +Y  +X  -Y (m2s convention)
        -Z
    world x,y,z == Right, Front, Up
'''

env_flags = [[False, True, False, False],
             [True,  True, True,  True],
             [False, True, False, False]]

env_tfs = [[None,          rotx(np.pi/2),  None,           None],
           [rotz(np.pi/2), np.eye(3),      rotz(-np.pi/2), rotz(np.pi)],
           [None,          rotx(-np.pi/2), None,           None]]
tf_front = rotx(np.pi/2)
for row in env_tfs:
    for i,tf in enumerate(row):
        if not tf is None:
            row[i] = tf @ tf_front
## Now 'env_tf' is a transform which
##              maps frame generated by 'SphereFrameField_from_persp'
##              to env. frame describe above

def envmap_stack(envimg: ArrayLike  # [3edge, 4edge,  *]
                )     -> ArrayLike: # [6, edge, edge, *]
    """
    Convert a cubemap image shape from [3*edge, 4*edge, *] to [6, edge, edge, *].
    """
    h, w = envimg.shape[:2]
    assert h%3 == 0 and w%4 == 0
    h_sq = h//3; w_sq = w//4
    assert h_sq == w_sq

    sqimg_list = []
    for i,row in enumerate(env_flags):
        for j,flag in enumerate(row):
            if flag:
                i0 = i*h_sq; i1 = (i+1)*h_sq
                j0 = j*w_sq; j1 = (j+1)*w_sq
                sqimg_list.append(envimg[i0:i1, j0:j1, ...])
    return np.stack(sqimg_list)

def envmap_unfold(siximg: ArrayLike,  # [6,h,w,*]
                  fill:   Optional[float] = 0.0
                 )     -> ArrayLike: # [3h,4w,*]
    """
    Convert a cubemap image shape from [6, edge, edge, *] to [3*edge, 4*edge, *].
    """
    assert siximg.shape[0] == 6
    h, w = siximg.shape[1:3]
    assert h == w

    envimg = np.full((3*h, 4*w)+siximg.shape[3:], fill, dtype=common_dtype(siximg, fill))
    idx_six = 0
    for i,row in enumerate(env_flags):
        for j,flag in enumerate(row):
            if flag:
                i0 = i*h; i1 = (i+1)*h
                j0 = j*w; j1 = (j+1)*w
                envimg[i0:i1, j0:j1, ...] = siximg[idx_six, ...]
                idx_six += 1
    return envimg


def convert_er_convention(img: ArrayLike   # [h, w, ...]
                         ) ->  np.ndarray: # [h, w, ...]
    """
    Convert [h, w, *] equirectangule image into a different coordinate convention.
    This function is the self-inverse so that we may not care about the direction of conversion,
    equilib -> polarsh or polarsh -> equilib
    """
    h, w = img.shape[:2]
    img = img[::-1, ::-1]
    w4 = w//4
    return np.concatenate((img[:, -w4:], img[:, :-w4]), 1)

def cubemap_hw2edge(h: int, w: int):
    edge = h//3
    assert h == 3*edge and w == 4*edge
    return edge

def cubemap_edge2hw(edge: int):
    h, w = 3*edge, 4*edge
    return h, w
from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union, Literal, Callable
import importlib
import IPython.display
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as scipyRotation
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import Colormap, ListedColormap, LinearSegmentedColormap, Normalize, CenteredNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import IPython

import vispy
import vispy.scene
import vispy.io

from .array import vec2comp, powmag, matmul_vec1d
from .sphere import rotation2quat
from .SH import CodType
# https://stackoverflow.com/questions/6028000/how-to-read-a-static-file-from-inside-a-python-package
from . import resource

Colormap2DList = ['yuv']


def rgb2gray(img:       ArrayLike,
             axis:      int = -1,
             keep_dims: bool = False
            ) ->        ArrayLike:
    '''
    Also it must be r=0, g=1, b=2.
    https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_rgb_to_gray.html
    '''
    assert img.shape[axis] == 3, f"The length of the shape {img.shape} along axis {axis} should be equal to 3."
    img = np.moveaxis(img, axis, -1)
    y = 0.2125 * img[...,0] + 0.7154 * img[..., 1] + 0.0721 * img[...,2]
    if keep_dims:
        y = y[..., None]
        y = np.moveaxis(y, -1, axis)
    return y

def gamma_signed(val, gamma=1/2.2):
        val_abs = np.abs(val)
        return val_abs**gamma * np.sign(val)

cdict = {
    'red': (
        (0.0,  1.0, 1.0),
        (0.5,  0.0, 0.0),
        (1.0,  0.15, 0.15),
    ),
    'green': (
        (0.0,  0.15, 0.15),
        (0.5,  0.0, 0.0),
        (1.0,  1.0, 1.0),
    ),
    'blue': (
        (0.0,  0.15, 0.15),
        (0.5,  0.0, 0.0),
        (1.0,  0.15, 0.15),
    )
}
pltcmap_rkg = LinearSegmentedColormap('rkg_colormap', segmentdata=cdict, N=4096)

##################################################
### Colormap from EdolView
rgb2xyz_mat = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]]
).T

xyz2rgb_mat = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252]]
).T

# d65 white reference
xn = 0.95047
yn = 1.0
zn = 1.08883

def gamma_f(rgb):
    return np.power(rgb, 1/2.2)

def inv_gamma_f(rgb):
    return np.power(rgb, 2.2)

def f(c):
    return np.where(c > 0.008856, np.power(c, 1.0 / 3.0), (903.3 * c + 16.0) / 116.0)

def xyz2lab(xyz):
    fx = f(xyz[...,0] / xn)
    fy = f(xyz[...,1] / yn)
    fz = f(xyz[...,2] / zn)
    return np.stack([
    116.0 * fx - 16.0,
    500.0 * (fx - fy),
    200.0 * (fy - fz)
    ], -1)

def f_inv(c):
    t = np.power(c, 3.0);
    return np.where(t > 0.008856, t, (116.0 * c - 16.0) / 903.3)

def lab2xyz(lab):
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    fy = (L + 16.0) / 116.0
    fz = fy - b / 200.0
    fx = a / 500.0 + fy

    return np.stack([f_inv(fx) * xn, f_inv(fy) * yn, f_inv(fz) * zn], -1)

def rgb2xyz(rgb):
    return inv_gamma_f(rgb) @ rgb2xyz_mat

def xyz2rgb(xyz):
    return gamma_f(xyz @ xyz2rgb_mat)

def lab2msh(lab):
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    M = np.sqrt(np.sum(np.power(lab, 2)))

    return np.stack([M, np.arccos(L / M), np.arctan2(b, a)], -1)

def msh2lab(msh):
	M = msh[..., 0]
	s = msh[..., 1]
	h = msh[..., 2]

	return np.stack(
		[M * np.cos(s),
		M * np.sin(s) * np.cos(h),
		M * np.sin(s) * np.sin(h)],
        -1
	)

def rgb2msh(rgb):
    return lab2msh(xyz2lab(rgb2xyz(rgb)))

def msh2rgb(msh):
	return xyz2rgb(lab2xyz(msh2lab(msh)))

def diverge_colormap(t):
	# low_color = np.array([0.230, 0.299, 0.754])
	# high_color = np.array([0.706, 0.016, 0.150])
	low_msh = np.array([80.0154, 1.0797, -1.1002])
	high_msh = np.array([80.0316, 1.0798, 0.5008])
	mid_msh = np.array([88.0, 0.0, 0.0])
	#// vec3 mid_msh = rgb2msh(vec3(0.8654, 0.8654, 0.8654));

	interp = np.abs(1.0 - 2.0 * t)

	#// vec3 msh = rgb2msh(t < 0.5 ? low_color : high_color);
	msh = np.where(t[..., None] < 0.5, low_msh, high_msh)
	msh[..., 0] = interp * msh[..., 0] + (1.0 - interp) * mid_msh[..., 0]
	msh[..., 1] = interp * msh[..., 1] + (1.0 - interp) * mid_msh[..., 1]
	rgb = msh2rgb(msh)
	return np.clip(rgb, 0.0, 1.0)

# https://www.kennethmoreland.com/color-maps/
def diverge2_colormap(t):
	low_msh = rgb2msh(np.array([0.436, 0.308, 0.631]))
	high_msh = rgb2msh(np.array([0.759, 0.334, 0.046]))
	mid_msh = np.array([88.0, 0.0, 0.0])

	interp = np.abs(1.0 - 2.0 * t)

	msh = np.where(t[..., None] < 0.5, low_msh, high_msh)
	msh[..., 0] = interp * msh[..., 0] + (1.0 - interp) * mid_msh[..., 0]
	msh[..., 1] = interp * msh[..., 1] + (1.0 - interp) * mid_msh[..., 1]
	rgb = msh2rgb(msh)
	return np.clip(rgb, 0.0, 1.0)

def diverge3_colormap(t):
	low_msh = rgb2msh(np.array([0.085, 0.532, 0.201]))
	high_msh = rgb2msh(np.array([0.436, 0.308, 0.631]))
	mid_msh = np.array([88.0, 0.0, 0.0])

	interp = np.abs(1.0 - 2.0 * t)

	msh = np.where(t[..., None] < 0.5, low_msh, high_msh)
	msh[..., 0] = interp * msh[..., 0] + (1.0 - interp) * mid_msh[..., 0]
	msh[..., 1] = interp * msh[..., 1] + (1.0 - interp) * mid_msh[..., 1]
	rgb = msh2rgb(msh)
	return np.clip(rgb, 0.0, 1.0)

pltcmap_diverge = ListedColormap(diverge_colormap(np.linspace(0, 1, 4096)), "diverge_colormap")
### End: Colormap from EdolView
##################################################
_vis_options = {'backend': 'matplotlib', 'cmap': pltcmap_diverge}
##################################################

def apply_colormap_sgn(img:  ArrayLike,
                       axis:  Optional[Union[int, None]] = -1, # axis for color channel. `None` for grayscale input
                       scale: Optional[float] = 1.0,
                       gamma: Optional[float] = 1.0, # 1/2.2, not 2.2
                       cmap:  Optional[Union[str, Colormap]] = None
                      ) ->    np.ndarray:
    # ---------- Default arguments ----------
    if cmap is None: cmap = _vis_options['cmap']

    # ---------- Main ----------
    if axis is None:
        gray = img * scale
    else:
        gray = rgb2gray(img, axis=axis) * scale
    if gamma != 1.0:
        gray = gamma_signed(gray, gamma=gamma)
    gray = np.clip((gray+1)/2, 0, 1)
    return np.asarray(cmap(gray))

def apply_colormap2d(img:   np.ndarray,
                     axis:  int = -1,
                     gamma: float = 1.0,
                     cmap:  str = 'yuv'
                    ) ->    np.ndarray:
    '''
    If `img` is realobj, `img[axis] == 2` and `res[axis] == 3`.
    If `img` is complexobj, `res[axis] == 3`
    -----
    Examples:
    [h,w] comp -> [h,w,3] real, when axis == 2 or -1
    [h,w,2] real -> [h,w,3] real, when axis == 2 or -1
    '''
    if np.isrealobj(img):
        img_comp = vec2comp(img, axis)
    else:
        img_comp = img

    if gamma != 1.0:
        img_comp = powmag(img_comp, gamma)

    cmap = cmap.lower()
    assert cmap in Colormap2DList, f"Given color map cmap='{cmap}' is not supported."
    if cmap == 'yuv':
        # YUV mapping
        _y = np.ones_like(img_comp.real[...,None]) * 0.5
        if False: # old version: adaptive color for zero and asymmetric bounds for real and imag
            _u = 0.436 * (2 * (img_comp.real[...,None] - img_comp.real.min()) / (img_comp.real.max() - img_comp.real.min()) - 1)
            _v = 0.615 * (2 * (img_comp.imag[...,None] - img_comp.imag.min()) / (img_comp.imag.max() - img_comp.imag.min()) - 1)
        if True: # new version: fixed zero color, same bounds for real and img
            bd = max(-np.nanmin(img_comp.real), np.nanmax(img_comp.real), -np.nanmin(img_comp.imag), np.nanmax(img_comp.imag))
            _u = 0.436 * img_comp.real[...,None] / bd
            _v = 0.615 * img_comp.imag[...,None] / bd
        yuv = np.concatenate((_y, _u, _v), axis=-1)
        yuv_ = yuv.reshape(-1, 3)
        yuv2rgb = np.array([[1, 0, 1.28033],[1, -0.21482, -0.38059],[1, 2.12798, 0]])
        _rgb = yuv2rgb @ yuv_.T
        rgb = np.clip(_rgb.T.reshape([*img_comp.shape, 3]), 0.0, 1.0)
    
    rgb = np.moveaxis(rgb, -1, axis)
    return rgb

def imshow_cmap(ax: plt.Axes, img: ArrayLike, cmap=None, vmin=None, vmax=None):
    """
    Make both `matplotlib.colors.colormap` and `functions` be available for the argument `cmap` 
    """
    if isinstance(cmap, (type(None), str, Colormap)):
        return ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        assert hasattr(cmap, '__call__')
        if vmin is None:
            vmin = -1
        if vmax is None:
            vmax = 1
        img_cmapped = cmap(np.clip((img-vmin)/(vmax-vmin), 0, 1))
        return ax.imshow(img_cmapped)


'''
########################################################################
# >>                 Plotting Stokes component images               << #
########################################################################
'''
def _default_labels(n_components: int) -> Sequence[str]:
    """
    Helper function for `plot_components_forsign` and `plot_components_graycmap` below.
    """
    match n_components:
        case 2:
            labels = ["$s_1$ ", "$s_2$ "]
        case 3:
            labels = ["$s_0$ ", "$s_1$ ", "$s_2$ "]
        case 4:
            labels = ["$s_0$ ", "$s_1$ ", "$s_2$ ", "$s_3$ "]
        case _:
            labels = ["" for _ in range(n_components)]
    return labels

def _title_suffix_mode(vismode: Literal['forsign', 'graycmap', 'pos'], j: Optional[int] = 0) -> str:
    match vismode:
        case 'forsign':
            return " pos." if j==0 else " neg."
        case 'graycmap':
            # return " signed grayscale"
            return " signed gray"
        case 'pos':
            return ""
        case _:
            raise ValueError(f"Invalid parameter: {vismode=}.")
        
def _title_suffix_gamma(gamma: float):
    return f"1/gm={1/gamma:.1f}"
def _title_suffix_scale(scale: float):
    if scale == 1.0:
        return ""
    else:
        return f"x{scale:.1f}"

def plot_components_forsign(img: ArrayLike, # e.g. [h,w,3,2-4], [h,w,2-4]
                            /,
                            gamma: Optional[float] = 1/2.2,
                            scale: Optional[float] = 1.0,
                            *,
                            fill_nan: Optional[float] = None,
                            row_major: Optional[bool] = True,
                            row_labels: Optional[Sequence[str]] = None,
                            figsize: Optional[Tuple[float, float]] = None,
                            suptitle: Optional[str] = ""
                           ) -> Figure:
    assert img.ndim in [3, 4]
    fig = plt.figure(figsize=figsize)
    suptitle += f" [1/gm={1/gamma:.1f}, scale={scale:.1f}]"
    fig.suptitle(suptitle)
    if row_labels is None:
        row_labels = _default_labels(img.shape[-1])
    col_labels = ["pos.", "neg."]
    if row_major:
        n_rows = len(row_labels)
        n_cols = len(col_labels)
        get_idx = lambda i, j: i*n_cols+j+1
    else:
        n_rows = len(col_labels)
        n_cols = len(row_labels)
        get_idx = lambda i, j: i+j*n_cols+1

    if img.ndim == 3:
        nanmask = np.isnan(img).any(-1)
    else:
        nanmask = np.isnan(img).any((-2, -1))[..., None]
    
    for i, ilabel in enumerate(row_labels):
        for j, jlabel in enumerate(col_labels):
            img_curr = img[..., i] * scale
            if j == 1:
                img_curr = -img_curr
            if not fill_nan is None:
                img_curr = np.where(nanmask, fill_nan, img_curr)
            ax = fig.add_subplot(n_rows, n_cols, get_idx(i, j))
            ax.imshow(np.clip(img_curr, 0, 1)**gamma)
            ax.set_title(f"{ilabel}{jlabel}")

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    return fig

def plot_componets_graycmap(img: ArrayLike, # e.g. [h,w,3,2-4]
                            /,
                            gamma: Optional[float] = 1/2.2,
                            scale: Optional[float] = 1.0,
                            *,
                            fill_nan: Optional[float] = None,
                            row_major: Optional[bool] = True,
                            cmap: Union[None, str, Colormap] = None,
                            labels: Optional[Sequence[str]] = None,
                            figsize: Optional[Tuple[float, float]] = None,
                            suptitle: Optional[str] = ""
                           ) -> Figure:
    assert img.ndim in [3, 4]
    fig = plt.figure(figsize=figsize)
    suptitle += f" [1/gm={1/gamma:.1f}, scale={scale:.1f}]"
    fig.suptitle(suptitle)
    if labels is None:
        labels = _default_labels(img.shape[-1])
    if row_major:
        n_rows = len(labels)
        n_cols = 1
    else:
        n_rows = len(labels)
        n_cols = 1

    if img.ndim == 3:
        nanmask = np.isnan(img).any(-1)
    else:
        nanmask = np.isnan(img).any((-2, -1))[..., None]
    for i, ilabel in enumerate(labels):
        if img.ndim == 3:
            img_curr = img[..., i] * scale
        else:
            img_curr = rgb2gray(img[..., i]) * scale
        if not fill_nan is None:
            img_curr = np.where(nanmask, fill_nan, img_curr)
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        img_curr = powmag(img_curr, gamma)
        if cmap is None:
            ax.imshow(img_curr, cmap=cmap)
        else:
            ax.imshow(img_curr, cmap=cmap, vmin=-1, vmax=1)
        ax.set_title(f"{ilabel} signed grayscale")

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.tight_layout()

class CompImshower:
    def __init__(self,
                 n_comp: Union[int, CodType],
                 vismode: Sequence[Literal['forsign', 'graycmap', 'pos']],
                 gamma: Optional[float] = 1/2.2,
                 scale: Optional[float] = 1.0,
                 *,
                 fill_nan: Optional[float] = None,
                 row_major: Optional[bool] = True,
                 cmap: Union[None, str, Colormap] = None,
                 labels: Optional[Sequence[str]] = None,
                 figsize: Optional[Tuple[float, float]] = None,
                 suptitle: Optional[str] = "",
                 axes_off: Optional[bool] = True
                ):
        self.n_comp = int(n_comp)
        self.vismode = self.broadcast(vismode)
        self.n_mode = 1
        for m in self.vismode:
            if m == 'forsign': self.n_mode = 2
            assert m in ['forsign', 'graycmap', 'pos'], f"Invalid {m}."

        self.gamma = self.broadcast(gamma)
        self.scale = self.broadcast(scale)
        self.fill_nan = fill_nan
        self.row_major = row_major
        self.cmap = self.broadcast(cmap)
        self.labels = self.merge_param(_default_labels(self.n_comp), labels)
        self.figsize = figsize
        self.suptitle = suptitle
        self.axes_off = axes_off
    
    @classmethod
    def from_Stokes(cls,
                    n_comp: Union[None, int, CodType] = 4,
                    gamma: Optional[float] = 1/2.2,
                    scale: Optional[float] = 1.0,
                    *,
                    fill_nan: Optional[float] = None,
                    row_major: Optional[bool] = True,
                    cmap: Union[None, str, Colormap] = None,
                    labels: Optional[Sequence[str]] = None,
                    figsize: Optional[Tuple[float, float]] = None,
                    suptitle: Optional[str] = "",
                    axes_off: Optional[bool] = True
                   ) -> CompImshower:
        # ---------- Default arguments ----------
        if cmap is None: cmap = _vis_options['cmap']

        # ---------- Main ----------
        vismode = ['pos'] + ['graycmap'] * (n_comp-1)
        return cls(n_comp, vismode, gamma=gamma, scale=scale, fill_nan=fill_nan, row_major=row_major,
                   cmap=cmap, labels=labels, figsize=figsize, suptitle=suptitle, axes_off=axes_off)
    
    @classmethod
    def from_diff(cls,
                  n_comp: Union[None, int, CodType] = 4,
                  gamma: Optional[float] = 1/2.2,
                  scale: Optional[float] = 1.0,
                  *,
                  fill_nan: Optional[float] = None,
                  row_major: Optional[bool] = True,
                  cmap: Union[None, str, Colormap] = 'bwr',
                  labels: Optional[Sequence[str]] = None,
                  figsize: Optional[Tuple[float, float]] = None,
                  suptitle: Optional[str] = "",
                  axes_off: Optional[bool] = True
                 ) -> CompImshower:
        vismode = 'graycmap'
        return cls(n_comp, vismode, gamma=gamma, scale=scale, fill_nan=fill_nan, row_major=row_major,
                   cmap=cmap, labels=labels, figsize=figsize, suptitle=suptitle, axes_off=axes_off)

    def broadcast(self, val):
        if isinstance(val, (tuple, list)):
            ## `hasattr(val, '__len__')` is not desirable due to string arguments.
            if len(val) != self.n_comp:
                raise ValueError(f"The length of parameter `val` ({len(val)} is given) should be equal to {self.n_comp=}).")
            return val
        else:
            return [val]*self.n_comp
    
    def merge_param(self, m_param, a_param):
        a_param = self.broadcast(a_param)
        return [m if a is None else a
               for m, a in zip(m_param, a_param)]
    
    def imshow(self,
               img: ArrayLike, # [h, w, (3), n_comp]
               /,
               gamma: Optional[float] = None,
               scale: Optional[float] = None,
               *,
               fill_nan: Optional[float] = None,
               row_major: Optional[bool] = None,
               cmap: Union[None, str, Colormap] = None,
               labels: Optional[Sequence[str]] = None,
               figsize: Optional[Tuple[float, float]] = None,
               suptitle: Optional[str] = None
              ) -> Figure:
        # ---------- Initialize parameters ----------
        gamma = self.merge_param(self.gamma, gamma)
        scale = self.merge_param(self.scale, scale)
        fill_nan = self.fill_nan if fill_nan is None else fill_nan
        row_major = self.row_major if row_major is None else row_major
        cmap = self.merge_param(self.cmap, cmap)
        labels = self.merge_param(self.labels, labels)
        figsize = self.figsize if figsize is None else figsize
        suptitle = self.suptitle if suptitle is None else suptitle

        # ---------- Preprocess ----------
        assert img.ndim in [3, 4], f"Invalid shape: {img.shape}"
        assert img.shape[-1] == self.n_comp, f"Invalid shape: {img.shape}. " + \
                              f"The length along the last index should be equal to {self.n_comp=}."
        if row_major:
            n_rows = self.n_comp
            n_cols = self.n_mode
            get_idx = lambda i_comp, i_mode: n_cols*i_comp + i_mode + 1
        else:
            n_rows = self.n_mode
            n_cols = self.n_comp
            get_idx = lambda i_comp, i_mode: i_comp + n_cols*i_mode + 1

        if len(set(gamma)) == 1:
            unique_gamma = True
            sup_suf_gamma = [_title_suffix_gamma(gamma[0])]
        else:
            unique_gamma = False
            sup_suf_gamma = []
        if len(set(scale)) == 1:
            unique_scale = True
            sup_suf_scale = [] if scale[0] == 1.0 else [_title_suffix_scale(scale[0])]
        else:
            unique_scale = False
            sup_suf_scale = []
        
        suf_list = sup_suf_scale + sup_suf_gamma
        if suf_list:
            suptitle += " [" + ", ".join(suf_list) + "]"
        
        fig = plt.figure(figsize=figsize)
        fig.suptitle(suptitle)
        

        nanmask = np.isnan(img).any(tuple(range(img.ndim))[2:], keepdims=True).squeeze(-1)

        for i, (mode, gm, sc, cm, label) in enumerate(zip(self.vismode, gamma, scale, cmap, labels)):
            suf = [] if unique_gamma else [_title_suffix_gamma(gm)]
            suf += [] if unique_scale or sc == 1.0 else [_title_suffix_scale(sc)]
            suf_sc_gm = " [" + ", ".join(suf) + "]" if suf else ""

            # `mode in ['forsign', 'graycmap', 'pos']`
            j_range = range(2) if mode == 'forsign' else range(1)
            img_curr = img[..., i] * sc
            for j in j_range:
                if mode == 'graycmap':
                    if img.ndim == 4:
                        img_curr = rgb2gray(img_curr)
                else:
                    if mode == 'forsign' and j == 1:
                        img_curr = -img_curr
                    img_curr = np.clip(img_curr, 0, 1)
                if not fill_nan is None:
                    img_curr = np.where(nanmask, fill_nan, img_curr)
                img_curr = powmag(img_curr, gm)

                ax = fig.add_subplot(n_rows, n_cols, get_idx(i, j))
                if (mode != 'graycmap') or (cm is None):
                    ax.imshow(img_curr)
                else:
                    # ax.imshow(img_curr, cmap=cm, vmin=-1, vmax=1)
                    imshow_cmap(ax, img_curr, cmap=cm, vmin=-1, vmax=1)
                ax.set_title(f"{label}{suf_sc_gm}{_title_suffix_mode(mode, j)}")

                if self.axes_off:
                    ax.set_axis_off() # Turn off box and axis
                # ax.get_xaxis().set_visible(False) # Turn off axes only
                # ax.get_yaxis().set_visible(False)
        fig.tight_layout()
        return fig



'''
########################################################################
# >>                            visualize                           << #
########################################################################
'''

def set_visoptions(*, backend: Optional[Literal['matplotlib', 'vispy']] = None, cmap = None):
    """
    Set global visualization options for `polarsh.plot`
    """
    opt = dict()
    if not backend is None:
        assert backend in ['matplotlib', 'vispy']
        opt['backend'] = backend
    if not cmap is None:
        assert isinstance(cmap, (str, Colormap))
        opt['cmap'] = cmap
    _vis_options.update(opt)


vispy.use(gl="gl+")
__vis_options_vispy_default = {
    # Canvas
    'bgcolor': 'white',
    'figsize': (500, 500),
    # Title text
    'title_size': 15,
    'title_color': 'black',
    # shared property for 3D visualization
    'cam2world': vispy.util.quaternion.Quaternion(np.cos(-np.pi*1/9), np.sin(-np.pi*1/9), 0, 0) \
               * vispy.util.quaternion.Quaternion(np.cos(np.pi*3/8), 0, np.sin(np.pi*3/8), 0), # under VisPy convention
    # *Grid, *Field classes
    'arrow_size':  0.05,
    'arrow_scale': 0.1,
    'arrow_gamma': 1.0,
    'arrow_color': 'black',
    'arrow_dsamp': 1,
    'help': True,
}
_vis_options_vispy = __vis_options_vispy_default.copy()

def __vis_options_assert(vis_options: dict):
    for key, value in vis_options.items():
        if not key in __vis_options_vispy_default:
            raise ValueError(f"Invalid name of configuration is given: {key}\n"
                             f"A configuration name must be one of the followings:\n" + \
                             str(list(__vis_options_vispy_default))[1:-1])
        elif key == 'figsize':
            assert hasattr(value, '__len__')
            assert len(value) == 2

def set_visoptions_vispy(**vis_options):
    __vis_options_assert(vis_options)
    _vis_options_vispy.update(vis_options)

def set_visoptions_vispy_default():
    global _vis_options_vispy
    _vis_options_vispy = __vis_options_vispy_default.copy()

def args_None2options(**args_dict):
    res_dict = dict()
    for key, val in args_dict.items():
        if key in _vis_options:
            if val is None:
                res_dict[key] = _vis_options[key]
            else:
                res_dict[key] = args_dict[key]
        elif key in __vis_options_vispy_default:
            if val is None:
                res_dict[key] = _vis_options_vispy[key]
            else:
                res_dict[key] = args_dict[key]
        else:
            raise ValueError(f"Invalid name of configuration is given: {key}\n"
                             f"A configuration name must be one of the followings:\n" + \
                             str(list(__vis_options_vispy_default))[1:-1], "\n",
                             str(list(_vis_options))[1:-1])
    return res_dict
            



def gen_vispy_canvas() -> Tuple[vispy.scene.canvas.SceneCanvas, vispy.scene.widgets.viewbox.ViewBox]:
    layout = visualize_layout.workspace
    if layout is None:
        canvas = vispy.scene.SceneCanvas(keys='interactive',
                                        bgcolor=_vis_options_vispy['bgcolor'],
                                        size=_vis_options_vispy['figsize'],
                                        show=True, resizable=True)
        view = canvas.central_widget.add_view()
    else:
        layout = visualize_layout.workspace
        canvas = layout.canvas
        idx_grid = next(layout.idx_iter)
        view = layout.grid.add_view(row=idx_grid[0], col=idx_grid[1], border_color=layout.border_color)
    
    view.camera = 'arcball'
    view.camera.set_range(x=[-1., 1.])
    view.camera._quaternion = _vis_options_vispy['cam2world']
    view.camera.depth_value = 1.0
    view.camera.scale_factor = 2.75

    if not layout is None:
        if layout.camera is None:
            layout.camera = view.camera
        else:
            layout.camera.link(view.camera)
    return canvas, view

def gen_vispy_axes(parent):
    axes = vispy.scene.visuals.XYZAxis(parent=parent)
    axes.transform = vispy.scene.STTransform(scale=(2, 2, 2))

    def func_toggle_visible():
        axes.visible = not axes.visible
    
    return axes, func_toggle_visible

def vispy_attach_title(canvas_view: Union[vispy.scene.SceneCanvas, vispy.scene.ViewBox],
                       text:        str,
                       size:        Optional[float] = None,
                       color:       Optional[str]   = None,
                       align:       Optional[Literal['top', 'center']] = 'top'):
    # ---------- Arguments ----------
    if size is None:  size  = _vis_options_vispy['title_size']
    if color is None: color = _vis_options_vispy['title_color']
    if not align in ['top', 'bottom']:
        raise ValueError(f"The parameter `align` must be either 'top' or 'center', but {align=} is given.")
    
    # ---------- Main ----------
    if isinstance(canvas_view, vispy.scene.SceneCanvas):
        parent = canvas_view.central_widget
    elif isinstance(canvas_view, vispy.scene.ViewBox):
        parent = canvas_view
    else:
        raise TypeError(f"{canvas_view.type = }")
    
    
    textObj = vispy.scene.Text(text, bold=True, font_size=size,
                               color=color, parent=parent)
    
    @canvas_view.events.resize.connect
    def func_resize(event):
        if align == 'top':
            pos_y = size
        else:
            pos_y = canvas_view.size[1]/2
        textObj.pos = (canvas_view.size[0]/2, pos_y)

class VisObj:
    def __init__(self, vertices, faces, texcoords):
        self.vertices = vertices
        self.faces = faces
        self.texcoords = texcoords

    @classmethod
    def from_file(cls, filename_obj: str):
        vertices, faces, _, texcoords = vispy.io.read_mesh(filename_obj)
        texcoords[:,1] = 1 - texcoords[:,1]
        return cls(vertices, faces, texcoords)
    
    def apply_rotation(self, rotation: ArrayLike) -> VisObj:
        ## [WARN NOTE] If det(rotation) != 1, wrong weight will be computed.
        rotation = rotation2quat(rotation).to_rotation_matrix.astype(np.float32)
        return VisObj(matmul_vec1d(rotation, self.vertices), self.faces.copy(), self.texcoords.copy())
        


class visualize_layout:
    """
    See also the function `gen_vispy_canvas`
    """
    workspace = None

    def __init__(self, rows: int, cols: int,
                 figsize: Optional[Tuple[int, int]] = None,
                 border_padding: Optional[float] = 0,#6,
                 border_color:   Optional[str] = None, #'gray'
                 show: Optional[bool] = True
                ):
        # ---------- Preprocess ----------
        if figsize is None: figsize = _vis_options_vispy['figsize']

        if not visualize_layout.workspace is None:
            raise RuntimeError(f"This constructor must be run in the `with visualize_layout(...) ...:` statement, "
                               "and nested statements are not allowed.")
        
        # ---------- Main ----------
        self.canvas = vispy.scene.SceneCanvas(keys='interactive',
                                              bgcolor=_vis_options_vispy['bgcolor'],
                                              size=figsize,
                                              show=True, resizable=True)
        self.grid = self.canvas.central_widget.add_grid()
        self.grid.padding = 6
        self.border_color = border_color
        self.idx_iter = np.ndindex(rows, cols)
        self.camera = None # initialized in `gen_vispy_canvas`
        self.show = bool(show)

    def __enter__(self):
        visualize_layout.workspace = self
        return self.canvas
    
    def __exit__(self, type, value, traceback):
        visualize_layout.workspace = None
        if self.show:
            if IPython.get_ipython() is None:
                self.canvas.show()
            else:
                IPython.display.display(self.canvas)
    
    @classmethod
    def skip(cls, n: Optional[int] = 1):
        if visualize_layout.workspace is None:
            raise RuntimeError(f"This constructor must be run inside the `with visualize_layout(...) ...:` statement.")
        n = int(n)
        assert n >= 0, f"Invalid argument: {n = }"
        for _ in range(n):
            next(cls.workspace.idx_iter)


    
# https://stackoverflow.com/questions/6028000/how-to-read-a-static-file-from-inside-a-python-package
__resource_dir = importlib.resources.files(resource)
visCubeSphere = VisObj.from_file(__resource_dir/"cube-sphere-apply4.obj")
visUVSphere = VisObj.from_file(__resource_dir/"uv-sphere-32.obj")
visArrow = VisObj.from_file(__resource_dir/"arrow-8-radi12.obj")
visBiArrow = VisObj.from_file(__resource_dir/"biarrow-8-radi25.obj")


'''
########################################################################
# >>                     General ploting methods                    << #
########################################################################
'''
def matmatshow(data:     ArrayLike,
               axis_row: Optional[int] = -2,
               axis_col: Optional[int] = -1,
               figsize:  Optional[Tuple[float, float]] = None,
               title:    Optional[str] = None,
               cmap:     Optional[Colormap] = pltcmap_diverge,
               norm:     Optional[Normalize] = Normalize(),
               colorbar: Optional[bool] = True,
               ticks:    Optional[bool] = False,
               frame:    Optional[bool] = True,
               fig:      Optional[Figure] = None
              ) ->       Figure:
    '''
    Show 4D array using nrows x ncols subplots of `plt.matshow`.
    '''
    # ========== Preprocess according to `data.ndim` ==========
    match data.ndim:
        case 2:
            nrows, ncols = 1, 1
            data = data[None]
        case 4:
            data = np.moveaxis(data.view(), (axis_row, axis_col), (0, 1))
            nrows, ncols = data.shape[:2]
            data = data.reshape(nrows*ncols, *data.shape[2:])
        case _:
            raise ValueError(f"`data.ndim` must be 2 or 4, but {data.ndim} is given.\n"
                             f"Note: {data.shape = }")
    
    # ========== Initialize `Figure` and `ImageGrid` instances ==========
    if fig is None:
        fig = plt.figure(figsize=figsize)
    elif figsize is not None:
        raise ValueError(f"One of arguments `{figsize=}` and `{fig=}` must be `None`.")
    
    if colorbar:
        # [NOTE] 7% may be too small for `SHVec.mathshow(long=True)`
        cbar_size = "30%" if data.shape[-1] == 1 else "7%"
        cbar_cfgs = dict(cbar_location="right",
                        cbar_mode="single",
                        cbar_size=cbar_size,
                        cbar_pad=0.15)
    else:
        cbar_cfgs = dict()
    
    grid = ImageGrid(fig, 111,          # as in `plt.subplot(111)`
                     nrows_ncols=(nrows, ncols),
                     axes_pad=0.10,
                     share_all=True,
                     **cbar_cfgs)

    # ========== Plot ==========
    for ax, mat in zip(grid, data):
        im = ax.matshow(mat, cmap=cmap, norm=norm)
        if not ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_frame_on(frame)
    if colorbar:
        ax.cax.colorbar(im)
    if not title is None:
        fig.suptitle(title)
    return fig



def compmatshow(data: ArrayLike, sign: str='ria',
                figsize=None, colorbar=True, title=None) -> plt.figure:
    '''
    `plt.matshow` for complex values
    '''
    assert len(sign) > 0
    for ch in sign:
        assert ch in 'ria'
    
    ncols = len(sign)
    fig = plt.figure(figsize=figsize)
    for i,ch in enumerate(sign):
        ax = fig.add_subplot(1, ncols, i+1)
        if ch == 'r':
            data_show = data.real
        elif ch == 'i':
            data_show = data.imag
        else:
            data_show = np.abs(data)
        im = ax.matshow(data_show)
        if colorbar:
            plt.colorbar(im)
    fig.suptitle(title)
    return fig

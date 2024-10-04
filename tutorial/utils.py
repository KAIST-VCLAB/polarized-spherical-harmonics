from typing import Union, Literal, Optional
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import polarsh as psh

def plot_text_only(fig_ax: Union[mpl.figure.Figure, mpl.axes.Axes],
                   text:   str,
                   size:   Optional[int] = 40
                  ) ->     mpl.axes.Axes:
    """Show only a text in given figure (or axes)"""
    if isinstance(fig_ax, (mpl.figure.Figure, mpl.figure.SubFigure)):
        ax = fig_ax.subplots()
    elif isinstance(fig_ax, mpl.axes.Axes):
        ax = fig_ax
    else:
        raise TypeError(f"Invalid type of the argumnet: {type(ax) = }")
    ax.text(0.0, 0.5, text, size=40)
    ax.set_axis_off()
    return ax

def imshow(ax:    mpl.axes.Axes,
           img:   np.ndarray, # [h, w] ('graycmap' only) or [h, w, 3] ('gamma' or 'graycmap')
           mode:  Union[int, Literal['gamma', 'graycmap']],
           norm:  Optional[mcolors.Normalize] = mcolors.CenteredNorm(),
           title: Optional[str] = None
          ) ->    mpl.image.AxesImage:
    # ---------- Parameters ----------
    mode_list = ['gamma', 'graycmap']
    if isinstance(mode, bool):
        mode = int(mode)
    if isinstance(mode, int):
        # For $s_i$ Stokes component image, `mode = i` shows gamma-corrected color image for $s_0$,
        # and colormaped grayscale image for $s_1$, $s_2$, and $s_3$.
        mode = mode_list[min(mode, 1)]
    assert mode in mode_list
    img = np.asarray(img)
    
    # ---------- `ax.imshow()` ----------
    if img.ndim == 2:
        im = ax.imshow(img, cmap=psh.pltcmap_diverge, norm=norm)
    elif img.ndim == 3:
        if mode == 'gamma':
            im = ax.imshow(np.clip(img, 0, 1) ** (1/2.2))
        else:
            im = ax.imshow(psh.rgb2gray(img), cmap=psh.pltcmap_diverge, norm=norm)
    else:
        raise ValueError(f"Invalid shape of the argument: {img.shape = }, {img.dtype = }")

    # ---------- Remained ----------
    if title is not None:
        ax.set_title(title)
    ax.set_axis_off()
    return im

def math_stokes(symbol: str) -> str:
    return r"{\overset{\leftrightarrow}{%s}}" % symbol
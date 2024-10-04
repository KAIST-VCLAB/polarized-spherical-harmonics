from typing import Sequence, Optional, Union, Literal, Callable
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import sys, platform, argparse, shutil

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as scipyRotation

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import polarsh as psh
from polarsh.util import flip_arg
from utils import imshow

class Log:
    def __init__(self, txt_file: Union[str, Path], mode: Optional[Literal['a', 'w']] = 'a'):
        self.file = open(txt_file, mode)
        now_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.file.write("# " + "="*30 + "\n"
                        f"# {now_string} from {platform.node()}\n"
                        "# " + "="*30 + "\n")

    def __del__(self):
        self.file.close()

    def print(self, *args, sep: Optional[str] = " ", end: Optional[str] = "\n"):
        print(*args, sep=sep, end=end)
        self.file.write(sep.join(args) + end)

def imshow_comp(ax:      mpl.axes.Axes,
                img_stk: np.ndarray,
                s_idx:   int,
                vabs:    Union[None, float, Sequence[float]] = None,
                title:   Union[str] = None) -> mpl.image.AxesImage:
    assert img_stk.shape[2:] in [(3, 3), (3, 4)], f"{img_stk.shape = }"
    p = img_stk.shape[-1]
    assert s_idx in range(p), f"{s_idx = }"
    if hasattr(vabs, '__len__'):
        vabs = vabs[s_idx - 1]
    norm = mcolors.CenteredNorm() if vabs is None else mcolors.Normalize(-vabs, vabs)
    return imshow(ax, img_stk[..., s_idx], s_idx, norm, title)

def cubemask2image(mask: np.ndarray, # [6, edge, edge]
                   border_width: int,
                   border_color: Sequence[float]
                  ) -> np.ndarray: # [3*edge + 2*borderwidth, 4*edge + 2*borderwidth, 4(RGBA)]
    edge = mask.shape[1]
    assert mask.shape == (6, edge, edge)
    bd = border_width
    color = np.concatenate([np.asarray(border_color), (1,)], 0)

    image = np.full((3*edge+2*bd, 4*edge+2*bd, 4), (1, 1, 1, 0), dtype=float)
    image[edge:-edge, :] = color
    image[:, edge:2*edge+2*bd] = color

    mask_rgba = np.concatenate([mask[..., None] + np.zeros((3,)), np.ones(mask.shape + (1,))], -1)
    mask_rgba = psh.envmap_unfold(mask_rgba)
    inner = image[bd:-bd, bd:-bd, :]
    wh = mask_rgba[..., -1] == 1
    inner[wh] = mask_rgba[wh]
    return image

def rotvec2str(rotvec):
    """ e.g. [10.00,20.00,0.20] """
    rotvec = np.asarray(rotvec)
    formatter = {'float_kind':lambda x:"%.2f"%x}
    return np.array2string(rotvec, formatter=formatter, separator=",")

def StokesField_nvSHCoeff(stkF:  psh.StokesField,
                          level: int,
                          type:  Optional[Literal[1, 2]] = 1
                         ) ->    psh.SHVec:
    """
    Note that type 1 and 2 yield the equivalent frequency domain representations,
    i.e., each subset of the basis for each order l>=0
    {Y_{l,-l}^{naive,type 1 or 2}, ...., Y_{l,l}^{naive,type 1 or 2}}
    spans the identical subspace.
    """
    sphG = stkF.SphGrid
    stkF = stkF.to_SphFF(sphG.ThetaPhiFrameField())
    if type == 1:
        stk_comp = np.stack([stkF.s0(), stkF.s12_comp(), stkF.s3()], -1)
        return psh.ScalarField(sphG, stk_comp).SHCoeff(level, "COMP")
    else:
        assert type == 2
        return psh.ScalarField(sphG, stkF.Stk).SHCoeff(level, "REAL")

def StokesField_from_nvSHCoeff(shv:   psh.SHVec,
                               sphFF: Union[psh.SphereGrid, psh.SphereFrameField],
                               type:  Optional[Literal[1, 2]] = 1
                              ) ->    psh.StokesField:
    if isinstance(sphFF, psh.SphereGrid):
        sphG = sphFF
        sphFF = sphG.ThetaPhiFrameField()
    elif isinstance(sphFF, psh.SphereFrameField):
        sphG = sphFF.SphGrid
    else:
        raise TypeError(f"Invalid type: {type(sphFF) = }")
    if type == 1:
        assert shv.sh_type == psh.SHType.COMP
        scalF = psh.ScalarField.from_SHCoeff(shv, sphFF)
        assert np.allclose(scalF.fval[..., (0, 2)].imag, 0)
        Stk = np.stack([scalF.fval[..., 0].real,
                        scalF.fval[..., 1].real, scalF.fval[..., 1].imag,
                        scalF.fval[..., 2].real], -1)
        return psh.StokesField(sphG.ThetaPhiFrameField(), Stk).to_SphFF(sphFF)
    else:
        assert type == 2
        assert shv.sh_type == psh.SHType.REAL
        scalF = psh.ScalarField.from_SHCoeff(shv, sphFF)
        return psh.StokesField(sphG.ThetaPhiFrameField(), scalF.fval).to_SphFF(sphFF)

def nvSHVec_wrt_persp(shv: psh.SHVec,
                      h: int, w: int,
                      fov_x_deg: float,
                      to_world: Optional[ArrayLike] = np.eye(3),
                      type:  Optional[Literal[1, 2]] = 1
                     ) -> psh.StokesField:
    fval = shv.wrt_persp(h, w, fov_x_deg, to_world)
    perspFF = psh.SphereFrameField.from_persp(h, w, fov_x_deg, to_world)
    if type == 1:
        assert shv.sh_type == psh.SHType.COMP
        assert np.allclose(fval[..., (0, 2)].imag, 0)
        Stk = np.stack([fval[..., 0].real,
                        fval[..., 1].real, fval[..., 1].imag,
                        fval[..., 2].real], -1)
    else:
        assert type == 2
        assert shv.sh_type == psh.SHType.REAL
        Stk = fval
    return psh.StokesField(perspFF.SphGrid.ThetaPhiFrameField(), Stk).wrt_SphFF(perspFF)
        

def StokesField_nvSH_upto(level:   int,
                          sphG:    psh.SphereGrid,
                          type:    Optional[Literal[1, 2]] = 1,
                          rotated: Optional[ArrayLike] = None
                         ) ->      psh.StokesField:
    if rotated is None:
        flag_rotation = False
    else:
        rotated = np.asarray(rotated)
        assert rotated.shape == (3,), f"{rotated.shape = }, {rotated}"
        flag_rotation = not np.array_equal(rotated, [0, 0, 0])
    
    if flag_rotation:
        sphG_invR = sphG.apply_rotation(-rotated)
    else:
        sphG_invR = sphG

    def not_rotated(level: int, sphG: psh.SphereGrid, type: Optional[Literal[1, 2]] = 1):
        if type == 1:
            scalF_SH = psh.ScalarField.from_SH_upto(level, sphG, psh.SHType.COMP)
            return psh.StokesField(sphG.ThetaPhiFrameField(), psh.comp2vec(scalF_SH.fval))
        elif type == 2:
            scalF_SH = psh.ScalarField.from_SH_upto(level, sphG, psh.SHType.REAL)
            Z = np.zeros_like(scalF_SH.fval)
            return psh.StokesField(sphG.ThetaPhiFrameField(), np.stack([scalF_SH.fval, Z], -1))
    stkF_invR = not_rotated(level, sphG_invR, type)

    if flag_rotation:
        tpF_RTR = scipyRotation.from_rotvec(rotated).as_matrix() @ stkF_invR.SphFF.F
        return psh.StokesField(psh.SphereFrameField(sphG, tpF_RTR), stkF_invR.Stk)
    else:
        return stkF_invR

def nvSHVec_apply_rotmat(shv: psh.SHVec,
                         rotvec:      psh.RotationLike,
                         rotmat_nvSH: np.ndarray, # [N, N] complex
                         type: Optional[Literal[1, 2]] = 1
                        ) -> psh.SHVec:
    assert shv.shape_chan in [(3, 3), (3, 4)]
    assert shv.cod_type == psh.CodType.SCALAR
    shv_s03 = shv.chan[:, [0, -1]]
    assert shv_s03.shape_chan == (3, 2)
    shv_s03_rot = rotvec @ shv_s03

    if type == 1:
        coeff_s12 = shv.coeff[:, 1]
    else:
        assert type == 2
        coeff_s12 = psh.vec2comp(shv.coeff[:, 1:3])
    coeff_s12_rot = psh.matmul_vec1d(rotmat_nvSH, coeff_s12)
    if type == 1:
        coeff_s12_rot = coeff_s12_rot[:, None]
    else:
        coeff_s12_rot = psh.comp2vec(coeff_s12_rot, -2)
    coeff_rot = np.concatenate([shv_s03_rot.coeff[:, 0:1],
                                coeff_s12_rot,
                                shv_s03_rot.coeff[:, -1:]], -2)
    return psh.SHVec(coeff_rot, shv.cod_type, shv.sh_type)

def fields2mat(stkF1: psh.StokesField, stkF2: psh.StokesField) -> np.ndarray[complex]:
    assert {stkF1.cod_type, stkF2.cod_type} == {psh.CodType.POLAR2}
    assert stkF1.SphGrid is stkF2.SphGrid
    res = np.zeros(stkF1.shape_chan + stkF2.shape_chan, dtype=complex)

    for itup in np.ndindex(stkF1.shape_chan):
        for jtup in np.ndindex(stkF2.shape_chan):
            res[*itup, *jtup] = stkF1.chan[itup].inner_comp(stkF2.chan[jtup])
    return res


DATATYPE_IO = {
    # 'data_type':    [read(file)->data, write(file, data)]
    'numpy':          [np.load, np.save],
    'Stokes image':   [psh.imread_Stk, psh.imwrite_Stk],
    'StokesField er': [psh.StokesField.from_equirectimage,
                       lambda file, data: psh.imwrite_Stk(file, data.Stk)],
    'SHVec':          [psh.SHVec.from_npz_file, flip_arg(psh.SHVec.save)],
    'SHMat':          [psh.SHMat.from_npz_file, flip_arg(psh.SHMat.save)],
    'SHConv':         [psh.SHConv.from_file, flip_arg(psh.SHConv.save)]
}
task_list = [] # decorated methods in `ReproduceSY24` will be appended
class ReproduceSY24:
    """
    A class for scripts to reproduce [Yi et al. 2024]
    "Spin-Weighted Spherical Harmonics for Polarized Light Transport" by
    Shinyoung Yi, Donggun Kim, Jiwoong Na, Xin Tong, and Min H. Kim
    (ACM Trans. Graph. Vol 43, Issue 4, No 127, 2024)
    """
    task_list = [] # decorated methods will be appended

    def __init__(self,
                 mode:         Literal['full', 'simple'],
                 remove_cache: bool,
                 indir:        Optional[Union[str, Path]] = "input",
                 outdir_base:  Optional[Union[str, Path]] = "reproduce",
                 log_file:     Optional[Union[str, Path]] = "log.txt"):
        self.outdir_base = Path(outdir_base)
        mode = str(mode).lower()
        self.indir = Path(indir)
        assert self.indir.exists(), f"{indir = } does not exist."
        if mode == 'full':
            self.outdir_data = self.outdir_base / "output"
            self.outdir_fig = self.outdir_base / "figure"
        else:
            assert mode == 'simple'
            self.outdir_data = self.outdir_base / "output_simple"
            self.outdir_fig = self.outdir_base / "figure_simple"
        
        
        self.outdir_fig.mkdir(exist_ok=True)
        log_path = self.outdir_base / log_file
        self.log = Log(log_path)
        
        self.done_list = [] # See the `self.task()` decorator
        self.log.print(f"# [ReproduceSY24] {mode=}\n"
                       f"# Data directory: {self.outdir_data}\n"
                       f"# Figure directory: {self.outdir_fig}\n"
                       f"# Log: {log_path}\n")
        
        if remove_cache:
            self.log.print("# Removing cached data...")
            if self.outdir_data.exists():
                n = len(list(self.outdir_data.iterdir()))
                shutil.rmtree(self.outdir_data)
                self.log.print(f"# {n} cached data files have been removed!\n")
            else:
                self.log.print(f"# No cached data file to remove exists!\n")
        self.outdir_data.mkdir(exist_ok=True)

        # ========== Parameters ==========
        self.nvSH_type = 1
        self.cube_edge = 1024        if mode == 'full' else 64
        self.level = 15              if mode == 'full' else 5
        self.hw_er = (512, 1024)     if mode == 'full' else (64, 128)

        self.rotmat_simul_edge = 512 if mode == 'full' else 64
        self.level_rotmat = 15       if mode == 'full' else 5
        self.level_rotmat_show = 6   if mode == 'full' else 5
        self.rotvec_list = [[np.pi/2, 0, 0], [1, 2, 3]]

        self.level_recon = self.level_rotmat

        self.file_obj = self.indir / "buddha.obj"
        self.i_vert = 17650
        self.edge_vis = 256          if mode == 'full' else 64
        self.level_TPmat = 5

        self.level_conv = 100        if mode == 'full' else 50

        self.level_roteq = 7         if mode == 'full' else 5
        self.cod_type_roteq = psh.CodType.POLAR3
        self.file_brdf = self.indir / "SHMat_6_gold_ds1_L20_ISOBI2.npz"
        self.grid_level_roteq = 6    if mode == 'full' else 4
        # ========== Parameters end ==========

        with psh.Tictoc("* Read a polarized environment map: %.4f sec", file=[sys.stdout, self.log.file]):
            self.stkF_cube = psh.StokesField.from_cubeimage(psh.data_dir/f"sponza_{self.cube_edge}_s%d.exr")
        
        with psh.Tictoc("* Read a SHVec with high level for Figure 16: %.4f sec", file=[sys.stdout, self.log.file]):
            """
            Note that `self.shv_high` can also be reproduced in a similar way to `self.shv`.
            However we just use the precomputed file
            since it is just input data of the experiment described in Figure 16.
            """
            self.shv_high = psh.SHVec.from_npz_file(self.indir/"SHVec_sponza_1024_L100.npz")
        
    def tictoc(self, name: str):
        return psh.Tictoc(f"* [Done] {name}: %.4f sec", f"* Compute {name}",
                          [sys.stdout, self.log.file])

    def savefig(self, fig: mpl.figure.Figure, filename: str):
        fig_filename = self.outdir_fig / filename
        fig.savefig(fig_filename)
        self.log.print(f"* [Done] Save a figure {fig_filename}\n")
    
    def task(func: Callable):
        """
        Decorator to mark methods which will be executed by `self.run_all()`.
        """
        def wrapper(self, *args, **kargs):
            res = func(self, *args, **kargs)
            self.done_list.append(func.__name__)
            return res
        global task_list
        task_list.append(wrapper)
        return wrapper
    
    def cached(self,
               filename:  str,
               data_type: Literal['numpy', 'Stokes image', 'StokesField er', 'SHVec', 'SHMat', 'SHConv'],
               name:      Optional[str] = None):
        """
        Decorator for cached evaluation
        Parameters:
            filename: str, file name for cached data without directory
        """
        assert data_type in DATATYPE_IO, f"{data_type = }"
        return psh.util.cached(self.outdir_data/filename, *DATATYPE_IO[data_type], tictoc=self.tictoc(name))

    @task
    def figure_5(self):
        h, w = self.hw_er # image size for equirectangular form
        self.log.print("="*20, "[ReproduceSY24.figure_5]", "="*20)
        
        # Equirectangular polarized environment map & Save it as `.exr`
        file_stk_er = self.outdir_data/f"sponza_er{h},{w}_s%d.exr"
        @self.cached(f"sponza_er{h},{w}_s%d.exr", 'Stokes image', "equirectangular Stokes images")
        def get_stkimage_er():
            return self.stkF_cube.wrt_equirect(h, w)
        self.stkComp_er = get_stkimage_er()

        # ========== Make Figure 5 ==========
        fig = plt.figure()
        subfigs = fig.subfigures(1, 3, width_ratios=[2, 1, 1])
        for i, subfig in enumerate(subfigs):
            img_stk = (
                self.stkComp_er[::-1], self.stkF_cube.Stk[5], self.stkF_cube.Stk[0]
            )[i]
            vabs = (0.2, 0.025, 0.2)[i]
            
            axes = subfig.subplots(3, 1)
            for j, ax in enumerate(axes):
                im = imshow_comp(ax, img_stk, j, vabs)
            if i < 2:
                subfig.suptitle(["\n(d) Stokes component map\nw.r.t. $\\theta\\phi$-frame field",
                                "\n(e) Stokes component map\nw.r.t. perspective frame field"][i])
            fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', shrink=0.6)
        fig.suptitle("Figure 5. Visualization of a Stokes vector field (polarized environment map)")
        self.savefig(fig, "Figure 5.png")
        # self.done_figure_5 = True
    
    @task
    def figure_7(self):
        fov_deg = 20.0
        to_world_list = [np.eye(3), psh.rotx(np.pi)]
        vabs = 0.2
        self.log.print("="*20, "[ReproduceSY24.figure_7]", "="*20)

        # ========== Polarized SH ==========
        @self.cached(f"SHVec_sponza_{self.cube_edge}_L{self.level}.npz", 'SHVec', "PSH coefficient vector")
        def get_PSH_coeff_vec():
            return self.stkF_cube.SHCoeff(self.level, psh.SHType.REAL)
        self.shv = get_PSH_coeff_vec()
        
        # ========== Naive approach: scalar SH + θφ-frame field ==========
        @self.cached(f"nv{self.nvSH_type}SHVec_sponza_{self.cube_edge}_L{self.level}.npz", 'SHVec', "nvSH coefficient vector")
        def get_nvSH_coeff_vec():
            return StokesField_nvSHCoeff(self.stkF_cube, self.level, self.nvSH_type)
        self.shv_nv = get_nvSH_coeff_vec()
        
        # ========== Make Figure 7 (a) ==========
        edge = self.stkF_cube.shape_grid[1] # 1024
        assert edge % 2 == 0
        assert self.stkF_cube.shape_grid == (6, edge, edge)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        edge_zoom = edge * np.tan(np.deg2rad(fov_deg)/2) / np.tan(np.pi/4)
        edge_zoom = int(edge_zoom/2)*2
        off = (edge-edge_zoom)//2
        title_list = ["FOV 90˚, $s_0$"] + [f"FOV {int(fov_deg)}˚, $s_{i}$" for i in range(3)]
        for (i,j), ax in np.ndenumerate(axes):
            face_idx = (5, 0)[i]
            img_stk = self.stkF_cube.Stk[face_idx]
            
            if j > 0: # FOV 20 degrees
                img_stk = img_stk[off:-off, off:-off]
            i_or_ii = "(" + ("i"*(i+1)) + ") " if j == 1 else ""
            im = imshow_comp(ax, img_stk, max(0, j-1), vabs, i_or_ii + title_list[j])

        fig.colorbar(im, ax = axes.ravel().tolist())
        fig.suptitle("Figure 7(a). Original polarized environment map")
        self.savefig(fig, "Figure 7(a).png")

        # ========== Make Figure 7(b, c) ==========
        img_stk_list = [[], []]
        for i_pos, to_world in enumerate(to_world_list):
            deco = self.cached(f"sponza_1024_nv{self.nvSH_type}SH_L{self.level}_persp_cam{i_pos},{edge_zoom}_s%d.exr", 'Stokes image', "Persp image from coefficients")
            img_stk_nvsh = deco(nvSHVec_wrt_persp)(self.shv_nv, edge_zoom, edge_zoom, fov_deg, to_world, self.nvSH_type)

            deco = self.cached(f"sponza_1024_SH_L{self.level}_persp_cam{i_pos},{edge_zoom}_s%d.exr", 'Stokes image', "Persp image from coefficients")
            img_stk_psh = deco(self.shv.wrt_persp)(edge_zoom, edge_zoom, fov_deg, to_world)

            img_stk_list[0].append(img_stk_nvsh)
            img_stk_list[1].append(img_stk_psh)

        fig = plt.figure(figsize=(16, 8))
        suf = f" ($l<{self.level}$)"
        suptitle_list = ["\n(b) Naive approach: Scalar SH + $\\vec{\\mathbf{F}}_{\\theta\\phi}$"+suf, "\n(c) Spin-2 SH"+suf]
        iterator = zip(fig.subfigures(1, 2), [self.shv_nv, self.shv], suptitle_list)
        for i, (subfig, shv_curr, suptitle) in enumerate(iterator):
            for (j, k), ax in np.ndenumerate(subfig.subplots(2, 2)):
                title = ""
                if k == 0:
                    title += "(" + ("i"*(j+1)) + ") "
                imshow_comp(ax, img_stk_list[i][j], k+1, vabs, f"{title}$s_{k+1}$")
            subfig.suptitle(suptitle)
        fig.suptitle("Figure 7(b, c). Freuqncy limited image of (a-i, ii) under each basis")
        self.savefig(fig, "Figure 7(b, c).png")

        # self.done_figure_7 = True

    def get_rotmat(self):
        self.log.print("="*20, "[ReproduceSY24.get_rotmat]", "="*20)
        level = self.level_rotmat
        edge = self.rotmat_simul_edge
        sphG = psh.SphereGridCube(edge)
        N = psh.level2num(level, psh.DomType.UNI)
        shape = (N, N)
        
        # ========== spin-2 SH rotation matrix ==========
        if not hasattr(self, 'rotmat_PSH_list'):
            self.rotmat_PSH_list = []
            # It will be more correct if time consumption to construct `stkF_s2SH` was taken into account.
            # However we are not aim to exactly measure time cost for grid-based simulation of basis rotation matrices.
            stkF_s2SH = psh.StokesField.from_s2SH_upto(level, sphG)
            for rotvec in self.rotvec_list:
                @self.cached(f"RotMat_L{level}_R{rotvec2str(rotvec)}_simul_cube{edge}.npy", 'numpy', f"RotMat L{level} on Cube{edge}")
                def get_rotmat_PSH():
                    stkF_s2SH_rot = psh.StokesField.from_s2SH_upto(level, sphG, rotvec)
                    return fields2mat(stkF_s2SH, stkF_s2SH_rot)
                rotmat_PSH = get_rotmat_PSH()
                assert rotmat_PSH.shape == shape
                self.rotmat_PSH_list.append(rotmat_PSH)
        
        # ========== Naive approach: SH + FF rotation matrix ==========
        if not hasattr(self, 'rotmat_nvSH_list'):
            self.rotmat_nvSH_list = []
            # It will be more correct if time consumption to construct `stkF_nvSH` was taken into account.
            # However we are not aim to exactly measure time cost for grid-based simulation of basis rotation matrices.
            if False:
                scalF_SH = psh.ScalarField.from_SH_upto(level, sphG, psh.SHType.REAL)
            stkF_nvSH = StokesField_nvSH_upto(level, sphG, self.nvSH_type)
            for rotvec in self.rotvec_list:
                @self.cached(f"nvRotMat_L{level}_R{rotvec2str(rotvec)}_simul_cube{edge}.npy", 'numpy', f"nvRotMat L{level} on Cube{edge}")
                def get_rotmat_nvSH():
                    stkF_nvSH_rot = StokesField_nvSH_upto(level, sphG, self.nvSH_type, rotvec)
                    return fields2mat(stkF_nvSH, stkF_nvSH_rot)
                rotmat_nvSH = get_rotmat_nvSH()
                assert rotmat_nvSH.shape == shape
                self.rotmat_nvSH_list.append(rotmat_nvSH)
    
    @task
    def figure_10(self):
        self.get_rotmat()
        i_rot = 1
        rotvec = self.rotvec_list[i_rot]
        rotmat_PSH = self.rotmat_PSH_list[i_rot]
        rotmat_nvSH = self.rotmat_nvSH_list[i_rot]
        N_cut = psh.level2num(self.level_rotmat_show, psh.DomType.UNI)
        self.log.print("="*20, "[ReproduceSY24.figure_10]", "="*20)

        # ========== Check ==========
        rotmat_SH = psh.SHMat.from_rotation(rotvec, self.level_rotmat, psh.CodType.SCALAR, psh.SHType.COMP).coeff
        rotmat_PSH_GT = psh.SHMat.from_rotation(rotvec, self.level_rotmat, psh.CodType.POLAR2, psh.SHType.REAL)
        GT_iso, GT_conj = psh.mat2comppair(rotmat_PSH_GT.coeff)
        print(f"* [Report] Grid-simulation error for spin-2 SH rotation:")
        print(f"           {psh.rms(rotmat_PSH - GT_iso) = }\t{psh.rms(GT_conj) = }")
        rotmat_list = [rotmat_SH, rotmat_nvSH, rotmat_PSH]

        # ========== Plot ==========
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        title_list = ["(a) Sclar SH\nScalar radiance",
                      "(b) Naive approach: Scalar SH + $\\vec{\\mathbf{F}}_{\\theta\\phi}$\nStokes vectors",
                      "(c) Spin-2 SH\nStokes vectors (ours)"]
        def matshow(ax: mpl.axes.Axes, mat: np.ndarray, title: str):
            im = ax.matshow(np.abs(mat), cmap='pink', vmin=0, vmax=1)
            ax.set_axis_off()
            ax.set_title(title)
            return im
        for ax, rotmat, title in zip(axes, rotmat_list, title_list):
            im = matshow(ax, rotmat[:N_cut, :N_cut], title)

        fig.colorbar(im, ax=axes.ravel().tolist())
        fig.suptitle(f"Figure 10. {rotvec=} (rad), $l < {self.level_rotmat_show}$")
        self.savefig(fig, "Figure 10.png")

    @task
    def figure_11(self):
        self.get_rotmat()
        i_rot = 0 # Different from `self.figure_10()`
        rotvec = self.rotvec_list[i_rot]
        rotmat_PSH = self.rotmat_PSH_list[i_rot]
        rotmat_nvSH = self.rotmat_nvSH_list[i_rot]
        
        level = self.level_recon
        shv = self.shv.cut(level)
        shv_nv = self.shv_nv.cut(level)

        h, w = self.hw_er
        sphFF = psh.SphereFrameField.from_equirect(h, w)
        vabs = 0.1
        self.log.print("="*20, "[ReproduceSY24.figure_11]", "="*20)
        # ========== Main ==========
        @self.cached(f"sponza_1024_SH_L{level}_er{h},{w}_s%d.exr", 'StokesField er', "Reconstruct SHVec to equirect image")
        def get_SHVec_recon():
            return psh.StokesField.from_SHCoeff(shv, sphFF)
        stkF_SHrec = get_SHVec_recon()

        @self.cached(f"sponza_1024_nv{self.nvSH_type}SH_L{level}_er{h},{w}_s%d.exr", 'StokesField er', "Reconstruct nvSHVec to equirect image")
        def get_nvSHVec_rec():
            return StokesField_from_nvSHCoeff(shv_nv, sphFF, self.nvSH_type)
        stkF_nvSHrec = get_nvSHVec_rec()

        sphFF_rot = sphFF.apply_rotation(rotvec)
        stkF_SH_rot_rec = psh.StokesField.from_SHCoeff(rotvec @ shv, sphFF_rot)
        shv_nv_rot = nvSHVec_apply_rotmat(shv_nv, rotvec, rotmat_nvSH, self.nvSH_type)
        stkF_nvSH_rot_rec = StokesField_from_nvSHCoeff(shv_nv_rot, sphFF_rot, self.nvSH_type)

        # ========== Figure ==========
        fig = plt.figure(figsize=(20, 4))
        subfigs = fig.subfigures(1, 3, width_ratios=[1, 3, 3])

        # Original envmap
        axes = subfigs[0].subplots(2, 1)
        for i, ax in enumerate(axes):
            im = imshow_comp(ax, self.stkComp_er[::-1], i+1, vabs, f"$s_{i+1}$ Stokes component")
        subfigs[0].colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', shrink=0.6)
        subfigs[0].suptitle("\nPolarized environment map\n(same as Figs. 5 & 6)")

        # (a) and (b)
        titles = ["(a) Naive approach: Scalar SH + $\\mathbf{F}_{\\theta\\phi}$",
                  "(b) Spin-2 SH"]
        for i_basis, (subfig, title) in enumerate(zip(subfigs[1:], titles)):
            subfig.suptitle(title)
            for (i,j), ax in np.ndenumerate(subfig.subplots(2, 2)):
                if j == 0:
                    stkF = [stkF_nvSHrec, stkF_SHrec][i_basis]
                    if i == 0:
                        ax.set_title("(ii) Not rotated")
                else:
                    stkF = [stkF_nvSH_rot_rec, stkF_SH_rot_rec][i_basis]
                    if i == 0:
                        ax.set_title("(iv) Rotated in the basis space (freq. domain)")
                imshow_comp(ax, stkF.Stk[::-1], i+1, vabs)
        fig.suptitle("Figure 11.")
        fig.set_layout_engine('constrained')
        self.savefig(fig, "Figure 11.png")
    
    def get_visibility(self, file_obj: Union[str, Path], i_vert: int, sphG: psh.SphereGrid) -> psh.ScalarField:
        self.log.print("="*20, "[ReproduceSY24.get_visibility]", "="*20)
        vis_label = f"{Path(file_obj).stem}_v{i_vert}_{sphG.edge}"
        level_shv = 2*self.level_TPmat-1

        # ========== Visibility mask ==========
        if not hasattr(self, 'vis_mask'):
            assert isinstance(sphG, psh.SphereGridCube)
            
            @self.cached(f"vis_mask_{vis_label}.npy", 'numpy', "Visibility mask")
            def get_visibility_mask() -> np.ndarray:
                import mitsuba as mi
                mi.set_variant('cuda_rgb', 'cuda_ad_rgb', 'llvm_rgb', 'llvm_ad_rgb')
                self.log.print(f"{mi.__version__ = }, {mi.variant() = }")

                scene: mi.Scene = mi.load_dict({'type':'scene', 'mesh':{'type': 'obj', 'filename': str(file_obj)}})
                mesh: mi.Mesh = scene.shapes()[0]
                assert i_vert > 0 and i_vert < mesh.vertex_count(), f"{i_vert = }, {mesh.vertex_count() = }"
                shape = sphG.shape
                dirs = -mi.Vector3f(sphG.vec.reshape(-1, 3))
                orgs = mesh.vertex_position(i_vert) + mi.math.RayEpsilon * dirs
                ray = mi.Ray3f(orgs, dirs)
                si = scene.ray_intersect(ray)
                mask = ~si.is_valid()
                mask_np = mask.numpy()
                del dirs, orgs, ray, si, mask
                return mask_np.reshape(shape)
            self.vis_mask = get_visibility_mask()
            self.log.print(f"# {self.vis_mask.shape = }")
        
        self.vis_scalF = psh.ScalarField(sphG, self.vis_mask)
        # ========== Visibility SH coefficients ==========
        if not hasattr(self, 'vis_shv'):
            @self.cached(f"vis_SHVec_{vis_label}_L{level_shv}.npz", 'SHVec', "Visibility SH coefficients")
            def get_visibility_SHVec() -> psh.SHVec:
                return self.vis_scalF.SHCoeff(level_shv, "REAL")
            self.vis_shv = get_visibility_SHVec()
        
        # ========== Visibility triple product matrix ==========
        if not hasattr(self, 'vis_TPmat'):
            @self.cached(f"vis_TPmat_{vis_label}_L{self.level_TPmat}.npz", 'SHMat', "Visibility TP matrix")
            def get_visibility_TPmat():
                return self.vis_shv.to_TPmat(self.level_TPmat, "POLAR3")
            self.vis_TPmat = get_visibility_TPmat()
            self.log.print(f"# {self.vis_TPmat.coeff.shape = }")

    @task
    def figure_13(self):
        self.get_visibility(self.file_obj, self.i_vert, psh.SphereGridCube(self.edge_vis))
        # Now got attributes `.vis_mask`, `.vis_scalF`, `.vis_shv`, and `.vis_TPmat`

        self.log.print("="*20, "[ReproduceSY24.figure_13]", "="*20)
        level_cut = 5
        shv_cut = self.vis_shv.cut(level_cut)
        for fig_mode in ["", " (alternative)"]:
            fig = plt.figure(figsize=(15, 8))
            subfigs = fig.subfigures(1, 2, width_ratios=[1, 1.2])

            # ========== Figure 13(a) ==========
            subfigs[0].suptitle("\n\n(a) Visibility at a vertex")
            axes = subfigs[0].subplots(1, 2, width_ratios=[8, 1])
            axes[0].imshow(cubemask2image(self.vis_mask, 4, [0.89019608, 0.34117647, 0.23137255]))
            axes[0].set_axis_off()

            vec = shv_cut.coeff[..., None]
            if fig_mode == "":
                axes[1].matshow(vec)
            else:
                im = axes[1].matshow(vec, cmap=psh.pltcmap_diverge, norm=mcolors.CenteredNorm())
                plt.colorbar(im, ax=axes[1])
            axes[1].set_axis_off()
            axes[1].set_title(f"SH coeff. vector")
            
            # ========== Figure 13(b) ==========
            subfigs[1].suptitle(f"\n\n(b) SH coefficients & Triple product matrix\n($l<{self.level_TPmat}$)")
            if fig_mode == "":
                for (i,j), ax in np.ndenumerate(subfigs[1].subplots(3, 3)):
                    im = ax.matshow(self.vis_TPmat.coeff[:,:,i,j])
                    ax.set_axis_off()
                    plt.colorbar(im, ax=ax, shrink=0.8)
            else:
                self.vis_TPmat.matshow(norm=mcolors.CenteredNorm(), fig=subfigs[1])
            title = "Figure 13. A coefficient vector of visibility to the triple product matrix"
            if fig_mode != "":
                title += "\n(using a colormap which is different from the paper but consistent with other figures)"
            fig.suptitle(title)
            self.savefig(fig, f"Figure 13{fig_mode}.png")
    
    @task
    def figure_16(self):
        self.log.print("="*20, "[ReproduceSY24.figure_16]", "="*20)

        level = self.level_conv
        outdir = self.outdir_data/"[convolution comparison] sponza_1024"
        h = int(level / np.sqrt(2))
        w = 2*h
        vabs = 1/3
        
        shv = self.shv_high.cut(level)
        kernel = psh.StokesConvKernel(lambda theta: (np.pi-np.expand_dims(theta, (-1,-2)))*np.eye(4))
        kernel_name = "isoLin" # Isotropic in Mueller matrix, linear in zenith
        n_sample = 16 * (level**2)

        @self.cached(f"SHConv_{kernel_name}_L{level}_{n_sample}.npy", 'SHConv', "PSH convolution coefficient")
        def get_kernel_coeff():
            return kernel.SHCoeff(level, n_sample)
        shc = get_kernel_coeff()

        # ========== Convolution in angular domain (sphere) ==========
        self.log.print("# === The first method: convolution in angular domain ===")
        psh.SphereGridEquirect.clear_cache()
        msg = "* [Report] Reconstruct PSH coefficient of the original envmap to the angular domain:\n" + \
              "           %.6f sec"
        with psh.Tictoc(msg, file=[sys.stdout, self.log.file]) as tictoc1a:
            sphG = psh.SphereGridEquirect(h, w)
            stkF = psh.StokesField.from_SHCoeff(shv, sphG.ThetaPhiFrameField())
        psh.imwrite_Stk(outdir/f"L{level}_recon_s%d.exr", stkF.Stk)

        msg = "* [Report] Perform convolution in the angular domain:\n" + \
              "           %.6f sec"
        with psh.Tictoc(msg, file=[sys.stdout, self.log.file]) as tictoc1b:
            stkF_angconv = kernel.apply(stkF)
        psh.imwrite_Stk(outdir/f"L{level}_recon_angconv_s%d.exr", stkF_angconv.Stk)

        # ========== Convolution in frequency domain (PSH) ==========
        self.log.print("# === The second method: convolution in frequency (PSH) domain ===")
        msg = "* [Report] Perform convolution in the frequency domain:\n" + \
              "           %.6f sec"
        with psh.Tictoc(msg, file=[sys.stdout, self.log.file]) as tictoc2a:
            shv_conv = shc @ shv
        shv_conv.save(outdir/f"L{level}_conv.npz")

        psh.SphereGridEquirect.clear_cache()
        msg = "* [Report] Reconstruct convolved PSH coefficient to the angular domain:\n" + \
              "           %.6f sec"
        with psh.Tictoc(msg, file=[sys.stdout, self.log.file]) as tictoc2b:
            sphG = psh.SphereGridEquirect(h, w)
            stkF_freqconv = psh.StokesField.from_SHCoeff(shv_conv, sphG.ThetaPhiFrameField())
        psh.imwrite_Stk(outdir/f"L{level}_freqconv_recon_s%d.exr", stkF.Stk)

        # ========== Figure ==========
        fig, axes = plt.subplots(2, 3, figsize=(18, 6))
        fig.suptitle("Figure 16(b). Resulting images of polarized spherical convolution in each domain"
                     f"(Coefficients in $l<{level}$ and ${w}\\times{h}$ images)")
        for (i,j), ax in np.ndenumerate(axes):
            stkF_ = [stkF, stkF_angconv, stkF_freqconv][j]
            
            title = ["Initial, ",
                     f"(${tictoc1a.recorded:.4f} + {tictoc1b.recorded:.4f}$ s)\nAngular domain conv., ",
                     f"(${tictoc2a.recorded:.4f} + {tictoc2b.recorded:.4f}$ s)\nPSH conv., "][j] if i==0 else ""
            title += f"$s_{i+1}$"
            im = imshow_comp(ax, stkF_.Stk[::-1], i+1, vabs, title)
        plt.colorbar(im, ax=axes.ravel().tolist())
        self.savefig(fig, f"Figure 16.png")

    @task
    def figure_17(self):
        cod_type = self.cod_type_roteq
        level = self.level_roteq
        grid_level = self.grid_level_roteq
        brdf = psh.SHMat.from_npz_file(self.file_brdf).cut(level, cod_type)
        """
        Note that this `psh.SHMat` object for a pBRDF can be reproduced by `psh.MuellerField.from_SB20().SHCoeff()`.
        However we just use the precomputed file
        since it is just input data of the experiment described in Figure 17.
        """
        self.log.print("="*20, "[ReproduceSY24.figure_17]", "="*20)

        # ========== Main ==========
        @self.cached(f"pBRDF_roteq_mip_6_gold_ds1_L{level}_P{int(cod_type)}_gridL{grid_level}.npy", 'numpy', "Average pBRDF over surface normals")
        def get_brdf_roteqmip():
            n_grid = 2**grid_level
            theta, phi = np.mgrid[1e-4:np.pi-1e-4:n_grid*1j, 0:2*np.pi:2*n_grid*1j]

            normal = psh.sph2vec(theta, phi, axis=-1)
            brdf_rots = np.zeros(normal.shape[:2], dtype=np.object_)

            for i,j in tqdm(np.indices(normal.shape[:2]).reshape(2,-1).T):
                brdf_rots[i, j] = brdf.set_normal(normal[i,j])
            assert brdf_rots[0, 0].dom_type == psh.DomType.UNI

            rbrdf_mip = np.zeros((grid_level,) + brdf.shape_chan + (psh.level2num(level, "BI"), int(cod_type), int(cod_type)), dtype=brdf.coeff.dtype)

            for idx_gl in range(grid_level):
                stride = 2**(grid_level-idx_gl-1)
                n_grid_curr = 2**(idx_gl+1)
                weight = np.sin(theta[::stride, ::stride]) * (np.pi/n_grid_curr) * (np.pi/n_grid_curr)
                brdf_rots_w = brdf_rots[::stride, ::stride] * weight
                rbrdf_mip[idx_gl] = np.mean(brdf_rots_w).to_domtype("BI").coeff
            return rbrdf_mip
        rbrdf_mip = get_brdf_roteqmip()
        print(f"# {rbrdf_mip.shape = }")

        def plot_conv_error(rbrdf_mip: ArrayLike):
            assert rbrdf_mip.ndim == 5 # [grid_level, RGB, N, po, pi]
            grid_level = rbrdf_mip.shape[0]
            N = rbrdf_mip.shape[-3]
            level = psh.num2level_assert(N, "BI")
            assert rbrdf_mip.shape[-2:] == (3, 3) # `psh.CodType.POLAR3` only

            ### Constraints for nonzero coefficients

            titles = ["spin 0-to-2", "spin 2-to-0", "spin 2-to-2"]
            figsize = (11.3, 11.5/3)
            nrow_plot = 1
            ncol_plot = len(titles)
            axes_mean = (1, 2, 3)
            fig = plt.figure(figsize=figsize)

            for i_polar, title in enumerate(titles):
                # rbrdf_mip_curr [grid_level, channel, N(DomType.BI), p]
                if i_polar == 0:
                    rbrdf_mip_curr = rbrdf_mip[..., 1:3, 0]
                elif i_polar == 1:
                    rbrdf_mip_curr = rbrdf_mip[..., 0, 1:3]
                # rbrdf_mip_curr [grid_level, channel, N(DomType.BI), po, pi]
                else:
                    rbrdf_mip_curr = rbrdf_mip[..., 1:3, 1:3]
                
                mse = np.zeros((grid_level, level-2)) # [grid_level, level]
                    
                K00_ms = np.zeros((grid_level,))
                for l in range(2, level):
                    idx00 = psh.lms2idx([l,l,0,0], "BI")
                    
                    # K00(a|b)[grid_level, channel] for m_i,m_0 == 0,0
                    if i_polar < 2:
                        K00 = psh.vec2comp(rbrdf_mip_curr[:,:,idx00,:]) 
                        K00_ms += np.mean(np.abs(K00)**2, 1)
                    else:
                        K00a, K00b = psh.mat2comppair(rbrdf_mip_curr[:,:,idx00,:,:])
                        K00_ms += np.mean(np.abs(K00a)**2, 1) + np.mean(np.abs(K00b)**2, 1)
                    
                    for m_abs in range(1, l+1):
                        # idx[+-, +-]
                        idx = psh.lms2idx([[ [l,l,m_abs,m_abs],  [l,l,-m_abs,m_abs] ],
                                    [ [l,l,m_abs,-m_abs], [l,l,-m_abs,-m_abs] ]], "BI")
                        
                        phase = (-1)**(m_abs % 2)
                        if i_polar < 2:
                            if i_polar == 0:
                                U = 1/np.sqrt(2) * np.array([[1,     -1j],
                                                            [phase, phase*1j]])
                            else:
                                U = 1/np.sqrt(2) * np.array([[1,  phase],
                                                            [1j, -phase*1j]]).conj()
                            # [grid_level, channel, +-, +-]
                            Kmm1 = psh.vec2comp(rbrdf_mip_curr[...,idx,:])
                            Kmm2 = U * np.expand_dims(K00, (-1,-2))
                            # [grid_level, +-, +-] | [grid_level]
                            mse_curr = np.mean(np.abs(Kmm1 - Kmm2)**2, axes_mean)
                        else:
                            Ua = np.eye(2)
                            Ub = phase * np.array([[0, 1],
                                                [1, 0]])
                            # [grid_level, channel, +-, +-]
                            Kmm1a, Kmm1b = psh.mat2comppair(rbrdf_mip_curr[...,idx,:,:])
                            Kmm2a = Ua * np.expand_dims(K00a, (-1,-2))
                            Kmm2b = Ub * np.expand_dims(K00b, (-1,-2))
                            # [grid_level, +-, +-] | [grid_level]
                            mse_curr = np.mean(np.abs(Kmm1a - Kmm2a)**2, axes_mean) + np.mean(np.abs(Kmm1b - Kmm2b)**2, axes_mean)
                        
                        mse[:,l-2,...] += mse_curr #/ np.expand_dims(np.mean(np.abs(K00)**2, 1), (1,2))
                        
                ## Plot
                X = 2 ** (np.arange(grid_level)+1)
                ax = fig.add_subplot(nrow_plot, ncol_plot, i_polar+1)
                ax.set_title(f"{title}")
                for l in np.arange(2, level):
                    label = f"order $l={l}$"
                    # ax.plot(X, np.sqrt(mse[:,l-2,so,si]), label=label)
                    ax.plot(X, np.sqrt(mse[:,l-2]), label=label)
                ax.set_xlabel('Grid resolution $n$')
                if i_polar == 0:
                    ax.set_ylabel('RMSE coeff. constraitns')
                plt.xscale('log'); plt.yscale('log')
                # ax.set_ylim([2.5e-8, [0.1, 0.1, 1][i_polar]])
                ax.set_ylim([2.5e-8, 1])
                
                ### Constraints for zero coefficients
                li, lo, mi, mo = psh.level2lms(level, "BI", unstack=True)
                mask_zero = np.abs(mi) != np.abs(mo)
                if i_polar < 2:
                    rmse_zero = psh.rms(rbrdf_mip_curr[:,:,mask_zero,:], (1,2,3))
                else:
                    rmse_zero = psh.rms(rbrdf_mip_curr[:,:,mask_zero,:,:], (1,2,3,4))
                # ax.plot(X, rmse_zero, label = 'zero coeff. ($|m_i|\\ne|m_o|$)')
                ax.plot(X, rmse_zero, label = '($|m_i|\\ne|m_o|$)')
            ax.legend()
            fig.suptitle("(b) Resulting errors"); fig.tight_layout()
            return fig
        self.savefig(plot_conv_error(rbrdf_mip), "Figure 17.png")

    def run_all(self):
        with psh.Tictoc("========== All session is done in %.4f sec. ==========", file=[sys.stdout, self.log.file]):
            for func in task_list:
                func(self)

mpl.rcParams['figure.figsize'] = [12.8, 9.6]
mpl.rcParams['figure.dpi'] = 300

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to reproduce Figures 5, 7, 10, 11, 13, 16, and 17 of the paper [Yi et al. 2024]\n"
                    '"Spin-Weighted Spherical Harmonics for Polarized Light Transport" by\n'
                    "Shinyoung Yi, Donggun Kim, Jiwoong Na, Xin Tong, and Min H. Kim\n"
                    "(ACM Trans. Graph. Vol 43, Issue 4, No 127, 2024)",
        epilog="Shinyoung Yi (syyi@vclab.kaist.ac.kr)")
    
    parser.add_argument('-m', '--mode', type=str, choices= ['full', 'simple'], default='full',
                        help="['full'] Exactly reproduce the figures in the paper (default)\n"
                             "['simple'] Reproduce the figures with the same method but different parameters"
                             "of lower image resolutions and lower frequency bands (less # of coeff.).\n"
                             "Users can check this script runs without any error with the 'simple' mode.")
    parser.add_argument('--remove_cache', action='store_true',
                        help="Remove cached data ('reproduce/output' for 'full' mode, 'reproduce/output_simple' for 'simple' mode)")
    args = parser.parse_args()
    rep = ReproduceSY24(args.mode, args.remove_cache)
    rep.run_all()
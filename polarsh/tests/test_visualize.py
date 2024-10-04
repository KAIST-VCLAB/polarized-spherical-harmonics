from typing import Tuple
import pytest
from pathlib import Path
import numpy as np
import polarsh as psh

@pytest.fixture
def outdir() -> Path:
    path = Path("output")
    path.mkdir(exist_ok=True)
    return path

@pytest.fixture
def figsize() -> Tuple[int, int]:
    return (512, 512)

@pytest.fixture
def stkF() -> psh.StokesField:
    return psh.StokesField.from_cubeimage(psh.data_dir/"sponza_64_s%d.exr")

@pytest.fixture
def level() -> int:
    return 5

@pytest.fixture
def shv4(level: int, stkF: psh.StokesField) -> psh.SHVec:
    res = stkF.SHCoeff(5)
    assert res.cod_type == psh.CodType.POLAR4
    return res

@pytest.fixture
def shm4(level) -> psh.SHMat:
    return psh.SHMat.from_rotation([np.pi/2, 0, 0], level, 4, "REAL")

def test_visualize(outdir: Path, stkF: psh.StokesField, figsize: Tuple[int, int]):
    sphG = stkF.SphGrid
    sphFF = stkF.SphFF
    scalF = stkF.s0_ScalarField()

    outfile_list = ["SphereGrid.png", "SphereFrameField.png", "ScalarField.png", "StokesField.png"]
    psh.set_visoptions_vispy(figsize=figsize, arrow_dsamp=4)
    for sphObj, outfile in zip([sphG, sphFF, scalF, stkF], outfile_list):
        canvas = sphObj.visualize()
        canvas.show()
        img = canvas.render()
        outpath = outdir / ("test_visualize_" + outfile)
        flag = psh.imwrite(outpath, img)
        assert flag, f"{outpath = }"
        assert outpath.exists()

def test_matshow(outdir: Path, shv4: psh.SHVec, shm4: psh.SHMat):
    pref = "test_matshow"
    for cod_type in psh.CodType:
        shv = shv4.cut(cod_type=cod_type)
        fig = shv.matshow(0)
        fig.savefig(outdir/f"{pref}_SHVec_{cod_type}.png")
        if cod_type > psh.CodType.SCALAR:
            fig = shv.matshow(0, long=True)
            fig.savefig(outdir/f"{pref}_SHVec_{cod_type}_long.png")

        shm = shm4.cut(cod_type=cod_type)
        fig = shm.matshow()
        fig.savefig(outdir/f"{pref}_SHMat_{cod_type}.png")

if __name__ == "__main__" and False:
    from vispy import app
    outdir = outdir()
    figsize = figsize()
    test_visualize(outdir, figsize)
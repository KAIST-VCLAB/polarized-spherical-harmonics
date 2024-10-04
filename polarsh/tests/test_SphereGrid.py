import numpy as np
from polarsh.array import matmul_vec1d
from polarsh.sphere import rotvec2rot
from polarsh.SH import DomType
from polarsh.grid import SphereGridFibonacci, SphereGridEquirect, SphereGridCube, \
                         SphereGridPersp, SphereGridMirrorball

def test_constructors():
    # ---------- Initialize ----------
    n_samples = 100
    h, w = 10, 20
    edge = 16
    pi4 = 4*np.pi
    atol = 1e-4
    rtol = 1e-2

    # ---------- Main ----------
    print("# ========== DomType.UNI ==========")
    sphG = SphereGridFibonacci(n_samples)
    assert sphG.vec.shape == (n_samples, 3)
    assert sphG.dom_type == DomType.UNI
    assert np.allclose(sphG.weight().sum(), pi4, atol=atol, rtol=rtol)
    print("`SphereGridFibonacci`: \n", sphG)
    
    sphG = SphereGridEquirect(h, w)
    assert sphG.vec.shape == (h, w, 3)
    assert sphG.dom_type == DomType.UNI
    assert np.allclose(sphG.weight().sum(), pi4, atol=atol, rtol=rtol)
    print("`SphereGridEquirect`: \n", sphG)

    def sph_area(x_tan_half, y_tan_half):
        x, y = x_tan_half, y_tan_half
        return 4*np.arctan(x*y/np.sqrt(1+x*x+y*y))
    
    def sph_area_from_hwfov(h, w, fov):
        x_tan_half = np.tan(np.deg2rad(fov)/2)
        y_tan_half = x_tan_half * h/w
        return sph_area(x_tan_half, y_tan_half)

    fov = 40
    sphG = SphereGridPersp(h, w, fov)
    assert sphG.vec.shape == (h, w, 3)
    assert sphG.dom_type == DomType.UNI
    assert np.allclose(sphG.weight().sum(), sph_area_from_hwfov(h, w, fov), atol=atol, rtol=rtol)
    print("`SphereGridPersp`: \n", sphG)
    
    to_world = rotvec2rot([10, 1, 0.1])
    sphG2 = SphereGridPersp(h, w, fov, to_world)
    assert np.allclose(matmul_vec1d(to_world, sphG.vec), sphG2.vec, atol=atol, rtol=rtol)
    assert np.allclose(sphG2.weight().sum(), sph_area_from_hwfov(h, w, fov), atol=atol, rtol=rtol)

    fov = 90
    sphG3 = SphereGridPersp(h, h, fov)
    assert np.allclose(sphG3.weight().sum(), pi4/6, atol=atol, rtol=rtol)

    sphG = SphereGridMirrorball(edge)
    assert sphG.vec.shape == (edge, edge, 3)
    assert sphG.dom_type == DomType.UNI
    # TODO: SphereGrid.weight() does not work for SphereGrid.from_mirrorball
    print("`SphereGridMirrorball`: \n", sphG)

    if False:
        print("\n\n# ========== DomType.ISOBI ==========")
        sphG = SphereGrid.from_meshgrid(DomType.ISOBI, np.lin)

        print("\n\n# ========== TODO: DomType.BI ==========")
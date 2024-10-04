import numpy as np
from polarsh.array import *
from polarsh.sphere import *
from polarsh.SH import CodType
from polarsh.grid.StokesField import stokes_frame_convert
from utils import Report, ReportOption
import pytest

@pytest.fixture()
def opt():
    return ReportOption(True, True, True)



def test_rotation():
    print("# rotvec2rot, axisang2rot")
    Z = np.zeros((3,))
    assert np.array_equal(rotvec2rot(Z), np.eye(3))
    assert np.array_equal(rotvec2rot(Z, homog=True), np.eye(4))
    # assert True

    ang = 1.23e-13
    vec = np.array([[ang, 0, 0], [0, ang, 0], [0, 0, ang]])
    R = rotvec2rot(vec)
    exdims = [(2,3,4), (0,1,2), (0,1,4), (0,2,4)]
    exdims2 = [(3,4,5), (0,1,2), (0,1,5), (0,2,5)]
    axiss = [1, 4, 3, 3]
    for exdim,exdim2,axis in zip(exdims, exdims2, axiss):
        vec2 = np.expand_dims(vec, axis=exdim)
        R2 = rotvec2rot(vec2, axis=axis)
        print(f"\n# Permutation error from {vec.shape} to {vec2.shape}:")
        print(np.linalg.norm(R - np.sum(R2, exdim2)))
        assert np.allclose(R, np.sum(R2, exdim2))

    ## Size test
    V = np.random.rand(2,5,3,6,4)
    ang1 = np.random.rand(2,5,6,4)
    ang2 = np.random.rand(2,5,1,6,4)
    Rs = []
    Rs.append(axisang2rot(V, ang1, axis=2))
    Rs.append(axisang2rot(V, ang1, axis=-3))
    Rs.append(axisang2rot(V, ang2, axis=2))
    Rs.append(axisang2rot(V, ang2, axis=-3))
    Rs.append(rotvec2rot(V, axis=2))
    Rs.append(rotvec2rot(V, axis=-3))
    shapes = [R.shape for R in Rs]
    assert set(shapes)=={(2,5,3,3,6,4)}, f"Shape bug, shapes={shapes}"

    # ==============================
    print("# rotation2scipy, rotation2quat")
    for rotvec in vec:
        R = rotvec2rot(rotvec)
        R_scipy = rotation2scipy(rotvec)
        R_quat = rotation2quat(rotvec)
        assert np.allclose(R, R_scipy.as_matrix())
        assert np.allclose(R, R_quat.to_rotation_matrix)
        for i, R_ in enumerate([R, R_scipy, R_quat]):
            if i != 2:
                assert np.allclose(R, rotation2scipy(R_).as_matrix())
            assert np.allclose(R, rotation2quat(R_).to_rotation_matrix)



def test_align_azimuth():
    # def test_align_azimuth(eps = 1e-8):
    eps = 1e-8
    thh  = np.array([0.0, 0.0, 180.0, 10, 10, 90, 90])
    phh1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    phh2 = np.array([1.0, -10, 11, -12, 13, -14, 15])
    thd  = np.array([0.1, 179.9, 0.15, 0.0, 180, 0.0, 89])
    phd  = np.array([0.0, 20, 25, 25, 30, 0, 180.0])
    p1veci, p1veco = rus2vec(phd, thd, thh, phh1, ang_type=AngType.DEG)
    p2veci, p2veco = rus2vec(phd, thd, thh, phh2, ang_type=AngType.DEG)
    dphis, Rs, errs = align_azimuth((p1veci, p1veco), (p2veci, p2veco), ang_type=AngType.DEG)

    for i, (dphi, R, err) in enumerate(zip(dphis, Rs, errs)):
        print(f"\n# {i}th test")
        print(f"pair1: veci={p1veci[i]}\tveco={p1veco[i]}")
        print(f"pair2: veci={p2veci[i]}\tveco={p2veco[i]}")
        dphi_ref = phh2[i]-phh1[i]
        print(f"dphi: computed={dphi}\tref={dphi_ref}")
        assert diff_cyclic(dphi, dphi_ref, 360) < eps, "The error is out of the tolerance!"
        print(f"Out-of-azimuthal error: {err}")



def test_Rusinkiewicz(opt: ReportOption):
    n = 31 #101
    phi_d_range = np.linspace(0, 2*np.pi, 2*n)
    theta_d_range = np.linspace(0, np.pi/2, n)
    theta_h_range = np.linspace(0, np.pi, n)
    phi_h_range = np.linspace(0, 2*np.pi, 2*n)

    print("# ========== Test `rus2vec` ==========")
    print("Case 1: theta_h = 0")
    phd, thd, phh = np.meshgrid(phi_d_range, theta_d_range, phi_h_range)
    veci, veco = rus2vec(phd, thd, 0, phh)
    Report.report(veci, sph2vec(thd, phh+phd), option=opt)
    Report.report(veco, sph2vec(thd, phh+phd+np.pi), option=opt)

    print("\nCase 2: theta_h = pi")
    veci, veco = rus2vec(phd, thd, np.pi, phh)
    Report.report(veci, sph2vec(np.pi-thd, phh-phd+np.pi), option=opt)
    Report.report(veco, sph2vec(np.pi-thd, phh-phd), option=opt)

    print("\nCase 3: theta_d = 0")
    phd, thh, phh = np.meshgrid(phi_d_range, theta_h_range, phi_h_range)
    veci, veco = rus2vec(phd, 0, thh, phh)
    Report.report(veci, sph2vec(thh, phh), option=opt)
    Report.report(veco, sph2vec(thh, phh), option=opt)

    print("\nCase 4: theta_d = pi/2")
    veci, veco = rus2vec(phd, np.pi/2, thh, phh)
    vech = sph2vec(thh, phh)
    Report.report(veci, -veco, option=opt)
    Report.report((veci*vech).sum(-1), 0, option=opt)

    print("\nCase 5: phi_d = 0")
    thd, thh, phh = np.meshgrid(theta_d_range, theta_h_range, phi_h_range)
    veci, veco = rus2vec(0, thd, thh, phh)
    Report.report(veci, sph2vec(thh+thd, phh), atol=3.e-8, option=opt)
    Report.report(veco, sph2vec(thh-thd, phh), atol=3.e-8, option=opt)

    print("\nCase 6: phi_d = pi")
    veci, veco = rus2vec(np.pi, thd, thh, phh)
    Report.report(veci, sph2vec(thh-thd, phh), atol=3.e-8, option=opt)
    Report.report(veco, sph2vec(thh+thd, phh), atol=3.e-8, option=opt)

    print("\nCase 7: phi_d = pi/2 and theta_h = pi/2")
    thd, phh = np.meshgrid(theta_d_range, phi_h_range)
    veci, veco = rus2vec(np.pi/2, thd, np.pi/2, phh)
    Report.report(veci, sph2vec(np.pi/2, phh+thd), option=opt)
    Report.report(veco, sph2vec(np.pi/2, phh-thd), option=opt)

    print("\nCase 6: phi_d = pi")
    veci, veco = rus2vec(-np.pi/2, thd, np.pi/2, phh)
    Report.report(veci, sph2vec(np.pi/2, phh-thd), option=opt)
    Report.report(veco, sph2vec(np.pi/2, phh+thd), option=opt)

    ## Declare variables
    def bisph2vec(theta_i, theta_o, phi_i, phi_o=0, ang_type = AngType.RAD):
        return sph2vec(theta_i, phi_i, ang_type=ang_type), sph2vec(theta_o, phi_o, ang_type=ang_type)
    
    theta_i_rad, theta_o_rad, phi_i_rad, phi_o_rad = np.mgrid[0:np.pi:n*1j, 0:np.pi:n*1j, 0:2*np.pi:n*1j, 0:2*np.pi:n*1j]
    theta_i_deg, theta_o_deg, phi_i_deg, phi_o_deg = np.mgrid[0:180:n*1j, 0:180:n*1j, 0:360:n*1j, 0:360:n*1j]
    sph_rad = np.array([theta_i_rad, theta_o_rad, phi_i_rad, phi_o_rad])
    sph_deg = np.array([theta_i_deg, theta_o_deg, phi_i_deg, phi_o_deg])
    vec_rad = bisph2vec(*sph_rad, ang_type=AngType.RAD)
    vec_deg = bisph2vec(*sph_deg, ang_type=AngType.DEG)

    phi_d_rad, theta_d_rad, theta_h_rad, phi_h_rad = np.mgrid[0:2*np.pi:n*1j, 0:np.pi/2:n*1j, 0:np.pi:n*1j, 0:2*np.pi:n*1j]
    phi_d_deg, theta_d_deg, theta_h_deg, phi_h_deg = np.mgrid[0:360:n*1j, 0:90:n*1j, 0:180:n*1j, 0:360:n*1j]
    rus_rad = np.array([phi_d_rad, theta_d_rad, theta_h_rad, phi_h_rad])
    rus_deg = np.array([phi_d_deg, theta_d_deg, theta_h_deg, phi_h_deg])

    sph2rus_rad = np.array(sph2rus(*sph_rad, ang_type=AngType.RAD))
    sph2rus_deg = np.array(sph2rus(*sph_deg, ang_type=AngType.DEG))
    rus2sph_rad = np.array(rus2sph(*rus_rad, ang_type=AngType.RAD))
    rus2sph_deg = np.array(rus2sph(*rus_deg, ang_type=AngType.DEG))
    rus2vec_rad = np.array(rus2vec(*rus_rad, ang_type=AngType.RAD))
    rus2vec_deg = np.array(rus2vec(*rus_deg, ang_type=AngType.DEG))

    sph2rus2sph = np.array(rus2sph(*sph2rus_rad))
    sph2rus2vec = np.array(rus2vec(*sph2rus_rad))
    rus2sph2rus = np.array(sph2rus(*rus2sph_rad))
    rus2sph2vec = np.array(bisph2vec(*rus2sph_rad))

    print("\n\n# ========== `rus2vec`: degree-radian consistency ==========")
    Report.report(rus2vec_rad, rus2vec_deg, atol=3e-8, option=opt)

    print("\n\n# ========== Test `sph2rus` ==========")
    print("[theta_d] RAD")
    Report.report_ge(sph2rus_rad[1], 0, atol=0, option=opt)
    Report.report_le(sph2rus_rad[1], np.pi/2, atol=1e-15, option=opt)
    print("[theta_h] RAD")
    Report.report_ge(sph2rus_rad[2], 0, atol=0, option=opt)
    Report.report_le(sph2rus_rad[2], np.pi, atol=0, option=opt)

    print("[theta_d] DEG")
    Report.report_ge(sph2rus_deg[1], 0, atol=0, option=opt)
    Report.report_le(sph2rus_deg[1], 90, atol=5e-14, option=opt)
    print("[theta_h] DEG")
    Report.report_ge(sph2rus_deg[2], 0, atol=0, option=opt)
    Report.report_le(sph2rus_deg[2], 180, atol=0, option=opt)


    print("[degree-radian consistency]")
    sph2rus2vec_deg = np.array(rus2vec(*sph2rus_deg, ang_type=AngType.DEG))
    Report.report(sph2rus2vec, sph2rus2vec_deg, option=opt)

    print("\n\n# ========== Validate `sph2rus` using `sph2vec` ==========")
    Report.report(sph2rus2vec, vec_rad, option=opt)

    print("\n\n# ========== Test `rus2sph` ==========")
    print("[rus2sph2vec == rus2vec]")
    Report.report(rus2sph2vec, rus2vec_rad, option=opt)

    print("\n[degree-radian consistency]")
    Report.report(rus2sph2vec, bisph2vec(*rus2sph_deg, ang_type=AngType.DEG), atol=3e-8, option=opt)

    print("\n\n# ========== Validate `rus2sph2rus` ==========")
    Report.report(rus2vec(*rus2sph2rus), rus2vec_rad, atol=1.5e-8, option=opt)

    print("\n\n# ========== Validate `sph2rus2sph` ==========")
    Report.report(bisph2vec(*sph2rus2sph), vec_rad, atol=3e-8, option=opt)



def test_polar_frame_convert():
    rotvecs = np.array([
        [1,0,0],[0,-2,0],[0,0,4.5],[1.1,2.0,-3.0],[0,0,0]])
    angles = [0.0, 1.0, -5.0, 30, -60]
    F1 = rotvec2rot(rotvecs)
    Rz = axisang2rot(unit_z, colvec(angles)*np.pi/180)
    F2 = F1 @ Rz
    Ms = []
    for ctype in list(CodType)[1:]:
        M = stokes_frame_convert(F1, F2, cod_type=ctype, quiet=False)
        Ms.append(M)
    for i,angle in enumerate(angles):
        cos2 = np.cos(2*angle*np.pi/180)
        sin2 = np.sin(2*angle*np.pi/180)
        M_ref = np.array([[cos2, sin2], [-sin2, cos2]])
        M_ref3 = np.eye(3)
        M_ref4 = np.eye(4)
        M_ref3[1:,1:] = M_ref
        M_ref4[1:3, 1:3] = M_ref
        assert_error_bound(Ms[0][i,:,:]-M_ref, name=f"{i}-th frames, CodType.POLAR2", quiet=False)
        assert_error_bound(Ms[1][i,:,:]-M_ref3, name=f"{i}-th frames, CodType.POLAR4", quiet=False)
        assert_error_bound(Ms[2][i,:,:]-M_ref4, name=f"{i}-th frames, CodType.POLAR4", quiet=False)

if __name__ == "__main__" and False:
    test_Rusinkiewicz(opt())
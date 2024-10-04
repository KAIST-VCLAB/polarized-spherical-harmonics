"""
Test `SHVec` and `SHMat` classes only in the coefficient (frequency) domain,
without using `SphereGrid`, `ScalarField`, `StokesField`, and `MuellerField` classes.
"""
import pytest
from typing import Tuple, List
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation
from polarsh import DomType, CodType, SHType, level2num, SHVec, SHMat

@pytest.fixture(scope="module")
def level() -> int:
    return 5

@pytest.fixture(scope="module")
def pairs_shv_shm(level: int) -> List[Tuple[SHVec, SHMat]]:
    N = level2num(level, DomType.UNI)
    res = []
    sh_type = SHType.REAL
    for cod_type in list(CodType):
        vcoeff = np.arange(N*int(cod_type))+0.1
        mcoeff = np.arange(N*N*int(cod_type)*int(cod_type))+0.1
        if cod_type != CodType.SCALAR:
            vcoeff = vcoeff.reshape(N, int(cod_type))
            mcoeff = mcoeff.reshape(N, N, int(cod_type), int(cod_type))
        else:
            mcoeff = mcoeff.reshape(N, N)
        
        ch_offset = 0.01*np.arange(3)
        vcoeffs = [vcoeff, vcoeff + np.expand_dims(ch_offset, tuple(np.arange(vcoeff.ndim)+1))]
        mcoeffs = [mcoeff, mcoeff + np.expand_dims(ch_offset, tuple(np.arange(mcoeff.ndim)+1))]
        for vc, mc in zip(vcoeffs, mcoeffs):
            res.append((SHVec(vc, cod_type, sh_type), SHMat(mc, DomType.UNI, cod_type, sh_type)))
    return res

@pytest.fixture()
def rotvecs() -> List[Tuple[float, float, float]]:
    """
    Use tuples for rotvecs rather than lists to make them dictionary keys.
    """
    res = [(0.1, 0, 0), (0, 0.2, 0), (0, 0, 0.3), (1, 0.3, 0.7)]
    return res

def test_SHCoeff_real_comp(pairs_shv_shm: List[Tuple[SHVec, SHMat]]):
    for shv, shm in pairs_shv_shm:
        assert shv.sh_type == SHType.REAL

        print("# ========== Cycle consistency of `SHVec.to_shtype` and `SHMat.to_shtype` ==========")
        shv_rc = shv.to_shtype(SHType.COMP)
        shm_rc = shm.to_shtype(SHType.COMP)
        shv_rcr = shv_rc.to_shtype(SHType.REAL)
        shm_rcr = shm_rc.to_shtype(SHType.REAL)
        assert shv.allclose(shv_rcr)
        assert shm.allclose(shm_rcr)

        shv_rcrc = shv_rcr.to_shtype(SHType.COMP)
        shm_rcrc = shm_rcr.to_shtype(SHType.COMP)
        assert shv_rc.allclose(shv_rcrc)
        assert shm_rc.allclose(shm_rcrc)
        
        print("# ========== Consistency between `SHVec.to_shtype` and `SHMat @ SHVec` ==========")
        shv2_r = shm @ shv
        shv2_c = shm_rc @ shv_rc
        assert shv2_r.allclose(shv2_c.to_shtype(SHType.REAL))
        assert shv2_c.allclose(shv2_r.to_shtype(SHType.COMP))
    
def test_rotation(level: int, pairs_shv_shm: List[Tuple[SHVec, SHMat]], rotvecs: List[ArrayLike]):
    print("# ========== Identity rotation ==========")
    N = level2num(level, DomType.UNI)
    shm_eye_dict = dict()
    for sh_type in list(SHType):
        for cod_type in list(CodType):
            shm = SHMat.from_rotation([0, 0, 0], level, cod_type, sh_type)
            shm_eye_dict[(cod_type, sh_type)] = shm

            if cod_type == CodType.SCALAR:
                assert np.allclose(shm.coeff, np.eye(N, N))
            else:
                p = int(cod_type)
                identity = np.moveaxis(np.eye(N*p, N*p).reshape(N, p, N, p), 1, 2)
                assert np.allclose(shm.coeff, identity)
        
    for shv_r, _ in pairs_shv_shm:
        assert shv_r.allclose([0, 0, 0] @ shv_r) # SHVec.__rmatmul__()
        for sh_type in list(SHType):
            if sh_type == shv_r.sh_type:
                shv_curr = shv_r
            else:
                shv_curr = shv_r.to_shtype(sh_type)
            shm = shm_eye_dict[(shv_curr.cod_type, sh_type)]
            assert shv_curr.allclose(shm @ shv_curr) # SHMat.from_rotation() followed by SHMat.__matmul__()
    

    print("# ========== Real-complex consistency of `SHMat.from_rotation` and `SHMat.to_shtype` ==========")
    shm_dict = dict()
    for rotvec in rotvecs:
        for cod_type in list(CodType):
            for sh_type in list(SHType):
                shm_dict[(rotvec, cod_type, sh_type)] = SHMat.from_rotation(rotvec, level, cod_type, sh_type)
            assert shm_dict[(rotvec, cod_type, SHType.REAL)].allclose(shm_dict[(rotvec, cod_type, SHType.COMP)].to_shtype(SHType.REAL))
            assert shm_dict[(rotvec, cod_type, SHType.COMP)].allclose(shm_dict[(rotvec, cod_type, SHType.REAL)].to_shtype(SHType.COMP))

    print("# ========== Consistency of `rotvec @ SHVec` and `SHMat.from_rotation() @ SHVec` ==========")
    print("# ========== Real-complex consistency of `rotvec @ SHVec` ==========")
    for shv_r, _ in pairs_shv_shm:
        shv_c = shv_r.to_shtype(SHType.COMP)
        for rotvec in rotvecs:
            shvrot1_r: SHVec = rotvec @ shv_r
            shvrot1_c: SHVec = rotvec @ shv_c
            shvrot2_r: SHVec = shm_dict[(rotvec, shv_r.cod_type, SHType.REAL)] @ shv_r
            shvrot2_c: SHVec = shm_dict[(rotvec, shv_r.cod_type, SHType.COMP)] @ shv_c
            assert shvrot1_r.allclose(shvrot2_r)
            assert shvrot1_c.allclose(shvrot2_c)
            assert shvrot1_r.allclose(shvrot1_c.to_shtype(SHType.REAL))
            assert shvrot1_c.allclose(shvrot1_r.to_shtype(SHType.COMP))
    
    print("# ========== Composition of rotations ==========")
    for rotvec1 in rotvecs:
        for rotvec2 in rotvecs:
            rotvec12 = (Rotation.from_rotvec(rotvec1) * Rotation.from_rotvec(rotvec2)).as_rotvec()
            for cod_type in list(CodType):
                for sh_type in list(SHType):
                    shm_from_r12 = SHMat.from_rotation(rotvec12, level, cod_type, sh_type)
                    shm_matmul12 = shm_dict[(rotvec1, cod_type, sh_type)] @ shm_dict[(rotvec2, cod_type, sh_type)]
                    assert shm_from_r12.allclose(shm_matmul12)

def test_SHCoeff_tabulate():
    level = 3
    N_UNI = level2num(level, "UNI")
    shape_list = [(), (1,), (3,), (2, 3), (2, 3, 4)]
    
    for dom_type in [None] + list(DomType):
        for cod_type in CodType:
            for i_sh, shape in enumerate(shape_list):
                if dom_type is None:
                    # SHVec
                    shape += (N_UNI,) if cod_type == CodType.SCALAR else (N_UNI, int(cod_type),)
                    coeff = np.arange(np.prod(shape), dtype=float).reshape(shape)
                    shc = SHVec(coeff, cod_type, "REAL")
                else:
                    # SHMat
                    if dom_type == DomType.UNI:
                        shape += (N_UNI, N_UNI)
                    else:
                        shape += (level2num(level, dom_type),)
                    shape += () if cod_type == CodType.SCALAR else (int(cod_type), int(cod_type))
                    coeff = np.arange(np.prod(shape), dtype=float).reshape(shape)
                    shc = SHMat(coeff, dom_type, cod_type, "REAL")

                for level_from in [0, 1, 2]:
                    for ch_show in [None, (), (0,), (0, 0), (0, 0, 0), (0, 0, 0, 0)]:
                        work_well = (ch_show is None) or (len(ch_show) == shc.ndim_chan)
                        try:
                            res_str = shc.tabulate(ch_show, level_show_from=level_from)
                        except Exception as e:
                            if work_well:
                                raise RuntimeError(f"False positive error!: {type(e)}\n{str(e)}")
                        else:
                            if not work_well:
                                raise RuntimeError(f"False negative error!: {ch_show = }\n{shc = }")
                            if ch_show is None:
                                res_none = res_str
                            else:
                                def second_lines(txt: str) -> str:
                                    lines = txt.split()
                                    "\n".join(lines[1:])
                                assert second_lines(res_str) == second_lines(res_none)


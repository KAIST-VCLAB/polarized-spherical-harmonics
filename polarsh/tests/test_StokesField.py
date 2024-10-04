from typing import Sequence
import pytest
import numpy as np
from polarsh import vec2comp, comp2vec, CodType, SHType, SphereGrid, SphereGridCube, ScalarField, StokesField, data_dir
from utils import assert_error


@pytest.fixture(scope='module')
def stkF() -> StokesField:
    return StokesField.from_cubeimage(data_dir/"sponza_64_s%d.exr")
@pytest.fixture(scope='module')
def level() -> int:
    return 5

def assert_operations(stki: StokesField, stkj: StokesField, nones_first: Sequence[None]):
    """
    Test only cases that `stki` is broadcastable to `stkj` in channels.
    Addition, inner products
    """
    Stki = stki.Stk[:,:,:,*nones_first]
    Stki_t = stki.Stk[..., *([None]*stkj.ndim_chan), :]
    Stkj_w_t = stkj.Stk_weighted()[:,:,:,*([None]*stki.ndim_chan)]

    # ========== Addition ==========
    if False:
        # Not implemented yet
        stk_sum = stki.chan + stkj.chan
        assert stk_sum.shape_chan == stkj.shape_chan
        assert np.allclose(stk_sum.Stk, Stki, stkj.Stk)

    # ========== Inner product ==========
    inner = stki.inner(stkj)
    assert inner.shape == stkj.shape_chan
    assert np.allclose(inner, (Stki*stkj.Stk_weighted()).sum(stki.axes_grid + (-1,)))

    inner_tdot = stki.inner(stkj, True)
    assert inner_tdot.shape == stki.shape_chan + stkj.shape_chan
    assert np.allclose(inner_tdot, (Stki_t*Stkj_w_t).sum(stki.axes_grid + (-1,)))
    
    for i in np.ndindex(stki.shape_chan):
        for j in np.ndindex(stkj.shape_chan):
            assert np.allclose(inner_tdot[*i, *j], stki.chan[i].inner(stkj.chan[j]))

    inner_tdot = stki.inner(stkj, True, False)
    assert inner_tdot.shape == stki.shape_chan + stkj.shape_chan + (int(stki.cod_type),)
    assert np.allclose(inner_tdot, (Stki_t*Stkj_w_t).sum(stki.axes_grid))

    for i in np.ndindex(stki.shape_chan):
        for j in np.ndindex(stkj.shape_chan):
            assert np.allclose(inner_tdot[*i, *j], stki.chan[i].inner(stkj.chan[j], sum_polar=False))

    # ========== Complex inner product ==========
    if stki.cod_type == CodType.POLAR2:
        inner_comp = stki.inner_comp(stkj)
        assert inner_comp.shape == stkj.shape_chan

        Stki_comp = vec2comp(Stki)
        Stkj_comp = vec2comp(stkj.Stk_weighted())
        assert np.allclose(inner_comp, (Stki_comp.conj()*Stkj_comp).sum(stki.axes_grid))

        inner_comp_tdot = stki.inner_comp(stkj, True)
        assert inner_comp_tdot.shape == stki.shape_chan + stkj.shape_chan

        Stki_comp = vec2comp(Stki_t)
        Stkj_comp = vec2comp(Stkj_w_t)
        assert np.allclose(inner_comp_tdot, (Stki_comp.conj()*Stkj_comp).sum(stki.axes_grid))
        
        for i in np.ndindex(stki.shape_chan):
            for j in np.ndindex(stkj.shape_chan):
                assert np.allclose(inner_comp_tdot[*i, *j], stki.chan[i].inner_comp(stkj.chan[j]))
    else:
        assert_error(stki.inner_comp)(stkj)


def test_StokesField_operation():
    # ========== Construct objects ==========
    sphG = SphereGridCube(4)
    assert sphG.shape == (6, 4, 4)
    temp = np.arange(np.prod(sphG.shape), dtype=float).reshape(sphG.shape); temp *= 0.01
    for cod_type in list(CodType)[1:]:
        p = int(cod_type)
        def gen_StokesField(shape_chan):
            shape = sphG.shape + shape_chan + (p,)
            Stk = np.arange(np.prod(shape), dtype=float).reshape(shape)
            Stk += np.expand_dims(temp, tuple(range(sphG.ndim, len(shape))))
            return StokesField(sphG.CubemapFrameField(), Stk)
    
        stk0 = gen_StokesField(())
        stk1 = gen_StokesField((3,))
        stk2 = gen_StokesField((9,))
        stk3 = gen_StokesField((5, 3))
        stk_list = [stk0, stk1, stk2, stk3]

        # ========== Arithmetic ==========
        for i, stki in enumerate(stk_list):
            for stkj in stk_list[i+1:]:
                @assert_error
                def add():
                    return stki + stkj
                add()
        
        assert_operations(stk0, stk1, (None,))
        assert_operations(stk0, stk2, (None,))
        assert_operations(stk0, stk3, (None, None))
        assert_operations(stk1, stk3, (None,))
    

def test_StokesField_to_SHVec(stkF: StokesField, level: int):
    shv = stkF.SHCoeff(level, "REAL") # [*c, N, 4]

    # ========== Validate `shv` ==========
    sphG = stkF.SphGrid    
    SH_upto = ScalarField.from_SH_upto(level, sphG, "REAL") # [*g, N]
    s2SH_upto = StokesField.from_s2SH_upto(level, sphG) # [*g, N, 2]
    PSH_upto = StokesField.from_PSH_upto(level, sphG) # [*g, N, 4, 4]

    # s0, s3
    scalF = ScalarField(sphG, stkF.Stk[..., stkF.cidx_scalar()]) # [*g, *c, 2]
    shv_scalar = scalF.SHCoeff(level, "REAL") # [*c, 2, N]
    assert np.allclose(shv.coeff[..., [0, 3]], np.moveaxis(shv_scalar.coeff, -2, -1))
    inner = SH_upto.inner(scalF, chan_tdot=True) # [N, *c, 2]
    assert np.allclose(shv.coeff[..., [0, 3]], np.moveaxis(inner, 0, -2))

    # s1, s2
    stkFp2 = stkF.s12_StokesField() # [*g, *c, 2]
    inner = s2SH_upto.inner_comp(stkFp2, chan_tdot=True) # [N, *c] complex
    assert np.allclose(shv.coeff[..., 1:3], comp2vec(np.moveaxis(inner, 0, -1)))

    # Total, using PSH
    inner = PSH_upto.inner(stkF, chan_tdot=True) # [N, 4, *c]
    assert np.allclose(shv.coeff, np.moveaxis(inner, (0,1), (-2,-1)))
    assert np.allclose(shv.coeff, stkF.inner(PSH_upto, chan_tdot=True))

if __name__ == "__main__":
    test_StokesField_operation()
    # test_StokesField_to_SHVec(level(), stkF())
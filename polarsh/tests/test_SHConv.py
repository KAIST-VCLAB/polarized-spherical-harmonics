import pytest
from typing import List, Tuple, Union
import numpy as np
import polarsh as psh

@pytest.fixture(scope="module")
def level() -> int:
    return 5

@pytest.fixture(scope="module")
def shv_llist(level: int) -> List[List[psh.SHVec]]:
    N = psh.level2num(level, "UNI")
    res = []
    for cod_type in psh.CodType:
        coeff = np.arange(N) * 1.4142
        if cod_type >= psh.CodType.POLAR2:
            coeff = coeff[..., None] + np.arange(int(cod_type)) * 1.713
        res.append([psh.SHVec(coeff, cod_type, "REAL"),
                    psh.SHVec(coeff[::-1], cod_type, "REAL")])
    return res

@pytest.fixture(scope="module")
def shm_list(level: int) -> List[psh.SHMat]:
    res = []
    sh_type = psh.SHType.REAL
    cod_type = psh.CodType.POLAR4
    N = psh.level2num(level, "UNI")
    mcoeff = np.arange(N*N*int(cod_type)*int(cod_type))+0.1
    mcoeff = mcoeff.reshape(N, N, int(cod_type), int(cod_type))
    
    ch_offset = 0.01*np.arange(3)
    mcoeffs = [mcoeff, mcoeff + np.expand_dims(ch_offset, tuple(np.arange(mcoeff.ndim)+1))]
    for mc in mcoeffs:
        res.append(psh.SHMat(mc, "UNI", cod_type, sh_type))
    return res


def gen_kernel(shape_chan: Tuple[int], cod_type: psh.CodType):
    cod_val = np.arange(16).reshape(4, 4) * 1.732
    def gaussian(theta: np.ndarray) -> np.ndarray:
        sigma, mu = 0.05, 0.0
        return  1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (theta - mu)**2 / (2 * sigma**2))
    def kernel(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta)
        res = gaussian(theta)
        axes = tuple(np.arange(len(shape_chan))+theta.ndim)
        res = np.expand_dims(res, axes) + \
                (np.arange(np.prod(shape_chan)).reshape(shape_chan)*1.414)
        res = res[..., None, None] + cod_val

        diff_0 = np.abs(theta)
        diff_pi = np.abs(theta-np.pi)
        axes = axes + (theta.ndim + len(shape_chan), theta.ndim + len(shape_chan)+1)
        damp_0 = np.expand_dims(np.where(diff_0 < 0.1, diff_0/0.1, 1), axes)
        damp_pi = np.expand_dims(np.where(diff_pi < 0.1, diff_pi/0.1, 1), axes)
        temp = damp_0 * damp_pi
        res[..., 0::3, 1:3] *= temp
        res[..., 1:3, 0::3] *= temp
        iso, conj = psh.mat2comppair(res[..., 1:3, 1:3])
        res[..., 1:3, 1:3] = psh.comp2mat(iso)*damp_pi + (psh.comp2mat(conj)*damp_0) @ psh.J_conj
        
        s_ = [0, np.s_[1:3], np.s_[:3], np.s_[:]][int(cod_type) - 1]
        return res[..., s_, s_]
    return kernel


def test_SHConv(level: int, shm_list: List[psh.SHMat], shv_llist: List[List[psh.SHVec]]):
    print("# Test `CodType` consistency for `ScalarConvKernel.SHCoeff` and `StokesConvKernel.SHCoeff`")
    for shape_chan in [(), (1,), (3,), (2, 3, 4)]:
        kernels: List[Union[psh.ScalarConvKernel, psh.StokesConvKernel]] = []
        for cod_type in psh.CodType:
            kernel_func = gen_kernel(shape_chan, cod_type)
            if cod_type==psh.CodType.SCALAR:
                kernels.append(psh.ScalarConvKernel(kernel_func))
            else:
                kernels.append(psh.StokesConvKernel(kernel_func))
        shc4 = kernels[-1].SHCoeff(level)
        for cod_type, kernel in zip(list(psh.CodType)[:-1], kernels[:-1]):
            shc_left = kernel.SHCoeff(level)
            shc_right = shc4.cut(cod_type=cod_type)
            assert shc_left.allclose(shc_right)
 

    first = True
    level_low = 3
    for shm in shm_list:
        for weighted in [False, True]:
            shc = shm.to_SHConv(weighted)
            for cod_type in psh.CodType:
                print("# Test consistency between `SHMat.cut` and `SHConv.cut`") if first else 0
                shc_left = shc.cut(level_low, cod_type)
                shc_right = shm.cut(level_low, cod_type).to_SHConv(weighted)
                assert shc_left.allclose(shc_right)

                shc_left = shc.cut(cod_type=cod_type)
                shc_right = shm.cut(cod_type=cod_type).to_SHConv(weighted)
                assert shc_left.allclose(shc_right)


                print("# Test consistency of `dom_type` for `SHConv.to_SHMat`") if first else 0
                shm_dlist = [shc_left.to_SHMat(dom_type) for dom_type in psh.DomType
                             if dom_type != psh.DomType.ISOBI or cod_type == psh.CodType.SCALAR]
                for i, shm1 in enumerate(shm_dlist):
                    for shm2 in shm_dlist[:i]:
                        assert shm1.allclose(shm2.to_domtype(shm1.dom_type))
                        assert shm2.allclose(shm1.to_domtype(shm2.dom_type))

                print("# Test consistency between `SHConv.to_Mat` and `SHConv.from_SHMat`") if first else 0
                for shm1 in shm_dlist:
                    assert shc_left.allclose(shm1.to_SHConv(weighted))

                print("Test `SHConv @ SHVec` and `SHConv.to_SHMat() @ SHVec`") if first else 0
                for shv in shv_llist[int(cod_type)-1]:
                    shv_left = shc_left @ shv
                    for shm1 in shm_dlist:
                        shv_right = shm1 @ shv
                        assert shv_left.allclose(shv_right)
                
                if False: # Currently wrong. It needs to substitute some coefficients to zero
                    print("Test `SHConv.cut() @ SHVec` and `(SHConv @ SHVec).cut()`")
                    for shv4 in shv_llist[-1]:
                        shv_left = shc_left @ shv4.cut(cod_type=cod_type)
                        shv_right = (shc @ shv4).cut(cod_type=cod_type)
                        assert shv_left.allclose(shv_right)
                first = False

if __name__ == "__main__" and True:
    level = level()
    test_SHConv(shm_list(level), shv_llist(level))
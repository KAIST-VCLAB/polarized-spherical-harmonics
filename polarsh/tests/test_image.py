import numpy as np
import tempfile
from pathlib import Path
import polarsh as psh

def test_imread_imwrite():
    img_exr = psh.imread(psh.data_dir/"sponza_64_s0.exr")

    print("# `imread('*.exr')` read an float image")
    h, w = 64*3, 64*4
    assert (img_exr.shape, img_exr.dtype.kind) == ((h, w, 3), 'f')
    assert np.allclose(img_exr.mean(), 0.17024781)

    with tempfile.TemporaryDirectory() as tempdir:
        root = Path(tempdir)

        print("# `imread()`-`imwrite()` consistency for EXR")
        tempexr = root/"temp.exr"
        psh.imwrite(tempexr, img_exr)
        img_exr2 = psh.imread(tempexr)
        assert np.allclose(img_exr, img_exr2)


        print("# `imread()`-`imwrite()` consistency for PNG (RGB) under conversion")
        temppng = root/"temp.png"
        psh.imwrite(temppng, img_exr)
        img_uint_rw = psh.imread(temppng)
        img_uint_manual = (np.clip(img_exr, 0, 1) ** (1/2.2) * 255).astype(np.uint8)
        img_uint_manual = (img_uint_manual / 255) ** 2.2
        assert np.allclose(img_uint_rw, img_uint_manual, atol=1e-6)

        psh.imwrite(temppng, img_uint_rw)
        assert np.allclose(img_uint_rw, psh.imread(temppng), atol=1e-2)


        print("# Applying a colormap yield an alpha channel")
        img_gray = psh.rgb2gray(img_exr)
        assert img_gray.shape == (h, w)

        img_cmap = psh.pltcmap_diverge(img_gray)
        assert img_cmap.shape == (h, w, 4)
        assert img_cmap.max() <= 1.0

        print("# `imread()`-`imwrite()` consistency for PNG (RGBA)")
        uint_manual = np.clip(img_cmap*255, 0, 255).astype(np.uint8) / 255
        psh.imwrite(temppng, img_cmap, gm_png=False)
        uint_rw = psh.imread(temppng, gm_png=False)
        assert np.allclose(uint_rw, uint_manual[:,:,:3])
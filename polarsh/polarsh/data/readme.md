# Data Sources

## Polarized Environment Map

### Files

In the following format, `{edge}` is either 64 or 1024, and  `{s}` is 0, 1, 2, or 3.

* `sponza_{edge}_s{s}.exr`
* `cathedral_{edge}_s{s}.exr`

### Data Source and Processing

Created by modifying 3D models provided in [McGuire Computer Graphics Archive](https://casual-effects.com/data/) and rendering them using a polarized variant of [Mitsuba 3](https://mitsuba-renderer.org).

Modifications made to the original scenes:

* Changed some materials of the 3D models to polarization-aware ones in Mitsuba 3.
* Added linear polarizing filters in front of each light source to enhance the visibility of polarized components more.


# Data-based pBRDF

### Files

* `pbrdf_table_info_dsamp5.mat`
* `6_gold_pbrdf_dsamp5.mat`

### 출처

Subsampled at intervals of 5 in the angular resolution (Rusinkiewicz coordinates $\phi_d, \theta_d,\theta_h$) from [KAIST Dataset of polarimetric BRDF](https://vclab.kaist.ac.kr/siggraph2020/pbrdfdataset/kaistdataset.html).

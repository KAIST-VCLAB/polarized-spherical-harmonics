this software is specifially designed to implement polarized spherical harmonics, which are basis functions for the directional intensity of polarized light. However, it is also a valuable tool for users who are primarily interested in the background knowledge of spherical harmonics themselves, or in handling spherical data of polarized light (such as polarized environment light) regardless of basis functions.

For users with partial interest, we intended that they could read only some of the tutorials in this folder.

First, `0. quickstart` demonstrate a quick overview of our main contribution in polarized spherical harmonics without requiring readers to have read other tutorials. More detailed explanations for each concept and usage of each API are given in Tutorials 1 to 6. The prerequisites for each tutorial and relevant sections in our paper are specified as following.

## Content

Tutorial 0. [Quickstart](./0. quickstart.ipynb)

Tutorial 1. [Spherical Functions](1_spherical_functions.ipynb)

* M-Sec. 4.1, S-Sec. 2

Tutorial 2. [Spherical Harmonics](2_spherical_harmonics.ipynb) 

* Prerequisite: Tutorial 1.
* M-Sec. 4.1, S-Sec. 2

Tutorial 3. [Polarization and Mueller calculus](3_polarization.ipynb)

* M-Sec. 4.2, S-Sec. 3

Tutorial 4. [Stokes Vector Fields](4_Stokes_vector_fields.ipynb)

* Prerequisite: Tutorial 1.
* M-Sec. 5, S-Sec. 4

Tutorial 5. [Polarized Spherical Harmonics](5_polarized_spherical_harmonics.ipynb) (1, 2, 3, 4)

* Prerequisite: Tutorials 1-4
* M-Sec. 6.1-6.3, S-Sec. 5.1-5.7

Tutorial 6. [Polarized Spherical Convolution](6_polarized_spherical_convolution.ipynb) (1,2, 3, 4, 5)

* Prerequisite: Tutorials 1-5
* M-Sec. 6.4, S-Sec. 5.8-5.9

In other words, a reader who only want to study the background of spherical harmonics can read only Tutorials 1 and 2. Similarly, a reader who is interested only in the basiscs of polarization can read just Tutorial 3.


Whithin the tutorials, the prefix "M-" in references such as "M-Eq. (1)" or "S-Def. 2.1" refers to our main paper [Yi et al. 2024], while "S-" refers to the supplemental document.

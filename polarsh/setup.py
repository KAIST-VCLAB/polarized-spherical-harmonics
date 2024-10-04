from setuptools import setup, find_packages

setup(
    name='polarsh',
    version='0.1.0',
    description='Polarized spherical harmonics',
    url="https://github.com/KAIST-VCLAB/polarized-spherical-harmonics",
    license="BSD",
    author='Shinyoung Yi',
    author_email='syyi.graphics@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy', 'scipy', 'spherical',
    ],
    package_data={'polarsh': [
        'resource/*',
        'data/*'
    ]},
    include_package_data=True,
    zip_safe=False
)
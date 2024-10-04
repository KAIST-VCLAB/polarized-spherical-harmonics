import pytest
import numpy as np
from polarsh import data_dir, SphereGrid, ScalarField

@pytest.fixture()
def scalF():
    # TODO
    return None
    return ScalarField(SphereGrid.f)

def test_RC(scalF):
    # TODO
    pass
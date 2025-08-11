from tempfile import NamedTemporaryFile
import pytest
from ligo.skymap.io.fits import write_sky_map
import numpy as np

from amplfi.utils import skymap


@pytest.fixture(params=[32, 64])
def n_side(request):
    return request.param


@pytest.fixture(params=[10000, 20000])
def n_samples(request):
    return request.param


def test_histogram_skymap(n_side, n_samples):
    # mock posterior samples
    ra_samples = np.random.uniform(0, 2 * np.pi, n_samples)
    dec_samples = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi, n_samples)
    dist_samples = np.random.uniform(1000, 2000, n_samples)
    # check skymap format with no extra metadata
    skymap_content = skymap.histogram_skymap(
        ra_samples, dec_samples, dist_samples, nside=n_side
    )
    # check FITS format
    with NamedTemporaryFile(mode="w+", suffix=".multiorder.fits") as fits_file:
        write_sky_map(fits_file.name, skymap_content)
        from astropy.table import QTable

        t = QTable.read(fits_file.name)

    # do basic checks on the format
    assert t.meta["PIXTYPE"] == "HEALPIX"
    assert t.meta["ORDERING"] == "NUNIQ"

    # check if extra metadata is correctly added
    skymap_content = skymap.adaptive_histogram_skymap(
        ra_samples,
        dec_samples,
        dist_samples,
        dist_nside=n_side,
        metadata={"INSTRUME": "H1,L1,V1"},
    )
    with NamedTemporaryFile(mode="w+", suffix=".multiorder.fits") as fits_file:
        write_sky_map(fits_file.name, skymap_content)

        t = QTable.read(fits_file.name)

    # do basic checks on the format
    assert t.meta["PIXTYPE"] == "HEALPIX"
    assert t.meta["ORDERING"] == "NUNIQ"
    assert t.meta["INSTRUME"] == "H1,L1,V1"

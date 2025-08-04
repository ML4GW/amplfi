from importlib.metadata import version, PackageNotFoundError
from collections import OrderedDict

import healpy as hp
from typing import Optional
import numpy as np
from astropy.table import Table
from astropy import units as u

from . import distance

_PROGRAM_NAME = "amplfi"
try:
    _PROGRAM_VERSION = version(_PROGRAM_NAME)
except PackageNotFoundError:
    _PROGRAM_VERSION = None

_DEFAULT_METADATA = OrderedDict(
    {
        "PIXTYPE": "HEALPIX",
        "ORDERING": "NUNIQ",
        "COORDSYS": "C",
        "INSTRUME": None,
        "CREATOR": _PROGRAM_NAME,
        "DISTMEAN": None,
        "DISTSTD": None,
        "VCSVERS": _PROGRAM_VERSION,
        "MOCORDER": None,
        "INDXSCHM": "EXPLICIT",
    }
)


def nest2uniq(nside: int, ipix: int):
    return 4 * nside * nside + ipix


def histogram_skymap(
    ra: np.ndarray,
    dec: np.ndarray,
    dist: Optional[np.ndarray] = None,
    nside: int = 32,
    min_samples_per_pix: int = 5,
    metadata: Optional[dict] = None,
) -> Table:
    """Given right ascension declination samples
    and optionally distance samples,
    calculate a HEALPix histogram skymap.

    Args:
        ra:
            Samples of right-ascension like parameter between 0 and 2pi.
            Samples outside this range will
            raise an error with the HEALPix library.
        dec:
            Declination samples between -pi/2 and pi/2.
            Samples outside this range will
            raise an error with the HEALPix library.
        dist:
            Distance samples in Mpc. If provided, will calculate distance
            ansatz parameters, `DISTMU`, `DISTSIGMA`, `DISTNORM`
            for each pixel containing more than `min_samples_per_pix`.
            If not provided, will use default values of
            `np.inf`, `1 Mpc`, and `0 / Mpc^2` respectively.
        nside:
            HEALPix nside parameter
        min_samples_per_pix:
            Minimum number of samples per pixel to
            calculate distance ansatz parameters.
            Otherwise, the default values are used.
        metadata:
            Extra metadata for the skymap header.

    Returns:
        astropy.table.Table: HEALPix histogram skymap
    """

    # convert declination to theta between 0 and pi
    theta = np.pi / 2 - dec

    num_samples = len(ra)

    # calculate number of samples in each pixel
    npix = hp.nside2npix(nside)
    order = hp.pixelfunc.nside2order(nside)
    ipix = hp.ang2pix(nside, theta, ra, nest=True)
    uniq, counts = np.unique(ipix, return_counts=True)

    # create empty map and then fill in non-zero pix with density
    # estimated by fraction of total samples in each pixel
    map = np.zeros(npix)
    map[np.in1d(range(npix), uniq)] = counts
    density = map / num_samples
    density /= hp.nside2pixarea(nside)
    density /= u.sr

    uniq_ipix = nest2uniq(nside, np.arange(npix))

    # convert to astropy table
    table = Table(
        [uniq_ipix, density],
        names=["UNIQ", "PROBDENSITY"],
        copy=False,
    )

    # defaults for distance parameters
    mu = np.ones(npix) * np.inf
    sigma = np.ones(npix)
    norm = np.zeros(npix)

    default_metadata = _DEFAULT_METADATA.copy()
    default_metadata.update({"MOCORDER": order})
    if metadata:
        default_metadata.update(metadata)

    if dist is not None:
        default_metadata["DISTMEAN"] = np.mean(dist)
        default_metadata["DISTSTD"] = np.std(dist)
        # compute distance ansatz for pixels containing
        # greater than a threshold number
        good_ipix = uniq[counts > min_samples_per_pix]
        dist_mu = []
        dist_sigma = []
        dist_norm = []
        for _ipix in good_ipix:
            _distance = dist[ipix == _ipix]
            _, _m, _s = distance.moments_from_samples_impl(_distance)
            _mu, _sigma, _norm = distance.ansatz_impl(_s, _m)
            dist_mu.append(_mu)
            dist_sigma.append(_sigma)
            dist_norm.append(_norm)

        mu[np.in1d(range(npix), good_ipix)] = np.array(dist_mu)
        sigma[np.in1d(range(npix), good_ipix)] = np.array(dist_sigma)
        norm[np.in1d(range(npix), good_ipix)] = np.array(dist_norm)

    mu *= u.Mpc
    sigma *= u.Mpc
    norm /= u.Mpc**2

    # add distance parameters to table
    table.add_columns(
        [mu, sigma, norm], names=["DISTMU", "DISTSIGMA", "DISTNORM"]
    )
    table.meta = default_metadata
    # convert to 32-bit precision
    table["PROBDENSITY"] = table["PROBDENSITY"].astype(np.float32)
    table["DISTMU"] = table["DISTMU"].astype(np.float32)
    table["DISTSIGMA"] = table["DISTSIGMA"].astype(np.float32)
    table["DISTNORM"] = table["DISTNORM"].astype(np.float32)
    table["UNIQ"] = table["UNIQ"].astype(np.int32)
    return table


def calculate_searched_area(
    healpix_map: np.ndarray,
    ra_true: float,
    dec_true: float,
    nside: int = 32,
):
    """Calculate the searched area for a given HEALPix map,
    true right ascension and declination. Also calculate estimates
    of 50% and 90% credible regions

    Args:
        healpix_map:
            HEALPix map
        ra_true:
            True right ascension in radians
        dec_true:
            True declination in radians
        nside:
            HEALPix nside parameter
    """
    theta_true = np.pi / 2 - dec_true
    true_ipix = hp.ang2pix(nside, theta_true, ra_true, nest=True)

    # sort pixels in descending order
    # count number of pixels before hitting the pixel with injection
    # in the sorted array
    sorted_idxs = np.argsort(healpix_map)[::-1]
    num_pix_before_injection = 1 + np.argmax(sorted_idxs == true_ipix)
    searched_area = num_pix_before_injection * hp.nside2pixarea(
        nside, degrees=True
    )
    healpix_cumsum = np.cumsum(healpix_map[sorted_idxs])
    fifty = np.argmin(healpix_cumsum < 0.5) * hp.nside2pixarea(
        nside, degrees=True
    )
    ninety = np.argmin(healpix_cumsum < 0.9) * hp.nside2pixarea(
        nside, degrees=True
    )
    return searched_area, fifty, ninety

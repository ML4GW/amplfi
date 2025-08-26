from importlib.metadata import version, PackageNotFoundError
from collections import OrderedDict
from typing import TYPE_CHECKING
import healpy as hp
from typing import Optional
import numpy as np
from astropy.table import Table
from astropy import units as u
from ligo.skymap.healpix_tree import adaptive_healpix_histogram
from ligo.skymap.bayestar import derasterize
from . import distance
import matplotlib.pyplot as plt
import ligo.skymap.plot  # noqa
from astropy.coordinates import SkyCoord

if TYPE_CHECKING:
    from pathlib import Path


_PROGRAM_NAME = "amplfi"
try:
    _PROGRAM_VERSION = version(_PROGRAM_NAME)
except PackageNotFoundError:
    _PROGRAM_VERSION = None

_DEFAULT_METADATA = OrderedDict(
    {
        "DISTMEAN": None,
        "DISTSTD": None,
        "VCSVERS": _PROGRAM_VERSION,
    }
)


def nest2uniq(nside: int, ipix: int):
    return 4 * nside * nside + ipix


def adaptive_histogram_skymap(
    ra: np.ndarray,
    dec: np.ndarray,
    dist: Optional[np.ndarray] = None,
    max_nside: int = 2048,
    dist_nside: int = 64,
    max_samples_per_pixel: int = 20,
    min_samples_per_pix_dist: int = 5,
    metadata: Optional[dict] = None,
) -> Table:
    """Given right ascension declination samples
    and optionally distance samples,
    calculate a HEALPix adaptive histogram skymap
    using `ligo.skymap.healpix_tree.adaptive_histogram_skymap`

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
        max_nside:
            Maximum HEALPix nside parameter for adaptive histogram
        dist_nside:
            Nside value to resample to after histogramming for distance
            ansatz estimation
        max_samples_per_pix:
            Max samples per pixel when performing
            adaptive histogramming
        min_samples_per_pix_dist:
            Minimum number of samples per pixel to
            calculate distance ansatz parameters.
            Otherwise, the default values are used.
        metadata:
            Extra metadata for the skymap header.

    Returns:
        astropy.table.Table: HEALPix histogram skymap
    """
    _metadata = _DEFAULT_METADATA.copy()

    dist_nside = dist_nside if dist is not None else -1
    # convert declination to theta (between 0 and pi)
    theta = np.pi / 2 - dec
    prob = adaptive_healpix_histogram(
        theta,
        ra,
        max_samples_per_pixel=max_samples_per_pixel,
        max_nside=max_nside,
        nside=dist_nside,
        nest=True,
    )
    skymap = Table({"PROB": prob})
    npix = len(skymap)

    # get corresponding nside of the
    # adaptive histogram skymap
    nside = hp.npix2nside(npix)

    # determine the nested pixel idx
    # for each posterior sample
    nested_ipix = hp.ang2pix(nside, theta, ra, nest=True)
    unique_ipix, counts = np.unique(nested_ipix, return_counts=True)

    # estimate distance ansatz parameters
    mu = np.ones(npix) * np.inf
    sigma = np.ones(npix)
    norm = np.zeros(npix)

    if dist is not None:
        _metadata["DISTMEAN"] = np.mean(dist)
        _metadata["DISTSTD"] = np.std(dist)

        # compute distance ansatz for pixels containing
        # greater than a threshold number
        good_ipix = unique_ipix[counts > min_samples_per_pix_dist]
        dist_mu = []
        dist_sigma = []
        dist_norm = []
        for _ipix in good_ipix:
            _distance = dist[nested_ipix == _ipix]
            _, _m, _s = distance.moments_from_samples_impl(_distance)
            _mu, _sigma, _norm = distance.ansatz_impl(_s, _m)
            dist_mu.append(_mu)
            dist_sigma.append(_sigma)
            dist_norm.append(_norm)

        mu[np.isin(range(npix), good_ipix)] = np.array(dist_mu)
        sigma[np.isin(range(npix), good_ipix)] = np.array(dist_sigma)
        norm[np.isin(range(npix), good_ipix)] = np.array(dist_norm)

    mu *= u.Mpc
    sigma *= u.Mpc
    norm /= u.Mpc**2

    # add distance parameters to table
    skymap.add_columns(
        [mu, sigma, norm], names=["DISTMU", "DISTSIGMA", "DISTNORM"]
    )

    if metadata:
        _metadata.update(metadata)
    skymap.meta = _metadata

    # finally derasterize to multiorder
    skymap = derasterize(skymap)
    return skymap


def histogram_skymap(
    ra: np.ndarray,
    dec: np.ndarray,
    dist: Optional[np.ndarray] = None,
    nside: int = 32,
    min_samples_per_pix_dist: int = 5,
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
        good_ipix = uniq[counts > min_samples_per_pix_dist]
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
    columns = ["PROBDENSITY", "DISTMU", "DISTSIGMA", "DISTNORM"]
    for column in columns:
        table[column] = table[column].astype(np.float32)
    return table


def plot_skymap(skymap: Table, ra_inj: float, dec_inj: float, outpath: "Path"):
    fig = plt.figure()
    ax = fig.add_subplot(projection="astro mollweide")
    ax.imshow_hpx(
        (skymap, "ICRS"), vmin=0, order="nearest-neighbor", cmap="cylon"
    )
    ax.plot_coord(
        SkyCoord(ra_inj, dec_inj, unit=u.rad),
        "x",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=10,
    )
    plt.savefig(outpath)
    plt.close()

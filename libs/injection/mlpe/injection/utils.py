from typing import Union

import numpy as np
from astropy.time import Time


def ra_from_phi(phi: Union[np.ndarray, float], gpstime: float) -> float:
    """
    Calculate the right ascension of a source given its relative
    azimuthal angle and the geocentric time of the observation
    """

    # get the sidereal time at the observation time
    t = Time(gpstime, format="gps", scale="utc")
    gmst = t.sidereal_time("mean", "greenwich").to("rad").value

    phi = np.array([phi])

    # convert phi from range [-pi, pi] to [0, 2pi]
    mask = phi < 0
    phi[mask] += 2 * np.pi

    return (phi + gmst) % (2 * np.pi)


def phi_from_ra(ra: Union[np.ndarray, float], gpstime: float) -> float:

    # get the sidereal time at the observation time
    t = Time(gpstime, format="gps", scale="utc")
    gmst = t.sidereal_time("mean", "greenwich").to("rad").value

    if isinstance(ra, float):
        ra = np.array([ra])

    # calculate the relative azimuthal angle in the range [0, 2pi]
    phi = ra - gmst
    mask = phi < 0
    phi[mask] += 2 * np.pi

    # convert phi from range [0, 2pi] to [-pi, pi]
    mask = phi > np.pi
    phi[mask] -= 2 * np.pi

    return phi

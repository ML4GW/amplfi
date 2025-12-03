import torch
from ml4gw.gw import get_ifo_geometry, compute_antenna_responses

def rescaled_distance_to_distance(M0=1.0, **parameters):
    """
    Rescale Distance only with Chirp Mass
    
    Args:
        chirp_mass:
            Chirp Mass
        rescaled_distance:
            Uniform distribution of distances
        M0:
            reference Chirp Mass (deafualt = 1 Mo)
            
    returns:
        distance:
            luminosity distance
    """

    chirp_mass = parameters["chirp_mass"]
    rescaled_distance = parameters["rescaled_distance"]
    return(chirp_mass / M0) ** (5/6) * rescaled_distance


def chirp_distance_to_distance(M0, ifos, **parameters):
    """
    Rescale Distance only with Chirp Mass
    
    Args:
        chirp_mass:
            Chirp Mass
        rescaled_distance:
            Uniform distribution of distances
        ifos:
            List of interferometers
        dec:
            Declination of each source in radians relative
            to the celestial north
        psi:
            Angle in radians between each source's
            natural polarization basis and the basis
            which has the 0th unit vector pointing along
            the celestial equator
        phi:
            Angle in radians between each source's right
            ascension and the right ascension of the
            geocenter
        M0:
            reference Chirp Mass (deafualt = 1 Mo)
        inclination:
            Source inclination
            
    returns:
        distance:
            luminosity distance
    """

    detector_tensors, _ = get_ifo_geometry(*ifos)

    dec = parameters["dec"]
    phi = parameters["phi"]
    psi = parameters["psi"]
    inclination = parameters["inclination"]
    chirp_mass = parameters["chirp_mass"]
    chirp_distance = parameters["chirp_distance"]

    theta = torch.pi / 2 - dec
    antenna_responses = compute_antenna_responses(
        theta, psi, phi, detector_tensors, ['plus', 'cross']
        )

    f_plus, f_cross = antenna_responses[:, 0], antenna_responses[:, 1]

    distances = []

    for i, ifo in enumerate(ifos):
        factor = (f_plus[:, i]*(1+torch.cos(inclination)**2))**2 + (f_cross[:, i]*torch.cos(inclination)**2)**2
        rescaled_distance = chirp_distance/torch.sqrt(factor)
        distances.append(rescaled_distance)

    distances = torch.vstack(distances)
    distances, _ = torch.max(distances, dim=0)
    parameters["rescaled_distance"] = distances
    del parameters["chirp_distance"]
    
    return rescaled_distance_to_distance(M0, **parameters)
    
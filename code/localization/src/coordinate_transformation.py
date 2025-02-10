"""
This script provides coordinate transformations from Geodetic -> ECEF,
ECEF -> ENU and Geodetic -> ENU
(the composition of the two previous functions).
Running the script by itself runs tests.
based on https://gist.github.com/govert/1b373696c9a27ff4c72a.
A good source to read up on the different reference frames is:
http://dirsig.cis.rit.edu/docs/new/coordinates.html
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

# from tf.transformations import euler_from_quaternion


a = 6378137  # EARTH_RADIUS_EQUA in Pylot, used in geodetic_to_enu
b = 6356752.3142
f = (a - b) / a
e_sq = f * (2 - f)
alt_offset = 331.00000
STD_LAT = 0.0
STD_LON = 0.0
STD_H = 0.0


class CoordinateTransformer:
    """Object that is used to transform Coordinates between
    xyz and gnss reference frame"""

    la_ref = STD_LAT
    ln_ref = STD_LON
    h_ref = STD_H
    ref_set = False

    def __init__(self):
        pass

    def gnss_to_xyz(self, lat, lon, h):
        return geodetic_to_enu(lat, lon, h)


def geodetic_to_enu(lat, lon, alt):
    """
    Method from pylot project to calculate coordinates
    https://github.com/erdos-project/pylot/blob/master/pylot/utils.py#L470

    Args:
        lat (float): latitude
        lon (float): longitude
        alt (float: altitude

    Returns:
        x, y, z: coordinates
    """

    scale = math.cos(CoordinateTransformer.la_ref * math.pi / 180.0)
    basex = scale * math.pi * a / 180.0 * CoordinateTransformer.ln_ref
    basey = (
        scale
        * a
        * math.log(math.tan((90.0 + CoordinateTransformer.la_ref) * math.pi / 360.0))
    )

    x = scale * math.pi * a / 180.0 * lon - basex
    y = scale * a * math.log(math.tan((90.0 + lat) * math.pi / 360.0)) - basey

    # Is not necessary in new version
    # y *= -1
    # alt_offset is needed to keep the hight of the map in mind
    # right now we don't really use the altitude anyways
    return x, y, alt + alt_offset


def geodetic_to_ecef(lat, lon, h):
    # (lat, lon) in WSG-84 degrees
    # h in meters
    lamb = math.radians(lat)
    phi = math.radians(lon)
    s = math.sin(lamb)
    N = a / math.sqrt(1 - e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x = (h + N) * cos_lambda * cos_phi
    y = (h + N) * cos_lambda * sin_phi
    z = (h + (1 - e_sq) * N) * sin_lambda

    return x, y, z


def ecef_to_enu(x, y, z, lat0, lon0, h0):
    lamb = math.radians(lat0)
    phi = math.radians(lon0)
    s = math.sin(lamb)
    N = a / math.sqrt(1 - e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    s_phi = math.sin(phi)
    c_phi = math.cos(phi)

    x0 = (h0 + N) * cos_lambda * c_phi
    y0 = (h0 + N) * cos_lambda * s_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda

    xd = x - x0
    yd = y - y0
    zd = z - z0

    xE = -s_phi * xd + c_phi * yd
    yN = -c_phi * sin_lambda * xd - sin_lambda * s_phi * yd + cos_lambda * zd
    zUp = cos_lambda * c_phi * xd + cos_lambda * s_phi * yd + sin_lambda * zd

    return xE, yN, zUp


def quat_to_heading(quaternion):
    """
    Converts a quaternion to a heading of the car in radians
    (see ../../doc/perception/coordinate_transformation.md)
    :param quaternion: quaternion of the car as a list [q.x, q.y, q.z, q.w]
                       where q is the quaternion
    :return: heading of the car in radians (float)
    """
    # Create a Rotation object from the quaternion
    rotation = Rotation.from_quat(quaternion)
    # Convert the Rotation object to a matrix
    rotation_matrix = rotation.as_matrix()
    # calculate the angle around the z-axis (theta) from the matrix
    theta = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # arctan2 returns a theta so that:
    # ---------------------------------------------------------------
    # | 0 = x-axis | pi/2 = y-axis | pi = -x-axis | -pi/2 = -y-axis |
    # ---------------------------------------------------------------
    # heading is positive in counter clockwise rotations

    heading = -theta

    return heading


# old functions
# def quat_to_heading(msg):
#     orientation_q = msg
#     orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z,
#                         orientation_q.w]
#     (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
#     heading = float(math.atan2(pitch, roll))
#     return -heading + math.pi

# if __name__ == '__main__':
#    def are_close(a, b):
#        return abs(a - b) < 1e-4
#
#
#    latLA = 34.00000048
#    lonLA = -117.3335693
#    hLA = 251.702
#
#    x0, y0, z0 = geodetic_to_ecef(latLA, lonLA, hLA)
#    x = x0 + 1
#    y = y0
#    z = z0
#    xEast, yNorth, zUp = ecef_to_enu(x, y, z, latLA, lonLA, hLA)
#    assert are_close(0.88834836, xEast)
#    assert are_close(0.25676467, yNorth)
#    assert are_close(-0.38066927, zUp)
#
#    x = x0
#    y = y0 + 1
#    z = z0
#    xEast, yNorth, zUp = ecef_to_enu(x, y, z, latLA, lonLA, hLA)
#    assert are_close(-0.45917011, xEast)
#    assert are_close(0.49675810, yNorth)
#    assert are_close(-0.73647416, zUp)
#
#    x = x0
#    y = y0
#    z = z0 + 1
#    xEast, yNorth, zUp = ecef_to_enu(x, y, z, latLA, lonLA, hLA)
#    assert are_close(0.00000000, xEast)
#    assert are_close(0.82903757, yNorth)
#    assert are_close(0.55919291, zUp)

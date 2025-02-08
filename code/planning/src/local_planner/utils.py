from scipy.spatial.transform import Rotation
import numpy as np
import math
import carla
import os


"""
This file represents the utility functions for the local planner and other
components.
It containes parameters and utility functions to reduce code in the ros nodes.
"""

# Distance to stop in Intersection, Lanechange, Overtake
TARGET_DISTANCE_TO_STOP = 5
TARGET_DISTANCE_TO_STOP_OVERTAKE = 7
# Number of waypoints to be used for the overtaking maneuver
NUM_WAYPOINTS = 9
# Earth radius in meters for location_to_GPS
EARTH_RADIUS_EQUA = 6378137.0


def get_distance(pos_1, pos_2):
    """Calculate the distance between two positions

    Args:
        pos1 (np.array): Position 1
        pos2 (np.array): Position 2
        # the np.array should be in the form of
        # np.array([data.pose.position.x,
                    data.pose.position.y,
                    data.pose.position.z])

    Returns:
        float: Distance
    """
    return np.linalg.norm(pos_1 - pos_2)


def location_to_gps(lat_ref: float, lon_ref: float, x: float, y: float):
    """Convert world coordinates to (lat,lon,z) coordinates
       Copied from:
       https://github.com/carla-simulator/scenario_runner/blob/master/srunner/tools/route_manipulation.py

    Args:
        lat_ref (float): reference lat value
        lon_ref (float): reference lat value
        x (float): x-Coordinate value
        y (float): y-Coordinate value

    Returns:
        dict: Dictionary with (lat,lon,z) coordinates
    """

    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    my = (
        scale
        * EARTH_RADIUS_EQUA
        * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
    )
    mx += x
    my -= y

    lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUA * scale)
    lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) / math.pi - 90.0
    z = 703

    return {"lat": lat, "lon": lon, "z": z}


def calculate_rule_of_thumb(emergency, speed):
    """Calculates the rule of thumb as approximation
    for the braking distance

    Args:
        emergency (bool): if emergency brake is initiated
        speed (float): speed of the vehicle (km/h)

    Returns:
        float: distance calculated with rule of thumb
    """
    reaction_distance = speed
    braking_distance = (speed * 0.36) ** 2
    if emergency:
        # Emergency brake is really effective in Carla
        return reaction_distance + braking_distance / 2
    else:
        return reaction_distance + braking_distance


def approx_obstacle_pos(
    distance: float, heading: float, ego_pos: np.array, speed: float
):
    """calculate the position of the obstacle in the global coordinate system
        based on ego position, heading and distance

    Args:
        speed (float): Speed of the ego vehicle
        distance (float): Distance to the obstacle
        heading (float): Ego vehivle heading
        ego_pos (np.array): Position in [x, y, z]

    Returns:
        np.array: approximated position of the obstacle
    """
    rotation_matrix = Rotation.from_euler("z", heading)

    # Create distance vector with 0 rotation
    relative_position_local = np.array([distance, 0, 0])

    # Rotate distance vector to match heading
    absolute_position_local = rotation_matrix.apply(relative_position_local)

    # Add ego position vector with distance vetor to get absolute position
    vehicle_position_global_start = ego_pos + absolute_position_local

    # Calculate the front and back of the vehicle
    length = np.array([3, 0, 0])
    length_vector = rotation_matrix.apply(length)

    # calculate the front left corner of the vehicle
    offset = np.array([1, 0, 0])
    rotation_adjusted = Rotation.from_euler("z", heading + math.radians(90))
    offset_front = rotation_adjusted.apply(offset)

    # calculate back right corner of the vehicle
    rotation_adjusted = Rotation.from_euler("z", heading + math.radians(270))
    offset_back = rotation_adjusted.apply(offset)

    vehicle_position_global_end = (
        vehicle_position_global_start + length_vector + offset_back
    )

    return vehicle_position_global_start + offset_front, vehicle_position_global_end


def convert_to_ms(speed: float):
    """Convert km/h to m/s

    Args:
        speed (float): speed in km/h

    Returns:
        float: speed in m/s
    """
    return speed / 3.6


def spawn_car(distance):
    """Only used for testing, spawns a car in the given distance

    Args:
        distance (float): distance
    """
    CARLA_HOST = os.environ.get("CARLA_HOST", "paf-carla-simulator-1")
    CARLA_PORT = int(os.environ.get("CARLA_PORT", "2000"))

    client = carla.Client(CARLA_HOST, CARLA_PORT)

    world = client.get_world()
    world.wait_for_tick()

    blueprint_library = world.get_blueprint_library()
    # bp = blueprint_library.filter('vehicle.*')[0]
    # vehicle = world.spawn_actor(bp, world.get_map().get_spawn_points()[0])
    bp = blueprint_library.filter("model3")[0]
    for actor in world.get_actors():
        if actor.attributes.get("role_name") == "hero":
            ego_vehicle = actor
            break

    spawnPoint = carla.Transform(
        ego_vehicle.get_location() + carla.Location(y=distance.data),
        ego_vehicle.get_transform().rotation,
    )

    vehicle = world.spawn_actor(bp, spawnPoint)
    vehicle.set_autopilot(False)
    # vehicle.set_target_velocity(carla.Vector3D(0, 6, 0))

    # Spawn second vehicle
    # spawnpoint2 = carla.Transform(ego_vehicle.get_location() +
    #                               carla.Location(x=2.5, y=distance.data + 1),
    #                               ego_vehicle.get_transform().rotation)
    # vehicle2 = world.spawn_actor(bp, spawnpoint2)
    # vehicle2.set_autopilot(False)


def interpolate_speed(
    speed_target: float, speed_current: float, lerp_factor: float
) -> float:
    return (1 - lerp_factor) * speed_current + lerp_factor * speed_target


def filter_vision_objects(float_array, oncoming):
    """Filters vision objects to calculate collision check
    It contains the classId, the absolute Euclidean distance
    and 6 coordinates for upper left and lower right corner
    of the bounding box
    This method will be obsolete with the intermediate layer soon but will be used
    for now.

    Array shape: [n, 3] with (class, x,y)

    Args:
        data (ndarray): numpy array with vision objects
    """

    # Reshape array to 3 columns and n rows (one row per object)
    float_array = np.asarray(float_array)
    float_array = np.reshape(float_array, (float_array.size // 3, 3))
    # Filter all rows that contain np.inf
    float_array = float_array[~np.any(np.isinf(float_array), axis=1), :]
    if float_array.size == 0:
        return None
    # Filter out all objects that are not cars, Persons, Bycicles,
    # Motorbikes, Busses or Trucks
    all_cars = float_array[(float_array[:, 0] != 0) & (float_array[:, 0] != 3)]

    # Get cars that are on our lane
    if oncoming:
        cars_in_front = all_cars[
            np.where(np.logical_and(all_cars[:, 2] >= 0.90, all_cars[:, 2] < 1.75))
        ]

    else:
        cars_in_front = all_cars[
            np.where(np.logical_and(all_cars[:, 2] < 0.90, all_cars[:, 2] > -0.90))
        ]

    if cars_in_front.size == 0:
        # no car in front
        return None
    # Return nearest car
    min_object_in_front = cars_in_front[np.argmin(cars_in_front[:, 1])]
    if oncoming:
        # As the overtaking / cruising step does not work properly at the moment this
        # is a workaround.
        if min_object_in_front[1] > 9.0:
            return None

    return min_object_in_front


def convert_pose_to_array(poses: np.ndarray) -> np.ndarray:
    """Convert an array of PoseStamped objects to a numpy array of positions.

    Args:
        poses (np.ndarray): Array of PoseStamped objects.

    Returns:
        np.ndarray: Numpy array of shape (n, 2) containing the x and y positions.
    """
    result_array = np.empty((len(poses), 2))
    for pose in range(len(poses)):
        result_array[pose] = np.array(
            [poses[pose].pose.position.x, poses[pose].pose.position.y]
        )
    return result_array

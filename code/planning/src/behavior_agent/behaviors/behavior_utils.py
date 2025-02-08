import rospy
import mapping_common.entity
from mapping_common.map import Map
from visualization_msgs.msg import Marker
from mapping_common.shape import Polygon

"""
This file represents the utility functions for the behavior tree
It containes parameters and utility functions to reduce code in the behaviors.
"""
MARKER_NAMESPACE: str = "behaviortree"


def get_hero_width(map: Map):
    """Returns the hero width

    Args:
        map (Map): Intermediate layer map object

    Returns:
        float:  width of the hero
    """
    hero = map.hero()
    if hero is None or hero.motion is None:
        # We currenly have no hero data.
        # -> cannot drive safely
        rospy.logerr("Behavior utils: No hero with motion found in map!")
        return 0
    return max(1.0, hero.get_width())


def get_marker_arr_in_front(entity, distance, hero, collision_masks):
    """Generates a marker array for visualization. Used for obstacle in front scenarios.

    Args:
        entity (Entity): The entity in front
        distance (float): Distance to the entity
        hero (Entity): The hero car
        collision_masks (Polygon Array): Array of polygon masks to visualize

    Returns:
        _type_: _description_
    """
    text_markers = []
    entity_markers = []
    shape_markers = []
    for mask in collision_masks:
        shape_markers.append((Polygon.from_shapely(mask), (0, 1.0, 1.0, 0.5)))
    current_velocity = hero.get_global_x_velocity() or 0.0
    entity_markers.append((entity, (1.0, 0.0, 0.0, 0.5)))

    lead_delta_velocity = (
        hero.get_delta_forward_velocity_of(entity) or -current_velocity
    )

    marker_text = (
        f"LeadDistance: {distance}\n"
        + f"LeadXVelocity: {entity.get_global_x_velocity()}\n"
        + f"DeltaV: {lead_delta_velocity}\n"
    )
    text_marker = Marker(type=Marker.TEXT_VIEW_FACING, text=marker_text)
    text_marker.pose.position.x = -2.0
    text_marker.pose.position.y = 0.0
    text_markers.append((text_marker, (1.0, 1.0, 1.0, 1.0)))

    marker_array = mapping_common.entity.shape_debug_marker_array(
        MARKER_NAMESPACE,
        entities=entity_markers,
        shapes=shape_markers,
        markers=text_markers,
    )
    return marker_array

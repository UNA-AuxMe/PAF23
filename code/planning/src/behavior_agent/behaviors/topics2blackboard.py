#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import py_trees
import py_trees_ros

from std_msgs.msg import Float32, Bool, Float32MultiArray, Int16, String
from carla_msgs.msg import CarlaSpeedometer
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from mapping.msg import Map as MapMsg

from mock.msg import Stop_sign
from perception.msg import Waypoint, LaneChange, TrafficLightState

"""
Source: https://github.com/ll7/psaf2
"""


def create_node(role_name):
    """
    This function initializes the topics which will be written to the decision
    tree blackboard and accessible by the decision tree.
    :param role_name: name of the agent
    :return: topics2blackboard the subtree of the topics in the blackboard
    """
    topics = [
        {
            "name": f"/carla/{role_name}/Speed",
            "msg": CarlaSpeedometer,
            "clearing-policy": py_trees.common.ClearingPolicy.NEVER,
        },
        {
            "name": f"/paf/{role_name}/slowed_by_car_in_front",
            "msg": Bool,
            "clearing-policy": py_trees.common.ClearingPolicy.NEVER,
        },
        {
            "name": f"/paf/{role_name}/waypoint_distance",
            "msg": Waypoint,
            "clearing-policy": py_trees.common.ClearingPolicy.ON_INITIALISE,
        },
        {
            "name": f"/paf/{role_name}/stop_sign",
            "msg": Stop_sign,
            "clearing-policy": py_trees.common.ClearingPolicy.NEVER,
        },
        {
            "name": f"/paf/{role_name}/Center/traffic_light_state",
            "msg": TrafficLightState,
            "clearing-policy": py_trees.common.ClearingPolicy.NEVER,
        },
        {
            "name": f"/paf/{role_name}/Center/traffic_light_y_distance",
            "msg": Int16,
            "clearing-policy": py_trees.common.ClearingPolicy.NEVER,
        },
        {
            "name": f"/paf/{role_name}/max_velocity",
            "msg": Float32,
            "clearing-policy": py_trees.common.ClearingPolicy.NEVER,
        },
        {
            "name": f"/paf/{role_name}/speed_limit",
            "msg": Float32,
            "clearing-policy": py_trees.common.ClearingPolicy.NEVER,
        },
        {
            "name": f"/paf/{role_name}/lane_change_distance",
            "msg": LaneChange,
            "clearing-policy": py_trees.common.ClearingPolicy.ON_INITIALISE,
        },
        {
            "name": f"/paf/{role_name}/collision",
            "msg": Float32MultiArray,
            "clearing-policy": py_trees.common.ClearingPolicy.ON_INITIALISE,
        },
        {
            "name": f"/paf/{role_name}/current_pos",
            "msg": PoseStamped,
            "clearing-policy": py_trees.common.ClearingPolicy.NEVER,
        },
        {
            "name": f"/paf/{role_name}/current_heading",
            "msg": Float32,
            "clearing-policy": py_trees.common.ClearingPolicy.NEVER,
        },
        {
            "name": f"/paf/{role_name}/overtake_success",
            "msg": Float32,
            "clearing-policy": py_trees.common.ClearingPolicy.NEVER,
        },
        {
            "name": f"/paf/{role_name}/oncoming",
            "msg": Float32,
            "clearing-policy": py_trees.common.ClearingPolicy.NEVER,
        },
        {
            "name": f"/paf/{role_name}/target_velocity",
            "msg": Float32,
            "clearing-policy": py_trees.common.ClearingPolicy.NEVER,
        },
        {
            "name": f"/paf/{role_name}/mapping/init_data",
            "msg": MapMsg,
            "clearing-policy": py_trees.common.ClearingPolicy.NEVER,
        },
        {
            "name": f"/paf/{role_name}/trajectory",
            "msg": Path,
            "clearing-policy": py_trees.common.ClearingPolicy.NEVER,
        },
        {
            "name": f"/paf/{role_name}/current_wp",
            "msg": Float32,
            "clearing-policy": py_trees.common.ClearingPolicy.NEVER,
        },
        {
            "name": f"/paf/{role_name}/curr_behavior",
            "msg": String,
            "clearing-policy": py_trees.common.ClearingPolicy.NEVER,
        },
    ]

    topics2blackboard = py_trees.composites.Parallel("Topics to Blackboard")
    for topic in topics:
        topics2blackboard.add_child(
            py_trees_ros.subscribers.ToBlackboard(
                name=topic["name"],
                topic_name=topic["name"],
                topic_type=topic["msg"],
                blackboard_variables={topic["name"]: None},
                clearing_policy=topic["clearing-policy"],
            )
        )
    return topics2blackboard

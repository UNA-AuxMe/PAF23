#!/usr/bin/env python

import functools
from py_trees.behaviours import Running
import py_trees_ros
import rospy
import sys
from behaviours import (
    intersection,
    lane_change,
    overtake,
    maneuvers,
    meta,
    road_features,
    topics2blackboard,
)
from py_trees.composites import Parallel, Selector, Sequence

"""
Source: https://github.com/ll7/psaf2
"""


def grow_a_tree(role_name):

    rules = Parallel(
        "Rules",
        children=[
            Selector(
                "Priorities",
                children=[
                    maneuvers.UnstuckRoutine("Unstuck Routine"),
                    Selector(
                        "Road Features",
                        children=[
                            maneuvers.LeaveParkingSpace("Leave Parking Space"),
                            Sequence(
                                "Intersection",
                                children=[
                                    road_features.IntersectionAhead(
                                        "Intersection Ahead?"
                                    ),
                                    Sequence(
                                        "Intersection Actions",
                                        children=[
                                            intersection.Approach(
                                                "Approach Intersection"
                                            ),
                                            intersection.Wait("Wait Intersection"),
                                            intersection.Enter("Enter Intersection"),
                                            intersection.Leave("Leave Intersection"),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    Selector(
                        "Laneswitching",
                        children=[
                            Sequence(
                                "Laneswitch",
                                children=[
                                    road_features.LaneChangeAhead("Lane Change Ahead?"),
                                    Sequence(
                                        "Lane Change Actions",
                                        children=[
                                            lane_change.Approach("Approach Change"),
                                            lane_change.Wait("Wait Change"),
                                            lane_change.Enter("Enter Change"),
                                            lane_change.Leave("Leave Change"),
                                        ],
                                    ),
                                ],
                            ),
                            Sequence(
                                "Overtaking",
                                children=[
                                    road_features.OvertakeAhead("Overtake Ahead?"),
                                    Sequence(
                                        "Overtake Actions",
                                        children=[
                                            overtake.Approach("Approach Overtake"),
                                            overtake.Wait("Wait Overtake"),
                                            overtake.Enter("Enter Overtake"),
                                            overtake.Leave("Leave Overtake"),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    maneuvers.Cruise("Cruise"),
                ],
            )
        ],
    )

    metarules = Sequence(
        "Meta",
        children=[meta.Start("Start"), rules, meta.End("End")],
    )
    root = Parallel(
        "Root",
        children=[
            topics2blackboard.create_node(role_name),
            metarules,
            Running("Idle"),
        ],
    )
    return root


def shutdown(behaviour_tree):
    behaviour_tree.interrupt()


def main():
    """
    Entry point for the demo script.
    """
    rospy.init_node("behavior_tree", anonymous=True)
    role_name = rospy.get_param("~role_name", "hero")
    root = grow_a_tree(role_name)
    behaviour_tree = py_trees_ros.trees.BehaviourTree(root)
    rospy.on_shutdown(functools.partial(shutdown, behaviour_tree))

    if not behaviour_tree.setup(timeout=15):
        rospy.logerr("Tree Setup failed")
        sys.exit(1)
    rospy.loginfo("tree setup worked")
    r = rospy.Rate(5.3)
    while not rospy.is_shutdown():
        behaviour_tree.tick()
        try:
            r.sleep()
        except rospy.ROSInterruptException:
            pass


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# ros node to test the curve generation

import rospy
import ros_compatibility as roscomp
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from ros_compatibility.node import CompatibleNode
import cv2
from scipy.interpolate import splprep, splev
import numpy as np


class CurveTestNode(CompatibleNode):
    def __init__(self):
        super(CurveTestNode, self).__init__("curve_test_node")

        self.trajectory = None

        self.trajectory_sub = self.new_subscription(
            Path,
            "/paf/acting/trajectory",
            self.__set_trajectory,
            qos_profile=1,
        )

    def __set_trajectory(self, msg: Path):
        self.trajectory = msg
        self.loginfo("Trajectory received")
        self.extract_curvature()

    def extract_curvature(self):
        coords = [
            [poseStamped.pose.position.x, poseStamped.pose.position.y]
            for poseStamped in self.trajectory.poses
        ]
        coords = np.array(coords)
        tck, u = splprep([coords[:, 0], coords[:, 1]], s=0.0009)
        u_fine = np.linspace(0, 1, 1000)
        x_smooth, y_smooth = splev(u_fine, tck)

        dx, dy = splev(u_fine, tck, der=1)
        ddx, ddy = splev(u_fine, tck, der=2)

        curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5

        return

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.loginfo("Tick")
            rate.sleep()


if __name__ == "__main__":
    roscomp.init("curve_test_node")
    node = CurveTestNode()
    node.run()

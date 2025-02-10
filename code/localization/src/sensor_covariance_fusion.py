#!/usr/bin/env python

"""
Carla publishes the IMU and GPS Data without covariances.

We have to decide on clever values and add them to the data.
This node reads a param file (sensor_covariances.yaml) and listens to topics.

When a message is received the covariance is added and the data is sent out again.
"""

import rospy
import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from sensor_msgs.msg import Imu, NavSatFix


from localization.cfg import SensorCovarianceConfig
from dynamic_reconfigure.server import Server
import numpy as np


class SensorCovarianceFusion(CompatibleNode):

    def __init__(self):
        """
        Constructor / Setup
        :return:
        """
        super().__init__("sensor_covariance_fusion")

        self.__use_gps_yaml_covariance: bool
        self.__gps_covariance: float

        self.__use_imu_yaml_covariance: bool
        self.__imu_orientation_covariance: float
        self.__imu_angular_velocity_covariance: float
        self.__imu_linear_acceleration_covariance_xy: float
        self.__imu_linear_acceleration_covariance_z: float
        Server(SensorCovarianceConfig, self.dynamic_reconfigure_callback)

        # The publishers (topic names have to coincide with ekf_config.yaml)
        self.imu_publisher = self.new_publisher(Imu, "/imu/data", qos_profile=10)
        self.gps_publisher = self.new_publisher(NavSatFix, "/gps/fix", qos_profile=10)

        # Initialize the subscriber for the IMU data
        self.imu_subscriber = self.new_subscription(
            Imu,
            "/carla/hero/IMU",
            self.imu_callback,
            qos_profile=1,
        )

        # Initialize the subscriber for the Gps data
        self.gps_subscriber = self.new_subscription(
            NavSatFix,
            "/carla/hero/GPS",
            self.gps_callback,
            qos_profile=1,
        )

    def dynamic_reconfigure_callback(self, config: "SensorCovarianceConfig", level):
        self.__use_gps_yaml_covariance = config["use_yaml_covariance_gps"]
        self.__gps_covariance = config["gps_covariance"]

        self.__use_imu_yaml_covariance = config["use_yaml_covariance_imu"]
        self.__imu_orientation_covariance = config["imu_orientation_covariance"]
        self.__imu_angular_velocity_covariance = config[
            "imu_angular_velocity_covariance"
        ]
        self.__imu_linear_acceleration_covariance_xy = config[
            "imu_linear_acceleration_covariance_xy"
        ]
        self.__imu_linear_acceleration_covariance_z = config[
            "imu_linear_acceleration_covariance_z"
        ]
        return config

    def imu_callback(self, imu: Imu):
        """An IMU message is received.

        Adds covariances and sends out again.
        """

        if self.__use_imu_yaml_covariance:
            imu.orientation_covariance = rospy.get_param("~imu_orientation")
            imu.angular_velocity_covariance = rospy.get_param("~imu_angular_velocity")
            imu.linear_acceleration_covariance = rospy.get_param(
                "~imu_linear_acceleration"
            )
        else:
            imu.orientation_covariance = np.diag(
                np.full(3, self.__imu_orientation_covariance)
            ).flatten()

            imu.angular_velocity_covariance = np.diag(
                np.full(3, self.__imu_angular_velocity_covariance)
            ).flatten()

            imu.linear_acceleration_covariance = np.diag(
                np.full(3, self.__imu_linear_acceleration_covariance_xy)
            ).flatten()

            imu.linear_acceleration_covariance[8] = (
                self.__imu_linear_acceleration_covariance_z
            )

        self.imu_publisher.publish(imu)

    def gps_callback(self, gps: NavSatFix):
        """GPS data is received.

        Add covariances and sent out again.
        """
        if self.__use_gps_yaml_covariance:
            gps.position_covariance = rospy.get_param("~gps_position")
        else:
            gps.position_covariance = np.diag(
                np.full(3, self.__gps_covariance)
            ).flatten()

        self.gps_publisher.publish(gps)


def main(args=None):
    roscomp.init("sensor_covariance_fusion", args=args)

    try:
        node = SensorCovarianceFusion()
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        roscomp.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python

"""
This Node listens to GPS Data and converts this data into an odometry message.
The odometry message is then passed into the global ekf.

It is a substitute for navsat_transform node which only works in UTM coordinates but
CARLA uses WGS coordinate system.
"""

from rospy import Publisher
import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry

from coordinate_transformation import CoordinateTransformer


class GpsTransform(CompatibleNode):

    def __init__(self):
        super().__init__("gps_transform")
        self.transformer = CoordinateTransformer()
        self.role_name = self.get_param("role_name", "hero")

        # Initalize publisher for Odometry data
        self.odometry_publisher: Publisher = self.new_publisher(
            Odometry, "/odometry/gps", qos_profile=10
        )

        # Initialize the subscriber for the GPS data
        self.gps_subscriber = self.new_subscription(
            NavSatFix,
            "/gps/fix",
            self.gps_callback,
            qos_profile=10,
        )

    def process_data(self, gps: NavSatFix):
        """Transforms GPS data to Odometry message

        Args:
            gps (NavSatFix): GPS Data to process
        """
        out = Odometry()
        out.header = gps.header
        out.header.frame_id = "global"
        out.child_frame_id = self.role_name

        (
            out.pose.pose.position.x,
            out.pose.pose.position.y,
            out.pose.pose.position.z,
        ) = self.transformer.gnss_to_xyz(gps.latitude, gps.longitude, gps.altitude)
        for i in range(3):
            out.pose.covariance[i + (i * 6)] = gps.position_covariance[i + (i * 3)]

        self.odometry_publisher.publish(out)

    def gps_callback(self, gps: NavSatFix):
        self.process_data(gps)


def main(args=None):
    roscomp.init("gps_transform", args=args)

    try:
        node = GpsTransform()
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        roscomp.shutdown()


if __name__ == "__main__":
    main()

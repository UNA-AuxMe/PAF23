#!/usr/bin/env python

import math
from tf.transformations import euler_from_quaternion
import numpy as np
import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix, Imu

# from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String
from coordinate_transformation import CoordinateTransformer
from coordinate_transformation import quat_to_heading
from xml.etree import ElementTree as eTree

GPS_RUNNING_AVG_ARGS: int = 10


class PositionHeadingPublisherNode(CompatibleNode):
    """
    This Node subcribes to the unfiltered GNSS and Heading Signals and
    publishes a (PoseStamped) gps signal as currentPos
    as well as a (Float32) heading signal as currentHeading.
    Both signals can be filtered with the available filters:
    Position Filter values:
        - "Kalman" (Default)
        - "RunningAvg"
        - "None"
    Heading Filter values:
        - "Kalman" (Default)
        - "None"
        - "Old" (Buggy for demonstration purposes only)
    These can be turned on and off in the launch file.

    !When creating a new filter, the corresponding subscriber and publisher
    must be added in the constructor for clean modular programming!

    For more information:
    ../../doc/perception/position_heading_publisher_node.md
    """

    def __init__(self):
        """
        Constructor / Setup
        :return:
        """

        super(PositionHeadingPublisherNode, self).__init__(
            "position_heading_publisher_node"
        )

        """
        Possible Filters:
        Pos: Kalman, RunningAvg, None
        Heading: Kalman, None
        """
        # Filter used:
        self.pos_filter = self.get_param("pos_filter", "Kalman")
        self.heading_filter = self.get_param("heading_filter", "Kalman")
        self.loginfo(
            "position_heading_publisher_node started with Pos Filter:"
            + self.pos_filter
            + " and Heading Filter: "
            + self.heading_filter
        )

        # basic info
        self.role_name = self.get_param("role_name", "hero")
        self.control_loop_rate = self.get_param("control_loop_rate", "0.05")

        # todo: automatically detect town
        self.transformer = None

        # 3D Odometry (GPS)
        self.avg_xyz = np.zeros((GPS_RUNNING_AVG_ARGS, 3))
        self.avg_gps_counter: int = 0

        # region Subscriber START

        self.map_sub = self.new_subscription(
            String,
            "/carla/" + self.role_name + "/OpenDRIVE",
            self.get_geoRef,
            qos_profile=1,
        )

        self.imu_subscriber = self.new_subscription(
            Imu,
            "/carla/" + self.role_name + "/IMU",
            self.publish_unfiltered_heading,
            qos_profile=1,
        )

        self.gps_subscriber = self.new_subscription(
            NavSatFix,
            "/carla/" + self.role_name + "/GPS",
            self.publish_unfiltered_gps,
            qos_profile=1,
        )

        # Create subscribers depending on the filter used
        # Pos Filter:
        if self.pos_filter == "Kalman":
            self.kalman_pos_subscriber = self.new_subscription(
                PoseStamped,
                "/paf/" + self.role_name + "/kalman_pos",
                self.publish_kalman_pos_as_current_pos,
                qos_profile=1,
            )
        elif self.pos_filter == "RunningAvg":
            self.gps_subscriber_for_running_avg = self.new_subscription(
                NavSatFix,
                "/carla/" + self.role_name + "/GPS",
                self.publish_running_avg_pos_as_current_pos,
                qos_profile=1,
            )
        elif self.pos_filter == "None":
            # No additional subscriber needed since the unfiltered GPS data is
            # subscribed by self.gps_subscriber
            pass

        # insert additional elifs for other filters here

        # Heading Filter:
        if self.heading_filter == "Kalman":
            self.kalman_heading_subscriber = self.new_subscription(
                Float32,
                "/paf/" + self.role_name + "/kalman_heading",
                self.publish_current_heading,
                qos_profile=1,
            )
        elif self.heading_filter == "None":
            # No additional subscriber needed since the unfiltered heading
            # data is subscribed by self.imu_subscriber
            pass

        # insert additional elifs for other filters here

        # endregion Subscriber END

        # region Publisher START
        # Orientation
        self.unfiltered_heading_publisher = self.new_publisher(
            Float32, f"/paf/{self.role_name}/unfiltered_heading", qos_profile=1
        )

        # 3D Odometry (GPS) for Filters
        self.unfiltered_gps_publisher = self.new_publisher(
            PoseStamped, f"/paf/{self.role_name}/unfiltered_pos", qos_profile=1
        )
        # Publishes current_pos depending on the filter used
        self.cur_pos_publisher = self.new_publisher(
            PoseStamped, f"/paf/{self.role_name}/current_pos", qos_profile=1
        )

        self.__heading: float = 0
        self.__heading_publisher = self.new_publisher(
            Float32, f"/paf/{self.role_name}/current_heading", qos_profile=1
        )

    # endregion Publisher END

    # region HEADING FUNCTIONS
    def publish_unfiltered_heading(self, data: Imu):
        """
        This method is called when new IMU data is received.
        It handles all necessary updates and publishes the heading.
        :param data: new IMU measurement
        :return:
        """
        data_orientation_q = [
            data.orientation.x,
            data.orientation.y,
            data.orientation.z,
            data.orientation.w,
        ]

        heading = quat_to_heading(data_orientation_q)

        # In the case of using "None" filter, the heading is
        # published as current heading, since it is not filtered
        if self.heading_filter == "None":
            self.__heading = heading
            self.__heading_publisher.publish(self.__heading)
        elif self.heading_filter == "Old":
            # In the case of using "Old" filter, the heading is
            # calculated the WRONG way, just for demonstration purposes
            roll, pitch, yaw = euler_from_quaternion(data_orientation_q)
            raw_heading = math.atan2(roll, pitch)

            # transform raw_heading so that:
            # ---------------------------------------------------------------
            # | 0 = x-axis | pi/2 = y-axis | pi = -x-axis | -pi/2 = -y-axis |
            # ---------------------------------------------------------------
            heading = (raw_heading - (math.pi / 2)) % (2 * math.pi) - math.pi
            self.__heading = heading
            self.__heading_publisher.publish(self.__heading)
        # insert additional elifs for other filters here
        else:
            # in each other case the heading is published as unfiltered heading
            # for further filtering in other nodes such as the kalman node
            self.unfiltered_heading_publisher.publish(heading)

    def publish_current_heading(self, data: Float32):
        """
        This method is called when new heading data is received.
        It handles all necessary updates and publishes the heading.
        :param data: new heading measurement
        :return:
        """
        self.__heading = data.data
        self.__heading_publisher.publish(self.__heading)

    # insert new heading functions here...

    # endregion HEADING FUNCTIONS END

    # region POSITION FUNCTIONS

    def publish_running_avg_pos_as_current_pos(self, data: NavSatFix):
        """
        This method is called when new GNSS data is received.
        The function calculates the average position and then publishes it.
        Measurements are also transformed to global xyz-coordinates
        :param data: GNSS measurement
        :return:
        """
        # Make sure position is only published when reference values have been
        # read from the Map
        if CoordinateTransformer.ref_set is False:
            self.transformer = CoordinateTransformer()
            CoordinateTransformer.ref_set = True
        if CoordinateTransformer.ref_set is True:
            lat = data.latitude
            lon = data.longitude
            alt = data.altitude

            x, y, z = self.transformer.gnss_to_xyz(lat, lon, alt)

            self.avg_xyz = np.roll(self.avg_xyz, -1, axis=0)
            self.avg_xyz[-1] = np.matrix([x, y, z])

            avg_x, avg_y, avg_z = np.mean(self.avg_xyz, axis=0)

            cur_pos = PoseStamped()

            cur_pos.header.stamp = data.header.stamp
            cur_pos.header.frame_id = "global"

            cur_pos.pose.position.x = avg_x
            cur_pos.pose.position.y = avg_y
            cur_pos.pose.position.z = avg_z

            cur_pos.pose.orientation.x = 0
            cur_pos.pose.orientation.y = 0
            cur_pos.pose.orientation.z = 1
            cur_pos.pose.orientation.w = 0

            self.cur_pos_publisher.publish(cur_pos)

    def publish_kalman_pos_as_current_pos(self, data: PoseStamped):
        """
        This method is called when new kalman filter data is received.
        The function publishes the kalman position as current position
        :param data: PoseStamped
        :return:
        """
        self.cur_pos_publisher.publish(data)

    def publish_non_linear_kalman_pos_as_current_pos(self, data: PoseStamped):
        """
        This method is called when new non-linear kalman filter data is
        received.
        The function publishes the non-linear kalman position as current
        position
        :param data: PoseStamped
        :return:
        """
        self.cur_pos_publisher.publish(data)

    def publish_unfiltered_gps(self, data: NavSatFix):
        """
        This method is called when new GNSS data is received.
        It publishes the unfiltered GPS data as XYZ (PoseStamped).
        :param data: GNSS measurement
        :return:
        """
        # Make sure position is only published when reference values have been
        # read from the Map
        if CoordinateTransformer.ref_set is False:
            self.transformer = CoordinateTransformer()
            CoordinateTransformer.ref_set = True
        if CoordinateTransformer.ref_set is True:
            lat = data.latitude
            lon = data.longitude
            alt = data.altitude

            x, y, z = self.transformer.gnss_to_xyz(lat, lon, alt)

            unfiltered_pos = PoseStamped()

            unfiltered_pos.header.stamp = data.header.stamp
            unfiltered_pos.header.frame_id = "global"

            unfiltered_pos.pose.position.x = x
            unfiltered_pos.pose.position.y = y
            unfiltered_pos.pose.position.z = z

            unfiltered_pos.pose.orientation.x = 0
            unfiltered_pos.pose.orientation.y = 0
            unfiltered_pos.pose.orientation.z = 1
            unfiltered_pos.pose.orientation.w = 0

            # In the case of using "None" filter, the pos is
            # published as current pos, since it is not filtered
            if self.pos_filter == "None":
                self.cur_pos_publisher.publish(unfiltered_pos)
            else:
                # in each other case the pos is published as unfiltered pos
                # for further filtering in other nodes such as the kalman node
                self.unfiltered_gps_publisher.publish(unfiltered_pos)

    # insert new position functions here...

    # endregion POSITION FUNCTIONS END

    def get_geoRef(self, opendrive: String):
        """_summary_
        Reads the reference values for lat and lon from the carla OpenDriveMap
        This is necessary for the coordinate transformation from GNSS to XYZ
        Args:
            opendrive (String): OpenDrive Map from carla
        """
        root = eTree.fromstring(opendrive.data)
        header = root.find("header")
        geoRefText = header.find("geoReference").text

        latString = "+lat_0="
        lonString = "+lon_0="

        indexLat = geoRefText.find(latString)
        indexLon = geoRefText.find(lonString)

        indexLatEnd = geoRefText.find(" ", indexLat)
        indexLonEnd = geoRefText.find(" ", indexLon)

        latValue = float(geoRefText[indexLat + len(latString) : indexLatEnd])
        lonValue = float(geoRefText[indexLon + len(lonString) : indexLonEnd])

        CoordinateTransformer.la_ref = latValue
        CoordinateTransformer.ln_ref = lonValue
        CoordinateTransformer.ref_set = True
        self.transformer = CoordinateTransformer()

    def run(self):
        """
        Control loop
        :return:
        """
        self.spin()


def main(args=None):
    """
    main function
    :param args:
    :return:
    """
    roscomp.init("position_heading_publisher_node", args=args)
    try:
        node = PositionHeadingPublisherNode()
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        roscomp.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python

"""
######### !!!!!!!!!! WARNING !!!!!!!!!!!!!!! ##########

This Node is currently not in use and in a non functional state.
Do not use this node unless you know what to do and how to use it.

######### !!!!!!!!!! WARNING !!!!!!!!!!!!!!! ##########
"""
import os
import math
import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Header

# from tf.transformations import euler_from_quaternion
from std_msgs.msg import Float32MultiArray
import rospy
import carla

GPS_RUNNING_AVG_ARGS: int = 10
DATA_SAVING_MAX_TIME: int = 45
FOLDER_PATH: str = "/Position_Heading_Datasets"


class Evaluator(CompatibleNode):
    """
    Node publishes a filtered gps signal.
    This is achieved using a rolling average.
    """

    def checkout_carla(self, timer_event):
        client = self.client
        world = client.get_world()
        for actor in world.get_actors():
            if actor.attributes.get("role_name") == "hero":
                self.carla_car = actor
        if self.carla_car is not None:
            print(self.carla_car.get_location())

    def __init__(self):
        """
        Constructor / Setup
        :return:
        """

        super().__init__("evaluator")

        # basic info
        self.role_name = self.get_param("role_name", "hero")
        self.control_loop_rate = self.get_param("control_loop_rate", "0.05")

        # carla attributes
        CARLA_HOST = os.environ.get(
            "CARLA_SIM_HOST", "paf-carla-simulator-1"
        )  # carla-simulator")
        CARLA_PORT = int(os.environ.get("CARLA_PORT", "2000"))

        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.carla_car = None
        self.new_timer(1, self.checkout_carla)

        # Tracked Attributes for Debugging
        self.current_pos = PoseStamped()
        self.current_heading = Float32()
        self.carla_current_pos = carla.Location()
        self.carla_current_heading = 0.0
        self.unfiltered_pos = PoseStamped()
        self.unfiltered_heading = Float32()

        self.position_debug_data = Float32MultiArray()
        self.heading_debug_data = Float32MultiArray()

        # test_filter attributes for any new filter to be tested
        # default is kalman filter
        self.test_filter_pos = PoseStamped()
        self.test_filter_heading = Float32()

        self.loginfo("Position Heading Filter Debug node started")

        # region Subscriber START

        # Current_pos subscriber:
        self.current_pos_subscriber = self.new_subscription(
            PoseStamped,
            f"/paf/{self.role_name}/current_pos",
            self.set_current_pos,
            qos_profile=1,
        )

        # Current_heading subscriber:
        self.current_heading_subscriber = self.new_subscription(
            Float32,
            f"/paf/{self.role_name}/current_heading",
            self.set_current_heading,
            qos_profile=1,
        )

        # test_filter_pos subscriber:
        self.test_filter_pos_subscriber = self.new_subscription(
            PoseStamped,
            f"/paf/{self.role_name}/kalman_pos",
            self.set_test_filter_pos,
            qos_profile=1,
        )
        # test_filter_heading subscriber:
        self.test_filter_heading_subscriber = self.new_subscription(
            Float32,
            f"/paf/{self.role_name}/kalman_heading",
            self.set_test_filter_heading,
            qos_profile=1,
        )

        # Unfiltered_pos subscriber:
        self.unfiltered_pos_subscriber = self.new_subscription(
            PoseStamped,
            f"/paf/{self.role_name}/unfiltered_pos",
            self.set_unfiltered_pos,
            qos_profile=1,
        )
        # Unfiltered_heading subscriber:
        self.unfiltered_heading_subscriber = self.new_subscription(
            Float32,
            f"/paf/{self.role_name}/unfiltered_heading",
            self.set_unfiltered_heading,
            qos_profile=1,
        )

        # endregion Subscriber END

        # region Publisher START

        # ideal carla publisher for easier debug with rqt_plot
        self.carla_heading_publisher = self.new_publisher(
            Float32, f"/paf/{self.role_name}/carla_current_heading", qos_profile=1
        )

        self.carla_pos_publisher = self.new_publisher(
            PoseStamped, f"/paf/{self.role_name}/carla_current_pos", qos_profile=1
        )

        # Error Publisher
        self.position_debug_publisher = self.new_publisher(
            Float32MultiArray, f"/paf/{self.role_name}/position_debug", qos_profile=1
        )
        self.heading_debug_publisher = self.new_publisher(
            Float32MultiArray, f"/paf/{self.role_name}/heading_debug", qos_profile=1
        )

        # endregion Publisher END

    # region Subscriber Callbacks

    def set_unfiltered_heading(self, data: Float32):
        """
        This method is called when new unfiltered_heading data is received.
        """
        self.unfiltered_heading = data

    def set_unfiltered_pos(self, data: PoseStamped):
        """
        This method is called when new unfiltered_pos data is received.
        """
        self.unfiltered_pos = data

    def set_current_pos(self, data: PoseStamped):
        """
        This method is called when new current_pos data is received.
        """
        self.current_pos = data

    def set_current_heading(self, data: Float32):
        """
        This method is called when new current_heading data is received.
        """
        self.current_heading = data

    def set_test_filter_pos(self, data: PoseStamped):
        """
        This method is called when new test_filter_pos data is received.
        """
        self.test_filter_pos = data

    def set_test_filter_heading(self, data: Float32):
        """
        This method is called when new test_filter_heading data is received.
        """
        self.test_filter_heading = data

    # endregion CSV data save methods

    def set_carla_attributes(self):
        """
        This method sets the carla attributes.
        """
        for actor in self.world.get_actors():
            if actor.attributes.get("role_name") == "hero":
                self.carla_car = actor
                break
        if self.carla_car is None:
            # self.logwarn("Carla Hero car still none!")
            return
        else:
            # save carla Position
            # for some reason the carla y is flipped
            carla_pos = self.carla_car.get_location()
            carla_pos.y = -carla_pos.y
            self.carla_current_pos = carla_pos

            # save carla Heading
            # for some reason the carla yaw is in flipped degrees
            # -> convert to radians
            # -> also flip the sign to minus
            self.carla_current_heading = -math.radians(
                self.carla_car.get_transform().rotation.yaw
            )

    def position_debug(self):
        if self.carla_car is None:
            # self.logwarn("""Carla Hero car still none!
            #              Can not record data for position_debug yet.""")
            return

        debug = Float32MultiArray()

        debug.data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # all x and y coordinates
        debug.data[0] = self.unfiltered_pos.pose.position.x
        debug.data[1] = self.unfiltered_pos.pose.position.y
        debug.data[2] = self.carla_current_pos.x
        debug.data[3] = self.carla_current_pos.y
        debug.data[4] = self.current_pos.pose.position.x
        debug.data[5] = self.current_pos.pose.position.y
        debug.data[6] = self.test_filter_pos.pose.position.x
        debug.data[7] = self.test_filter_pos.pose.position.y

        # error between carla_current_pos and unfiltered_pos
        debug.data[8] = self.carla_current_pos.x - self.unfiltered_pos.pose.position.x
        debug.data[9] = self.carla_current_pos.y - self.unfiltered_pos.pose.position.y
        debug.data[10] = math.sqrt(
            (self.carla_current_pos.x - self.unfiltered_pos.pose.position.x) ** 2
            + (self.carla_current_pos.y - self.unfiltered_pos.pose.position.y) ** 2
        )

        # error between carla_current_pos and current_pos
        debug.data[11] = self.carla_current_pos.x - self.current_pos.pose.position.x
        debug.data[12] = self.carla_current_pos.y - self.current_pos.pose.position.y
        debug.data[13] = math.sqrt(
            (self.carla_current_pos.x - self.current_pos.pose.position.x) ** 2
            + (self.carla_current_pos.y - self.current_pos.pose.position.y) ** 2
        )

        # error between carla_current_pos and test_filter_pos
        debug.data[14] = self.carla_current_pos.x - self.test_filter_pos.pose.position.x
        debug.data[15] = self.carla_current_pos.y - self.test_filter_pos.pose.position.y
        debug.data[16] = math.sqrt(
            (self.carla_current_pos.x - self.test_filter_pos.pose.position.x) ** 2
            + (self.carla_current_pos.y - self.test_filter_pos.pose.position.y) ** 2
        )

        self.position_debug_data = debug
        self.position_debug_publisher.publish(debug)

        # for easier debugging with rqt_plot
        # Publish carla Location as PoseStamped:
        self.carla_pos_publisher.publish(
            carla_location_to_pose_stamped(self.carla_current_pos)
        )

    def heading_debug(self):
        """
        This method is called every loop_rate.
        It publishes and saves:
        unfiltered_heading,
        carla_current_heading,
        current_heading,
        test_filter_heading

        It also publishes and saves heading differences between:
        carla_current_heading and unfiltered_heading,
        carla_current_heading and current_heading,
        carla_current_heading and test_filter_heading
        to make filters comparable.

        # unfiltered_heading in debug.data[0]
        # carla_current_heading in debug.data[1]
        # current_heading in debug.data[2]
        # test_filter_heading in debug.data[3]
        # carla_current_heading - unfiltered_heading in debug.data[4]
        # carla_current_heading - current_heading in debug.data[5]
        # carla_current_heading - test_filter_heading in debug.data[6]
        """

        if self.carla_car is None:
            # self.logwarn("""Carla Hero car still none!
            #              Can not record data for heading_debug yet.""")
            return

        debug = Float32MultiArray()

        debug.data = [0, 0, 0, 0, 0, 0, 0]

        # all heading values
        debug.data[0] = self.unfiltered_heading.data
        debug.data[1] = self.carla_current_heading
        debug.data[2] = self.current_heading.data
        debug.data[3] = self.test_filter_heading.data

        # error between carla_current_heading and unfiltered_heading
        debug.data[4] = self.carla_current_heading - self.unfiltered_heading.data

        # error between carla_current_heading and current_heading
        debug.data[5] = self.carla_current_heading - self.current_heading.data

        # error between carla_current_heading and test_filter_heading
        debug.data[6] = self.carla_current_heading - self.test_filter_heading.data

        self.heading_debug_data = debug
        self.heading_debug_publisher.publish(debug)

        # for easier debugging with rqt_plot
        self.carla_heading_publisher.publish(self.carla_current_heading)

    # Main method of the node
    def run(self):
        self.spin()


def carla_location_to_pose_stamped(location: carla.Location) -> PoseStamped:
    """
    Convert a carla.Location to a geometry_msgs/PoseStamped message.
    """
    pose_stamped = PoseStamped()
    # Fill in the header
    pose_stamped.header = Header()
    pose_stamped.header.stamp = rospy.Time.now()

    # Fill in the pose
    pose_stamped.pose.position.x = location.x
    pose_stamped.pose.position.y = location.y
    pose_stamped.pose.position.z = location.z

    # Assuming you have no orientation information
    pose_stamped.pose.orientation.x = 0
    pose_stamped.pose.orientation.y = 0
    pose_stamped.pose.orientation.z = 0
    pose_stamped.pose.orientation.w = 1

    return pose_stamped


def main(args=None):
    """
    main function
    :param args:
    :return:
    """

    roscomp.init("evaluator", args=args)
    try:
        node = Evaluator()
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        roscomp.shutdown()


if __name__ == "__main__":
    main()

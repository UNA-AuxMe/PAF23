#!/usr/bin/env python
# from typing import List
import math
from math import atan, sqrt, sin, cos
import numpy as np
import ros_compatibility as roscomp
from carla_msgs.msg import CarlaSpeedometer
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from ros_compatibility.node import CompatibleNode
from rospy import Publisher, Subscriber
from std_msgs.msg import Float32
from acting.msg import StanleyDebug

from acting.helper_functions import vector_angle, points_to_vector

from dynamic_reconfigure.server import Server
from control.cfg import StanleyConfig


class StanleyController(CompatibleNode):
    def __init__(self):
        super(StanleyController, self).__init__("stanley_controller")
        self.loginfo("StanleyController node started")

        self.control_loop_rate = self.get_param("control_loop_rate", 0.05)
        self.role_name = self.get_param("role_name", "ego_vehicle")

        # Subscribers
        self.position_sub: Subscriber = self.new_subscription(
            Path, "/paf/acting/trajectory", self.__set_path, qos_profile=1
        )

        self.path_sub: Subscriber = self.new_subscription(
            PoseStamped,
            "/paf/acting/current_pos",
            self.__set_position,
            qos_profile=1,
        )

        self.velocity_sub: Subscriber = self.new_subscription(
            CarlaSpeedometer,
            f"/carla/{self.role_name}/Speed",
            self.__set_velocity,
            qos_profile=1,
        )

        self.heading_sub: Subscriber = self.new_subscription(
            Float32,
            "/paf/acting/current_heading",
            self.__set_heading,
            qos_profile=1,
        )

        self.stanley_steer_pub: Publisher = self.new_publisher(
            Float32, f"/paf/{self.role_name}/stanley_steer", qos_profile=1
        )

        self.debug_publisher: Publisher = self.new_publisher(
            StanleyDebug, f"/paf/{self.role_name}/stanley_debug", qos_profile=1
        )

        self.__position: tuple[float, float] = None  # x , y
        self.__path: Path = None
        self.__heading: float = None
        self.__velocity: float = None

        self.K_CROSSERR: float = 0.4  # 1.24 was optimal in dev-launch!
        Server(StanleyConfig, self.dynamic_reconfigure_callback)

    def dynamic_reconfigure_callback(self, config, level):
        self.K_CROSSERR = config["k_crosserr"]
        return config

    def run(self):
        """
        Starts the main loop of the node
        :return:
        """
        self.loginfo("StanleyController node running")

        def loop(timer_event=None):
            """
            Main loop of the acting node
            :param timer_event: Timer event from ROS
            :return:
            """
            if self.__path is None:
                self.logwarn(
                    "StanleyController hasn't received a path yet "
                    "and can therefore not publish steering"
                )
                return
            if self.__position is None:
                self.logwarn(
                    "StanleyController hasn't received the"
                    "position of the vehicle yet "
                    "and can therefore not publish steering"
                )
                return

            if self.__heading is None:
                self.logwarn(
                    "StanleyController hasn't received the"
                    "heading of the vehicle yet and"
                    "can therefore not publish steering"
                )
                return

            if self.__velocity is None:
                self.logwarn(
                    "StanleyController hasn't received the "
                    "velocity of the vehicle yet "
                    "and can therefore not publish steering"
                )
                return
            self.stanley_steer_pub.publish(self.__calculate_steer())

        self.new_timer(self.control_loop_rate, loop)
        self.spin()

    def __calculate_steer(self) -> float:
        """
        Calculates the steering angle based on the current information
        using the Stanley algorithm
        :return: steering angle
        """
        # do not allow dividing by 0 or < 1
        if self.__velocity < 1:
            return 0
        current_velocity: float = self.__velocity

        closest_point_idx = self.__get_closest_point_index()
        # calculate heading_err from current_heading and traj_heading
        traj_heading = self.__get_path_heading(closest_point_idx)
        heading_err = self.__heading - traj_heading
        heading_err = ((heading_err + math.pi) % (2 * math.pi)) - math.pi
        # calculate cross_err from current_position and closest traj_point
        closest_point: PoseStamped = self.__path.poses[closest_point_idx]
        cross_err = self.__get_cross_err(closest_point.pose.position)
        # * -1 because it is inverted compared to PurePursuit
        steering_angle = 1 * (
            heading_err + atan((self.K_CROSSERR * cross_err) / current_velocity)
        )
        # -> for debugging
        debug_msg = StanleyDebug()
        debug_msg.heading = self.__heading
        debug_msg.path_heading = traj_heading
        debug_msg.cross_err = abs(cross_err)
        debug_msg.heading_err = heading_err
        debug_msg.steering_angle = steering_angle
        self.debug_publisher.publish(debug_msg)
        # for debugging <-
        return steering_angle

    def __get_closest_point_index(self) -> int:
        """
        Returns index of the nearest point of the trajectory
        :return: Index of the closest point
        """
        if len(self.__path.poses) < 2:
            return -1

        min_dist = 10e100
        min_dist_idx = -1

        for i in range(0, len(self.__path.poses)):
            temp_pose: PoseStamped = self.__path.poses[i]
            dist = self.__dist_to(temp_pose.pose.position)
            if min_dist > dist:
                min_dist = dist
                min_dist_idx = i
        return min_dist_idx

    def __get_path_heading(self, index: int) -> float:
        """
        Calculates the heading of the current path at index
        :param index: point of interest
        :return: heading at path[index]
        """
        cur_pos: Point = self.__path.poses[index].pose.position
        l_path = len(self.__path.poses)

        if l_path == 1:
            return 0

        heading_sum = 0
        heading_sum_args = 0

        if index > 0:
            # Calculate heading from the previous point on the trajectory
            prv_point: Point = self.__path.poses[index - 1].pose.position

            prv_v_x, prv_v_y = points_to_vector(
                (prv_point.x, prv_point.y), (cur_pos.x, cur_pos.y)
            )

            heading_sum += vector_angle(prv_v_x, prv_v_y)
            heading_sum_args += 1

        elif index < l_path - 1:
            # Calculate heading to the following point on the trajectory
            aft_point: Point = self.__path.poses[index + 1].pose.position

            aft_v_x, aft_v_y = points_to_vector(
                (aft_point.x, aft_point.y), (cur_pos.x, cur_pos.y)
            )

            heading_sum += vector_angle(aft_v_x, aft_v_y)
            heading_sum_args += 1

        return heading_sum / heading_sum_args

    def __get_cross_err(self, pos: Point) -> float:
        """
        Returns the Distance between current position and target position.
        The distance is negative/positive based on whether the closest point
        is to the left or right of the vehicle.
        :param pos:
        :return:
        """
        dist = self.__dist_to(pos)
        x = self.__position[0]
        y = self.__position[1]

        alpha = 0
        if self.__heading is not None:
            alpha = self.__heading + (math.pi / 2)
        v_e_0 = (0, 1)
        v_e = (
            cos(alpha) * v_e_0[0] - sin(alpha) * v_e_0[1],
            sin(alpha) * v_e_0[0] + cos(alpha) * v_e_0[1],
        )

        # define a vector (v_ab) with length 10 centered on the cur pos
        # of the vehicle, with a heading parallel to that of the vehicle
        a = (x + (v_e[0] * 2.5), y + (v_e[1] * 2.5))
        b = (x - (v_e[0] * 2.5), y - (v_e[1] * 2.5))

        v_ab = (b[0] - a[0], b[1] - a[1])
        v_am = (pos.x - a[0], pos.y - a[1])

        c = np.array([[v_ab[0], v_am[0]], [v_ab[1], v_am[1]]])
        temp_sign = np.linalg.det(c)

        min_sign = 0.01  # to avoid rounding errors

        if temp_sign > -min_sign:
            sign = -1
        else:
            sign = 1

        res = dist * sign

        return res

    def __dist_to(self, pos: Point) -> float:
        """
        Distance between current position and target position (only (x,y))
        :param pos: targeted position
        :return: distance
        """
        x_cur = self.__position[0]
        y_cur = self.__position[1]
        x_target = pos.x
        y_target = pos.y
        d = (x_target - x_cur) ** 2 + (y_target - y_cur) ** 2
        return sqrt(d)

    def __set_position(self, data: PoseStamped, min_diff=0.001):
        """
        Updates the current position of the vehicle
        To avoid problems when the car is stationary, new positions will only
        be accepted, if they are a certain distance from the current one
        :param data: new position as PoseStamped
        :param min_diff: minium difference between new and current point for
        the new point to be accepted
        :return:
        """
        if self.__position is None:
            x0 = data.pose.position.x
            y0 = data.pose.position.y
            self.__position = (x0, y0)
            return

        # check if the new position is valid
        dist = self.__dist_to(data.pose.position)
        if dist < min_diff:
            self.logdebug(
                "New position disregarded, "
                f"as dist ({round(dist, 3)}) to current pos "
                f"< min_diff ({round(min_diff, 3)})"
            )
            return

        new_x = data.pose.position.x
        new_y = data.pose.position.y
        self.__position = (new_x, new_y)

    def __set_path(self, data: Path):
        path_len = len(data.poses)
        if path_len < 1:
            self.loginfo("Stanley: Empty path received and disregarded")
            return
        self.__path = data

    def __set_heading(self, data: Float32):
        self.__heading = data.data

    def __set_velocity(self, data: CarlaSpeedometer):
        self.__velocity = data.speed


def main(args=None):
    """
    Main function starts the node
    :param args:
    """
    roscomp.init("stanley_controller", args=args)

    try:
        node = StanleyController()
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        roscomp.shutdown()


if __name__ == "__main__":
    main()

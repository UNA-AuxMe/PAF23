#!/usr/bin/env python
import math
import time

import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaSpeedometer
from ros_compatibility.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Bool, Float32, String

from dynamic_reconfigure.server import Server
from control.cfg import ControllerConfig
import rospy


class VehicleController(CompatibleNode):
    """
    This node is responsible for collecting all data needed for the
    vehicle_control_cmd and sending it.
    The node also decides weather to use the stanley or pure pursuit
    controller.
    If the node receives an emergency msg, it will bring the vehicle to a stop
    and send an emergency msg with data = False back, after the velocity of the
    vehicle has reached 0.
    INFO: Currently the loop of the node has a sleep command in it. The control
    command triggers the carla simulator to render the next frame. If the loop
    does not have the time to sleep the simulator will run as fast as the system
    allows it to run. If your system is too slow to run with the 0.1 loop_sleep_time
    you could slow it down by setting the loop_sleep_time to a higher value.
    """

    def __init__(self):
        super(VehicleController, self).__init__("vehicle_controller")
        self.loginfo("VehicleController node started")

        # Configuration parameters
        self.control_loop_rate = self.get_param("control_loop_rate", 0.05)
        self.role_name = self.get_param("role_name", "ego_vehicle")

        # State variables
        self.__curr_behavior = None
        self.__reverse = False
        self.__emergency = False
        self.__velocity = 0.0
        self.__brake = 0.0
        self.__throttle = 0.0
        self._p_steer = 0.0
        self._s_steer = 0.0

        # Manual control and Stanley controller flags
        self.MANUAL_OVERRIDE = False
        self.MANUAL_STEER = 0.0
        self.MANUAL_THROTTLE = 0.0
        self.STANLEY_OFF = False

        # Initialize publishers
        self.control_publisher = self.new_publisher(
            CarlaEgoVehicleControl,
            f"/carla/{self.role_name}/vehicle_control_cmd",
            qos_profile=10,
        )
        self.status_pub = self.new_publisher(
            Bool,
            f"/carla/{self.role_name}/status",
            qos_profile=QoSProfile(
                depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL
            ),
        )
        self.controller_pub = self.new_publisher(
            Float32,
            f"/paf/{self.role_name}/controller",
            qos_profile=QoSProfile(
                depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL
            ),
        )
        self.emergency_pub = self.new_publisher(
            Bool,
            f"/paf/{self.role_name}/emergency",
            qos_profile=QoSProfile(
                depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL
            ),
        )

        # Initialize subscribers
        self.new_subscription(
            String,
            f"/paf/{self.role_name}/curr_behavior",
            self.__set_curr_behavior,
            qos_profile=1,
        )
        self.new_subscription(
            Bool,
            f"/paf/{self.role_name}/emergency",
            self.__set_emergency,
            qos_profile=1,
        )
        self.new_subscription(
            CarlaSpeedometer,
            f"/carla/{self.role_name}/Speed",
            self.__get_velocity,
            qos_profile=1,
        )
        self.new_subscription(
            Float32,
            f"/paf/{self.role_name}/throttle",
            self.__set_throttle,
            qos_profile=1,
        )
        self.new_subscription(
            Float32, f"/paf/{self.role_name}/brake", self.__set_brake, qos_profile=1
        )
        self.new_subscription(
            Bool, f"/paf/{self.role_name}/reverse", self.__set_reverse, qos_profile=1
        )
        self.new_subscription(
            Float32,
            f"/paf/{self.role_name}/pure_pursuit_steer",
            self.__set_pure_pursuit_steer,
            qos_profile=1,
        )
        self.new_subscription(
            Float32,
            f"/paf/{self.role_name}/stanley_steer",
            self.__set_stanley_steer,
            qos_profile=1,
        )

        # Dynamic reconfigure
        Server(ControllerConfig, self.dynamic_reconfigure_callback)

        # Control message
        self.message = CarlaEgoVehicleControl()

    def dynamic_reconfigure_callback(self, config: ControllerConfig, level):
        self.MANUAL_STEER = config["manual_steer"]
        self.MANUAL_THROTTLE = config["manual_throttle"]
        self.MANUAL_OVERRIDE = config["manual_override_active"]
        self.STANLEY_OFF = config["stanley_off"]
        return config

    def update_control_message(self):
        """Update the control message based on the current state."""
        if self.MANUAL_OVERRIDE:
            self.message.reverse = self.MANUAL_THROTTLE < 0
            self.message.throttle = self.MANUAL_THROTTLE
            self.message.steer = self.MANUAL_STEER
            self.message.brake = 0.0
            self.message.hand_brake = False
        elif self.__emergency:
            self.__emergency_brake(True)
        else:
            if not self.STANLEY_OFF and self.__velocity > 5:
                steer = self._s_steer
            else:
                steer = (
                    0
                    if self.__curr_behavior in ["us_unstuck", "us_stop"]
                    else self._p_steer
                )

            self.message.reverse = self.__reverse
            self.message.throttle = self.__throttle
            self.message.brake = self.__brake
            self.message.steer = steer
            self.message.hand_brake = False
            self.message.manual_gear_shift = False

        self.message.header.stamp = rospy.get_rostime()

    def run(self):
        """Main loop of the node."""
        self.status_pub.publish(True)
        self.loginfo("VehicleController node running")

        def spin_loop(timer_event=None):
            self.loop_sleep_time = self.get_param("loop_sleep_time", 0.1)
            self.update_control_message()
            self.control_publisher.publish(self.message)
            time.sleep(self.loop_sleep_time)

        self.new_timer(self.control_loop_rate, spin_loop)
        self.spin()

    # Subscriber callbacks
    def __set_curr_behavior(self, data: String):
        self.__curr_behavior = data.data

    def __set_emergency(self, data: Bool):
        if data.data and not self.__emergency:
            self.logerr("Emergency braking engaged")
            self.__emergency = True

    def __get_velocity(self, data: CarlaSpeedometer):
        self.__velocity = data.speed
        if self.__emergency and data.speed < 0.1:
            self.__emergency_brake(False)
            for _ in range(7):
                self.emergency_pub.publish(Bool(False))
            self.loginfo("Emergency braking disengaged")

    def __set_throttle(self, data: Float32):
        self.__throttle = data.data

    def __set_brake(self, data: Float32):
        self.__brake = data.data

    def __set_reverse(self, data: Bool):
        self.__reverse = data.data

    def __set_pure_pursuit_steer(self, data: Float32):
        self._p_steer = data.data / (math.pi / 2)

    def __set_stanley_steer(self, data: Float32):
        self._s_steer = data.data / (math.pi / 2)

    def __emergency_brake(self, active: bool):
        if active:
            self.message.throttle = 0.0
            self.message.steer = 0.0
            self.message.brake = 1.0
            self.message.reverse = False
            self.message.hand_brake = True
        else:
            self.__emergency = False
            self.message.brake = 0.0
            self.message.hand_brake = False


def main(args=None):
    roscomp.init("vehicle_controller", args=args)
    try:
        node = VehicleController()
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        roscomp.shutdown()


if __name__ == "__main__":
    main()

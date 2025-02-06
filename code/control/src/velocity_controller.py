#!/usr/bin/env python
import ros_compatibility as roscomp
from carla_msgs.msg import CarlaSpeedometer
from ros_compatibility.node import CompatibleNode
from rospy import Publisher, Subscriber
from simple_pid import PID
from std_msgs.msg import Float32, Bool
import rospy

from dynamic_reconfigure.server import Server
from control.cfg import VelocityConfig


class VelocityController(CompatibleNode):
    """
    This node controls the velocity of the vehicle.
    For this it uses a PID controller.
    """

    def __init__(self):
        super(VelocityController, self).__init__("velocity_controller")
        self.loginfo("VelocityController node started")

        self.control_loop_rate = self.get_param("control_loop_rate", 0.05)
        self.role_name = self.get_param("role_name", "ego_vehicle")

        self.target_velocity_sub: Subscriber = self.new_subscription(
            Float32,
            "/paf/acting/target_velocity",
            self.__get_target_velocity,
            qos_profile=1,
        )

        self.velocity_sub: Subscriber = self.new_subscription(
            CarlaSpeedometer,
            f"/carla/{self.role_name}/Speed",
            self.__get_current_velocity,
            qos_profile=1,
        )

        self.throttle_pub: Publisher = self.new_publisher(
            Float32, f"/paf/{self.role_name}/throttle", qos_profile=1
        )

        self.brake_pub: Publisher = self.new_publisher(
            Float32, f"/paf/{self.role_name}/brake", qos_profile=1
        )

        self.reverse_pub: Publisher = self.new_publisher(
            Bool, f"/paf/{self.role_name}/reverse", qos_profile=1
        )

        self.__current_velocity: float = None
        self.__target_velocity: float = None
        self.pid_t = PID(0.60, 0.00076, 0.63)

        self.FIXED_SPEED_OVERRIDE: bool
        self.FIXED_SPEED: float

        Server(VelocityConfig, self.dynamic_reconfigure_callback)

    def dynamic_reconfigure_callback(self, config: VelocityConfig, level):
        self.pid_t.Kp = config["pid_p"]
        self.pid_t.Ki = config["pid_i"]
        self.pid_t.Kd = config["pid_d"]
        self.FIXED_SPEED = config["fixed_speed"]
        self.FIXED_SPEED_OVERRIDE = config["fixed_speed_active"]
        return config

    def run(self):
        """
        Starts the main loop of the node
        :return:
        """
        self.loginfo("VelocityController node running")
        # PID for throttle

        # since we use this for braking aswell, allow -1 to 0.
        self.pid_t.output_limits = (-1.0, 1.0)

        def loop(timer_event=None):
            """
            Calculates the result of the PID controller and publishes it.
            Never publishes values above speed limit
            (Publish speed_limit = -1 to drive without speeed limit)
            :param timer_event: Timer event from ROS
            :return:
            """
            if self.__target_velocity is None:
                self.logdebug(
                    "VelocityController hasn't received target"
                    "_velocity yet. target_velocity has been set to"
                    "default value 0"
                )
                self.__target_velocity = 0

            if self.__current_velocity is None:
                self.logdebug(
                    "VelocityController  hasn't received "
                    "current_velocity yet and can therefore not"
                    "publish a throttle value"
                )
                return

            target_velocity = (
                self.__target_velocity
                if not self.FIXED_SPEED_OVERRIDE
                else self.FIXED_SPEED
            )

            if target_velocity < 0:
                # self.logerr("VelocityController doesn't support backward "
                #             "driving yet.")
                if target_velocity == -3:
                    #  -3 is the signal for reverse driving
                    reverse = True
                    throttle = 1
                    brake = 0
                    rospy.loginfo("VelocityController: reverse driving")
                elif self.FIXED_SPEED_OVERRIDE:
                    reverse = True
                    throttle = 1
                    brake = 0
                else:
                    #  other negative values only lead to braking
                    reverse = False
                    brake = 1
                    throttle = 0

            # very low target_velocities -> stand
            elif target_velocity < 0.1:
                reverse = False
                brake = 1
                throttle = 0
            else:
                reverse = False

                v = target_velocity
                self.pid_t.setpoint = v
                throttle = self.pid_t(self.__current_velocity)
                # any throttle < 0 is used as brake signal
                if throttle < 0:
                    brake = abs(throttle)
                    throttle = 0
                else:
                    brake = 0

            self.reverse_pub.publish(reverse)
            self.brake_pub.publish(brake)
            self.throttle_pub.publish(throttle)

        self.new_timer(self.control_loop_rate, loop)
        self.spin()

    def __get_current_velocity(self, data: CarlaSpeedometer):
        self.__current_velocity = float(data.speed)

    def __get_target_velocity(self, data: Float32):
        self.__target_velocity = float(data.data)


def main(args=None):
    """
    Main function starts the node
    :param args:
    """
    roscomp.init("velocity_controller", args=args)

    try:
        node = VelocityController()
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        roscomp.shutdown()


if __name__ == "__main__":
    main()

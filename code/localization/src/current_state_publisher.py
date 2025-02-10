#!/usr/bin/env python

import rospy
from rospy import Publisher
from ros_compatibility.node import CompatibleNode
import ros_compatibility as roscomp

from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float32
from scipy.spatial.transform import Rotation as R
import tf2_ros


class CurrentStatePublisher(CompatibleNode):
    def __init__(self):
        super().__init__("current_state_publisher")
        # Parameters
        self.role_name = self.get_param("role_name", "hero")
        self.loop_rate = self.get_param("control_loop_rate", 0.05)

        # Publishes current_pos and current_heading from hero frame out of tf-graph
        self.position_publisher: Publisher = self.new_publisher(
            PoseStamped, f"/paf/{self.role_name}/current_pos", qos_profile=1
        )

        self.heading_publisher: Publisher = self.new_publisher(
            Float32, f"/paf/{self.role_name}/current_heading", qos_profile=1
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.new_timer(self.loop_rate, self.publish_heading)

    def publish_heading(self, timer_event):
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                "global",  # Target frame
                "hero",  # Source frame
                rospy.Time(0),  # Get the latest available transform
                rospy.Duration(self.loop_rate),
            )  # Timeout duration
            position = PoseStamped()
            position.header.frame_id = "global"
            position.pose.orientation = transform.transform.rotation
            position.pose.position = transform.transform.translation
            quaternion = (
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            )
            rot = R.from_quat(quaternion)
            heading = rot.as_euler("xyz", degrees=False)[2]

            self.position_publisher.publish(position)
            self.heading_publisher.publish(Float32(data=heading))
        except Exception as ex:
            self.loginfo(ex)

    def run(self):
        self.loginfo("Started CurrentStatePublisher Node!")
        self.spin()


def main(args=None):
    """
    Main function starts the node
    :param args:
    """
    roscomp.init("current_state_publisher", args=args)

    try:
        node = CurrentStatePublisher()
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        roscomp.shutdown()


if __name__ == "__main__":
    main()

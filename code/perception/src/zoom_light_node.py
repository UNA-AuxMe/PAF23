#!/usr/bin/env python3

from datetime import datetime
import threading
from time import sleep
from ros_compatibility.node import CompatibleNode
import ros_compatibility as roscomp
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image as ImageMsg
from perception.msg import TrafficLightState
from std_msgs.msg import Int16
from cv_bridge import CvBridge
from traffic_light_detection.src.traffic_light_detection.traffic_light_inference import (  # noqa: E501
    TrafficLightInference,
)
import cv2
import numpy as np
import rospy


class ZoomLightNode(CompatibleNode):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        # general setup
        self.bridge = CvBridge()
        self.role_name = self.get_param("role_name", "hero")
        self.side = self.get_param("side", "Zoom")
        print("------------------------------------------------------------------")
        print(f"zoom_light_node.py -> self.side: {self.side}")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        self.classifier = TrafficLightInference(self.get_param("model", ""))
        self.last_info_time: datetime = None
        self.last_state = None
        threading.Thread(target=self.auto_invalidate_state).start()

        # publish / subscribe setup
        self.setup_camera_subscriptions()
        self.setup_traffic_light_publishers()

        self.counter = 0

    def setup_camera_subscriptions(self):
        self.new_subscription(
            msg_type=numpy_msg(ImageMsg),
            callback=self.handle_camera_image,
            topic=f"/paf/{self.role_name}/{self.side}/segmented_traffic_light",
            qos_profile=1,
        )

    def setup_traffic_light_publishers(self):
        self.traffic_light_publisher = self.new_publisher(
            msg_type=TrafficLightState,
            topic=f"/paf/{self.role_name}/{self.side}/zoom_traffic_light_state",
            qos_profile=1,
        )
        self.traffic_light_distance_publisher = self.new_publisher(
            msg_type=Int16,
            topic=f"/paf/{self.role_name}/{self.side}"
            + "/zoom_traffic_light_y_distance",
            qos_profile=1,
        )

    def auto_invalidate_state(self):
        while True:
            sleep(1)

            if self.last_info_time is None:
                continue

            if (datetime.now() - self.last_info_time).total_seconds() >= 2:
                msg = TrafficLightState()
                msg.state = 0
                self.traffic_light_publisher.publish(msg)
                self.traffic_light_distance_publisher.publish(Int16(0))
                self.last_info_time = None

    def handle_camera_image(self, image):
        self.counter += 1
        rospy.logerr(f"Erkannte Ampeln: {self.counter} - ZoomLightNode")
        distance = int(image.header.frame_id)
        print("something happens in handel_camera_image - ZoomLightNode")
        cv2_image = self.bridge.imgmsg_to_cv2(image)
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        result, data = self.classifier(cv2_image)

        if (
            data[0][0] > 1e-15
            and data[0][3] > 1e-15
            or data[0][0] > 1e-10
            or data[0][3] > 1e-10
        ):
            print("returned via if-statement ------------------")
            return  # too uncertain, may not be a traffic light

        if not is_front(rgb_image):
            print("not front facing +++++++++++++++++++++++++")
            return  # not a front facing traffic light

        state = result if result in [1, 2, 4] else 0
        print(f"traffic_light_state: {state} - ZoomLightNode")
        if self.last_state == state:
            # 1: Green, 2: Red, 4: Yellow, 0: Unknown
            msg = TrafficLightState()
            msg.state = state
            self.traffic_light_publisher.publish(msg)

            if distance is not None:
                self.traffic_light_distance_publisher.publish(Int16(distance))
        else:
            self.last_state = state

        # Automatically invalidates state (state=0) in auto_invalidate_state()
        if state != 0:
            self.last_info_time = datetime.now()

    def run(self):
        self.spin()


def get_light_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for the hue, saturation, and value
    # - Red and Yellow combined since they are so close in the color spectrum
    lower_red_yellow = np.array([0, 75, 100])
    upper_red_yellow = np.array([40, 255, 255])
    lower_green = np.array([40, 200, 200])
    upper_green = np.array([80, 255, 255])

    # Mask where the pixels within the bounds are white, otherwise black
    m1 = cv2.inRange(hsv, lower_red_yellow, upper_red_yellow)
    m2 = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_or(m1, m2)

    return mask


def is_front(image):
    mask = get_light_mask(image)

    # Find contours in the thresholded image, use only the largest one
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    contour = contours[0] if contours else None

    if contour is None:
        return False

    _, _, width, height = cv2.boundingRect(contour)
    aspect_ratio = width / height

    # If aspect ratio is within range of a square (therefore a circle)
    if 0.75 <= aspect_ratio <= 1.3:
        return True
    else:
        return False


if __name__ == "__main__":
    roscomp.init("ZoomLightNode")
    node = ZoomLightNode("ZoomLightNode")
    node.run()

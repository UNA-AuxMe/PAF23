#!/usr/bin/env python3

from ros_compatibility.node import CompatibleNode
import ros_compatibility as roscomp
from sklearn.cluster import DBSCAN
import torch
import cv2
from vision_node_helper import coco_to_carla, carla_colors
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from torchvision.utils import draw_segmentation_masks
import numpy as np
from ultralytics import YOLO
import rospy
from ultralytics.utils.ops import scale_masks
from mapping.msg import ClusteredPointsArray
from perception_utils import array_to_clustered_points

from time import time_ns, sleep
from copy import deepcopy
import asyncio


class ZoomNode(CompatibleNode):
    """
    VisionNode:

    The Vision-Node provides advanced object-detection features.
    It can handle different camera angles, easily switch between
    pretrained models and distances of objects.

    Advanced features are limited to ultralytics models and center view.
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        # dictionary of pretrained models
        self.model_dict = {
            "yolo11n-seg": (YOLO, "yolo11s-seg.pt", "segmentation", "ultralytics"),
        }

        # general setup
        self.bridge = CvBridge()
        self.role_name = self.get_param("role_name", "hero")
        self.view_camera = self.get_param("view_camera")
        self.camera_resolution = self.get_param("camera_resolution")

        self.depth_images = []
        self.dist_array = None
        self.lidar_array = None

        self.setup_subscriber()
        self.setup_publisher()
        self.setup_model()
        rospy.logfatal("ZoomNode alive")

    def setup_subscriber(self):

        self.new_subscription(
            msg_type=numpy_msg(ImageMsg),
            callback=self.handle_zoom_image,
            topic=f"/carla/{self.role_name}/Zoom/image",
            qos_profile=1,
        )

    def setup_publisher(self):
        """
        sets up all publishers for the Vision-Node
        """

        self.traffic_light_publisher = self.new_publisher(
            msg_type=numpy_msg(ImageMsg),
            topic=f"/paf/{self.role_name}/Zoom/segmented_traffic_light",
            qos_profile=1,
        )

    def setup_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_info = self.model_dict[self.get_param("model")]
        self.model = model_info[0]
        self.weights = model_info[1]
        self.type = model_info[2]
        self.framework = model_info[3]
        self.save = True

        print("Zoom Node Configuration:")
        print("Device -> ", self.device)
        print(f"Model -> {self.get_param('model')},")
        print(f"Type -> {self.type}, Framework -> {self.framework}")

        sleep(5)
        if self.framework == "ultralytics":
            self.model = self.model(self.weights)
            rospy.logfatal("Modell gestartet - ZoomNode")
            print("ZoomNode - Ultralyt Modell läuft")
        else:
            rospy.logerr("Framework not supported")

    def handle_zoom_image(self, image):
        """
        This function handles the image of the zoom camera and calls the
        process_traffic_light function when one has been recognized

        Args:
            image (image msg): Image from camera scubscription
        """

        rospy.logfatal("handle_zoom_image - ZoomNode")
        # free up cuda memory
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # vision_result =
        self.predict_ultralytics(
            image, return_image=self.view_camera, image_size=self.camera_resolution
        )

    def predict_ultralytics(self, image, return_image=True, image_size=640):
        """
        This function takes in an image from a camera, predicts
        an ultralytics model on the image and looks for lidar points
        in the bounding boxes.

        This function also implements a visualization
        of what has been calculated for RViz.

        Args:
            image (image msg): image from camera subsription

        Returns:
            (cv image): visualization output for rvizw
        """
        rospy.logfatal("predict_ultralytics - ZoomNode")
        scaled_masks = None
        # preprocess image
        cv_image = self.bridge.imgmsg_to_cv2(
            img_msg=image, desired_encoding="passthrough"
        )
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        # run model prediction

        output = self.model.track(
            cv_image, half=True, verbose=False, imgsz=image_size  # type: ignore
        )
        if not (
            hasattr(output[0], "masks")
            and output[0].masks is not None
            and hasattr(output[0], "boxes")
            and output[0].boxes is not None
            and self.dist_array is not None
            and self.lidar_array is not None
        ):
            return None

        if image.header.frame_id == "hero/Zoom":
            rospy.logfatal("in zoom if")
            if 9 in output[0].boxes.cls:
                asyncio.run(
                    self.process_traffic_lights(
                        output[0], cv_image, deepcopy(image.header)
                    )
                )
            return

    def publish_distance_output(self, points, carla_classes):
        """
        Publishes the distance output of the object detection
        """
        distance_output = np.column_stack(
            (carla_classes, points[:, 0], points[:, 1])
        ).ravel()

        if distance_output.size > 0:
            self.distance_publisher.publish(Float32MultiArray(data=distance_output))

    def calculate_depth_values(self, dist_array):
        """
        Berechnet die Tiefenwerte basierend auf den Lidar-Daten
        """
        abs_distance = np.sqrt(
            dist_array[..., 0] ** 2 + dist_array[..., 1] ** 2 + dist_array[..., 2] ** 2
        )
        return abs_distance

    async def process_traffic_lights(self, prediction, cv_image, image_header):
        indices = (prediction.boxes.cls == 9).nonzero().squeeze().cpu().numpy()
        indices = np.asarray([indices]) if indices.size == 1 else indices
        max_y = 360  # middle of image
        min_prob = 0.030

        for index in indices:
            box = prediction.boxes.cpu().data.numpy()[index]
            print(
                f"box 0, 1, 2, 3, 4: {box[0]}, {box[1]}, {box[2]}, {box[3]}, {box[4]}"
            )
            # if box[4] < min_prob:
            #     continue

            # if (box[2] - box[0]) * 1.5 > box[3] - box[1]:
            #     continue  # ignore horizontal boxes

            # if box[1] > max_y:
            #     continue

            box = box[0:4].astype(int)
            segmented = cv_image[box[1] : box[3], box[0] : box[2]]

            traffic_light_y_distance = box[1]

            traffic_light_image = self.bridge.cv2_to_imgmsg(segmented, encoding="rgb8")
            traffic_light_image.header = image_header
            traffic_light_image.header.frame_id = str(traffic_light_y_distance)
            rospy.logfatal("publish traffic_light_image")
            self.traffic_light_publisher.publish(traffic_light_image)

    def run(self):
        self.spin()
        pass


if __name__ == "__main__":
    roscomp.init("ZoomNode")
    node = ZoomNode("ZoomNode")
    node.run()

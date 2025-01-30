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

from time import time_ns
from copy import deepcopy
import asyncio


class VisionNode(CompatibleNode):
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
            "yolov8x-seg": (YOLO, "yolov8x-seg.pt", "segmentation", "ultralytics"),
            "yolo11n-seg": (YOLO, "yolo11n-seg.pt", "segmentation", "ultralytics"),
            "yolo11s-seg": (YOLO, "yolo11s-seg.pt", "segmentation", "ultralytics"),
            "yolo11m-seg": (YOLO, "yolo11m-seg.pt", "segmentation", "ultralytics"),
            "yolo11l-seg": (YOLO, "yolo11l-seg.pt", "segmentation", "ultralytics"),
            "yolo11x-seg": (YOLO, "yolo11l-seg.pt", "segmentation", "ultralytics"),
        }

        self.timesum = 0
        self.timenum = 0
        self.timestart = False

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

    def setup_subscriber(self):
        self.new_subscription(
            msg_type=numpy_msg(ImageMsg),
            callback=self.handle_camera_image,
            topic=f"/carla/{self.role_name}/Center/image",
            qos_profile=1,
        )

        self.new_subscription(
            msg_type=numpy_msg(ImageMsg),
            callback=self.handle_dist_array,
            topic="/paf/hero/Center/dist_array",
            qos_profile=1,
        )

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

        self.pointcloud_publisher = self.new_publisher(
            msg_type=numpy_msg(ClusteredPointsArray),
            topic=f"/paf/{self.role_name}/visualization_pointcloud",
            qos_profile=1,
        )

        self.publisher_center = self.new_publisher(
            msg_type=numpy_msg(ImageMsg),
            topic=f"/paf/{self.role_name}/Center/segmented_image",
            qos_profile=1,
        )

        self.distance_publisher = self.new_publisher(
            msg_type=Float32MultiArray,
            topic=f"/paf/{self.role_name}/Center/object_distance",
            qos_profile=1,
        )

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

        print("Vision Node Configuration:")
        print("Device -> ", self.device)
        print(f"Model -> {self.get_param('model')},")
        print(f"Type -> {self.type}, Framework -> {self.framework}")

        if self.framework == "ultralytics":
            self.model = self.model(self.weights)
        else:
            rospy.logerr("Framework not supported")

    def handle_camera_image(self, image):
        """
        This function handles a new camera image and publishes the
        calculated visualization according to the correct camera angle

        Args:
            image (image msg): Image from camera scubscription
        """

        # free up cuda memory
        if self.device == "cuda":
            torch.cuda.empty_cache()

        vision_result = self.predict_ultralytics(
            image, return_image=self.view_camera, image_size=self.camera_resolution
        )

        if vision_result is None:
            return

        # publish vision result to rviz
        img_msg = self.bridge.cv2_to_imgmsg(vision_result, encoding="bgr8")
        img_msg.header = image.header
        self.publisher_center.publish(img_msg)

    def handle_zoom_image(self, image):
        """
        This function handles a new camera image and publishes the
        calculated visualization according to the correct camera angle

        Args:
            image (image msg): Image from camera scubscription
        """

        # free up cuda memory
        if self.device == "cuda":
            torch.cuda.empty_cache()

        self.predict_ultralytics(image, return_image=self.view_camera, image_size=720)

    def handle_dist_array(self, dist_array):
        """
        This function overwrites the current lidar depth image from
        the lidar distance node with the latest depth image.
        The function also calculates the depth values of the lidar

        Args:
            dist_array (image msg): Depth image frim Lidar Distance Node
        """
        # callback function for lidar depth image
        # since frequency is lower than image frequency
        # the latest lidar image is saved
        lidar_array = self.bridge.imgmsg_to_cv2(
            img_msg=dist_array, desired_encoding="passthrough"
        )
        lidar_array_copy = np.copy(lidar_array)
        # add camera height to the z-axis
        lidar_array_copy[..., 2] += 1.7
        self.lidar_array = lidar_array_copy
        self.dist_array = self.calculate_depth_values(lidar_array_copy)

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

        scaled_masks = None
        # preprocess image
        cv_image = self.bridge.imgmsg_to_cv2(
            img_msg=image, desired_encoding="passthrough"
        )
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        # run model prediction

        output = self.model(
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

        carla_classes = np.array(coco_to_carla)[
            output[0].boxes.cls.to(torch.int).cpu().numpy()  # type: ignore
        ]

        masks = torch.tensor(output[0].masks.data)
        scaled_masks = scale_masks(
            masks.unsqueeze(1), cv_image.shape[:2], True
        ).squeeze(1)
        valid_points, class_indices = self.process_segmentation_mask(
            scaled_masks.cpu().numpy(),
            carla_classes=carla_classes,
            distance_array=self.dist_array,
            lidar_array=self.lidar_array,
        )
        if valid_points is None or valid_points.size == 0:
            return None
        clustered_points, cluster_indices, carla_classes_indices = self.cluster_points(
            valid_points, class_indices, carla_classes
        )
        if clustered_points is None or clustered_points.size == 0:
            return None
        try:
            self.publish_distance_output(clustered_points, carla_classes_indices)
            clustered_lidar_points_msg = array_to_clustered_points(
                clustered_points,
                cluster_indices,
                object_class_array=carla_classes_indices,
            )
            self.pointcloud_publisher.publish(clustered_lidar_points_msg)
        except Exception as e:
            rospy.logerr(f"Error in publishing pointcloud: {e}")
            return None
        # proceed with traffic light detection
        # if 9 in output[0].boxes.cls:
        #     asyncio.run(
        #         self.process_traffic_lights(output[0], cv_image, deepcopy(image.header))
        #     )

        if return_image is False or scaled_masks is None:
            return None

        transposed_image = np.transpose(cv_image, (2, 0, 1))
        image_tensor = torch.tensor(transposed_image, dtype=torch.uint8)
        masks_tensor = torch.tensor(scaled_masks > 0, dtype=torch.bool)

        class_colors = np.array(carla_colors)[carla_classes].tolist()
        drawn_images = draw_segmentation_masks(
            image_tensor, masks_tensor, alpha=0.6, colors=class_colors
        )

        np_image = drawn_images.permute(1, 2, 0).cpu().numpy()
        return cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

    def process_segmentation_mask(
        self, segmentation_array, carla_classes, distance_array, lidar_array
    ):
        # Only process the segmentation mask if the distance array is not None
        valid_distances_mask = distance_array > 0
        car_length = 4.9
        car_width = 1.86436
        # Filter out points that are not in the car and not on the road and are not zero
        car_filter_mask = (
            # filter out points that are not in the car
            (lidar_array[..., 0] >= -car_length / 2)
            & (lidar_array[..., 0] <= car_length / 2)
            & (lidar_array[..., 1] >= -car_width / 2)
            & (lidar_array[..., 1] <= car_width / 2)
        )
        # Filter out points that are on the road
        road_filter_mask = lidar_array[..., 2] >= 0.3
        # Filter out points that are zero
        zero_filter_mask = (
            ~(lidar_array[..., 0] == 0.0)
            & ~(lidar_array[..., 1] == 0.0)
            & ~(lidar_array[..., 2] == 1.7)
        )
        lidar_filter_mask = ~car_filter_mask & road_filter_mask & zero_filter_mask
        combined_mask = valid_distances_mask & lidar_filter_mask
        # tiled_mask holds all the valid points
        valid_points_from_mask = (
            segmentation_array.astype(bool) & combined_mask[None, ...]
        )
        # get the x, y, z values of the valid points
        valid_indices = np.nonzero(valid_points_from_mask)
        valid_points = lidar_array[valid_indices[1], valid_indices[2]]
        return valid_points, valid_indices[0]

    def cluster_points(
        self, points, class_indices, carla_classes, eps=0.5, min_samples=2
    ):
        """
        Clusters all points in the point cloud and determines the largest cluster for
        each segmentation class in one pass.

        Parameters:
            points (numpy structured array): Array of points with fields 'x', 'y', 'z'.
            class_indices (numpy array): Array of segmentation mask indices for each
                point
            eps (float): Maximum distance between points to be considered in the same
                neighborhood
            min_samples (int): Minimum number of points to form a dense region (cluster)

        Returns:
            clustered_points (numpy structured array): Points belonging to the largest
                cluster for each class index.
            valid_labels (numpy array): Labels corresponding to each class index (one
                per class).
            valid_class_indices (numpy array): Class indices corresponding to the
                returned points.
        """
        if points.size == 0:
            return np.array([], dtype=points.dtype), [], []

        # Apply DBSCAN clustering to all points at once
        db = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = db.fit_predict(points)

        # Combine class indices and cluster labels to identify unique groups
        combined_labels = np.vstack((class_indices, cluster_labels)).T

        # Ignore noise points (cluster_label == -1)
        valid_mask = cluster_labels != -1
        valid_points = points[valid_mask]
        valid_combined_labels = combined_labels[valid_mask]
        valid_class_indices = class_indices[valid_mask]

        # Find the largest cluster for each class
        unique_combinations, inverse_indices = np.unique(
            valid_combined_labels, axis=0, return_inverse=True
        )
        counts = np.bincount(inverse_indices)

        # Map the largest cluster for each class
        largest_clusters = {}
        for idx, (class_idx, _) in enumerate(unique_combinations):
            if (
                class_idx not in largest_clusters
                or counts[idx] > counts[largest_clusters[class_idx]]
            ):
                largest_clusters[class_idx] = idx

        # Extract points belonging to the largest cluster for each class
        selected_points_mask = np.isin(
            inverse_indices,
            [largest_clusters[class_idx] for class_idx in largest_clusters],
        )
        clustered_points = valid_points[selected_points_mask]

        return (
            clustered_points,
            valid_class_indices[selected_points_mask],
            carla_classes[valid_class_indices[selected_points_mask]],
        )

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
            self.traffic_light_publisher.publish(traffic_light_image)

    def run(self):
        self.spin()
        pass


if __name__ == "__main__":
    roscomp.init("VisionNode")
    node = VisionNode("VisionNode")
    node.run()

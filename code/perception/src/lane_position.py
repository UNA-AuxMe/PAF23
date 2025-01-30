#!/usr/bin/env python
# ROS imports
import rospy
import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from sensor_msgs.msg import Image as ImageMsg
from rospy.numpy_msg import numpy_msg
from cv_bridge import CvBridge

# intermediate Layer imports
from mapping_common.shape import Rectangle
from mapping_common.entity import Lanemarking, Flags
from mapping_common.transform import Transform2D, Vector2
from mapping_common.map import Map
from mapping.msg import Map as MapMsg

# clustering imports
import numpy as np
from sklearn.cluster import DBSCAN
import cv2


class lane_position(CompatibleNode):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.bridge = CvBridge()
        self.dist_arrays = []

        # get parameters from launch file
        self.line_length = self.get_param(
            "line_length"
        )  # predefined length of the lanemarkings
        self.line_width = self.get_param("line_width")  # width of the lanemarkings
        self.epsilon = self.get_param(
            "epsilon_clustering"
        )  # epsilon for clustering algorithm
        self.min_samples = self.get_param(
            "min_samples_clustering"
        )  # min samples for clustering

        # confidence parameters:
        # weights for confidence calculation
        self.angle_weight = self.get_param("angle_weight")
        self.size_weight = self.get_param("size_weight")
        self.std_dev_weight = self.get_param("std_dev_weight")
        self.angle_normalization = self.get_param(
            "angle_normalization"
        )  # max acceptable angle for normalization
        self.size_normalization = self.get_param(
            "size_normalization"
        )  # max acceptable cluster size for normalization
        self.std_dev_normalization = self.get_param(
            "std_dev_normalization"
        )  # max acceptable standard deviation in linearregression for normalization
        self.angle_prediction_threshold = self.get_param("angle_prediction_threshold")
        self.confidence_threshold = self.get_param("confidence_threshold")

        self.y_tolerance = self.get_param(
            "y_tolerance"
        )  # min distance that lanemarkings have to have, new lanemarkings within this
        # distance are ignored

        self.setup_subscriptions()
        self.setup_publishers()

    def run(self):
        self.spin()
        pass

    def setup_subscriptions(self):
        """
        sets up a subscriber to the lanemask
        """

        self.new_subscription(
            msg_type=numpy_msg(ImageMsg),
            callback=self.lanemask_handler,
            topic="/paf/hero/Center/lane_mask",
            qos_profile=1,
        )

        """
        sets up a subscription to the lidar
        depth image of the selected camera angle
        """

        self.new_subscription(
            msg_type=numpy_msg(ImageMsg),
            callback=self.distance_array_handler,
            topic="/paf/hero/Center/dist_array",
            qos_profile=1,
        )

    def setup_publishers(self):
        """sets up a publisher for the visulization of the preprocessed lanemask
        (not necessary for functionality)
        topic: /Lane/label_image
        """
        self.label_image_publisher = self.new_publisher(
            msg_type=numpy_msg(ImageMsg),
            topic="/paf/hero/Lane/label_image",
            qos_profile=1,
        )
        """sets up a publisher for the map with lanemarking entities
        topic: /paf/hero/mapping/init_lanemarkings
        """
        self.map_publisher = self.new_publisher(
            msg_type=MapMsg,
            topic=self.get_param(
                "~map_init_topic", "/paf/hero/mapping/init_lanemarkings"
            ),
            qos_profile=1,
        )

    def lanemask_handler(self, ImageMsg):
        """main function that converts the lanemask to lanemarking entities
        and publishes them"""

        lanemask = self.bridge.imgmsg_to_cv2(img_msg=ImageMsg, desired_encoding="8UC1")
        stamp = ImageMsg.header.stamp
        lanemask = self.remove_horizontal_lines(lanemask)
        positions, angles, confidences = self.process_lanemask(
            lanemask,
        )
        lanemarkings = self.lanemarking_from_coordinates(
            positions, angles, confidences, stamp, predicted=False
        )
        self.publish_Lanemarkings_map(lanemarkings)

    def remove_horizontal_lines(self, lanemask):
        """removes horizontal lines in the mask,
        so markings for parking spots are ignored in the lane detection

        Args:
            lanemask

        Returns:
            preprocessed lanemask
        """
        image = np.array(lanemask)
        # define kernel for horizontal lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))

        # detect horizontal line in the mask
        horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        # Eremove horizontal lines from the mask
        updated_mask = image - horizontal_lines

        ros_image = self.bridge.cv2_to_imgmsg(updated_mask)
        self.label_image_publisher.publish(ros_image)
        return updated_mask

    def process_lanemask(self, lanemask):
        """masks lidar points with given lanemask, then clusters the remaining lidar
        points, calculates the center and angle for each cluster
        and calculates the confidences

        Args:
            lanemask

        Returns:
            position, angle and confidence of each lanemarking
        """
        points = self.mask_lidarpoints(lanemask)
        clusters = self.cluster_points(points)
        positions, angles, deviations = self.get_lanemarking_position_and_angle(
            clusters
        )
        confidences = self.determine_confidences(angles, clusters, deviations)
        return positions, angles, confidences

    def lanemarking_from_coordinates(
        self, positions, angles, confidences, stamp, predicted=False
    ):
        """creates a lanemarking entity for the position, angle, confidence,
        timespamp and he predicted flag

        Args:
            positions (_type_): x,ycoordinates of the center
            angles (_type_):angle of the lanemarking
            confidences (_type_): confidence
            stamp (_type_): timestamp
            predicted (bool, optional): is predicted or not. Defaults to False.

        Returns:
           entities: Lanemarking entities
        """
        if positions is None:
            return []
        entities = []
        position_indices = self.calc_position_indices(positions)
        for position, position_index, angle, confidence in zip(
            positions, position_indices, angles, confidences
        ):
            if confidence < self.confidence_threshold:
                continue
            x, y = position
            transform = Transform2D.new_rotation_translation(angle, Vector2.new(x, y))
            shape = Rectangle(
                width=self.line_width, length=self.calc_line_length(x, angle)
            )  # move to ros param
            style = Lanemarking.Style.SOLID
            flags = Flags(is_lanemark=True)
            timestamp = stamp
            lanemarking = Lanemarking(
                style=style,
                confidence=confidence,
                priority=1,
                shape=shape,
                transform=transform,
                timestamp=timestamp,
                flags=flags,
                predicted=predicted,
                position_index=int(position_index),
            )
            entities.append(lanemarking)
        return entities

    def calc_line_length(self, x, angle):
        """
        Calculates the total length of a line that is symmetric around (x, y)
        and ends at the y-axis.
        Args:
        x: x-coordinate of the midpoint
        angle: angle of the line in degrees
        Returns: total length of the line
        """

        # Convert angle from degrees to radians
        # theta = np.deg2rad(angle)

        # Calculate the total length of the line
        if np.isclose(np.cos(angle), 0):
            length = self.line_length
        else:
            line_length = abs(x / np.cos(angle)) * 2
            length = max(line_length, self.line_length)
        return length

    def calc_position_indices(self, points):
        """gives each lanemarking a unique index, where 1 is the lane next to the car
        on the left and -1 on the right. higher numbers indicate, that the marking is
        for the next or further lane. E.g. indices = [2, 1, -1, -2]

        Args:
            points: arrays with coordinates of the centers of the lanemarkings

        Returns:
            array with labels for the lanemarkings in the same order as the input array
        """
        if len(points) < 1:
            return [0]
        # Extrahieren der y-Koordinaten
        y_coords = np.array(points)[:, 1]

        # Positive und negative y-Koordinaten trennen
        positive_indices = np.where(y_coords > 0)[0]
        negative_indices = np.where(y_coords < 0)[0]

        # Positive und negative Werte nach Abstand zu 0 sortieren
        sorted_positive_indices = positive_indices[
            np.argsort(y_coords[positive_indices])
        ]
        sorted_negative_indices = negative_indices[
            np.argsort(-y_coords[negative_indices])
        ]

        # Indizes initialisieren
        indices = np.zeros(len(y_coords), dtype=int)

        # Positive Indizes zuweisen
        for rank, idx in enumerate(sorted_positive_indices, start=1):
            indices[idx] = rank

        # Negative Indizes zuweisen
        for rank, idx in enumerate(sorted_negative_indices, start=1):
            indices[idx] = -rank
        return indices

    def publish_Lanemarkings_map(self, Lanemarkings):
        """creates and publishes a Map with all found lanemarking Entities

        Args:
            Lanemarkings: all lanemarking entities
        """
        if Lanemarkings is None:
            return

        stamp = rospy.get_rostime()
        map = Map(timestamp=stamp, entities=Lanemarkings)
        msg = map.to_ros_msg()
        self.map_publisher.publish(msg)

    def determine_confidences(self, angles, cluster_dict, linear_deviations):
        """determines a confidence for every found lanemarking.
        confidence score consists of:
            - Median Angle of the lanemarking to the others,
            - cluster size
            - deviation of the linear regression

        Args:
            angles np.array(rad): all angles between lane marking and car
            cluster_dict: dictoinary with all clusters
            deviations (float): the standard deviations
            of the linear regression algorithm

        Returns:
            array with confidence scores
        """
        confidences = []
        # Use map to get the lengths of the clusters and convert it to a list
        cluster_sizes = list(map(lambda item: len(item[1]), cluster_dict.items()))
        median_angle_deviations = self.get_median_angle(angles)
        # calculate the confidence for each lanemarking
        for angle, cluster_size, deviation in zip(
            median_angle_deviations, cluster_sizes, linear_deviations
        ):
            # look for strong confidence violations
            if (
                deviation > self.std_dev_normalization * 2 / 3
                or cluster_size < self.size_normalization / 3
                or angle > self.angle_normalization * 2 / 3
            ):
                confidence = 0
            else:
                # normalize all values so the maximum confidence is 1
                normalized_angle = max(
                    0, 1 - (abs(angle) / np.deg2rad(self.angle_normalization))
                )
                normalized_size = min(1, cluster_size / self.size_normalization)
                normalized_std_dev = max(
                    0, 1 - (deviation / self.std_dev_normalization)
                )
                confidence = (
                    self.angle_weight * normalized_angle
                    + self.size_weight * normalized_size
                    + self.std_dev_weight * normalized_std_dev
                )
            confidences.append(confidence)

        return confidences

    def get_median_angle(self, angles):
        """calculates the median angle for every angle to the others

        Args:
            angles : array with angles in radiant
        Returns:
            median_angle_deviations: array with median deviations
        """
        angles = np.array(angles)
        # calculate the median deviation to other lanemarkings
        angle_diff = np.abs(angles[:, None] - angles)
        # Set diagonal to np.nan, to ignore it while calculating the median
        np.fill_diagonal(angle_diff, np.nan)
        median_angle_deviations = np.nanmedian(angle_diff, axis=1)
        return median_angle_deviations

    def mask_lidarpoints(self, lanemask):
        """masks lidar points with the lanemask and removes all points,
        that aren't close to the ground

        Args:
            lanemask

        Returns:
            filtered lidar points [[x,y,z][x,y,z],...]
        """
        if len(self.dist_arrays) == 0:
            return []
        points = self.dist_arrays[lanemask == 255]
        points = points[~np.all(points == [0.0, 0.0, 0.0], axis=1)]
        filtered_points = points[np.abs(points[:, 2] - (-1.7)) <= 0.15]
        return filtered_points

    def distance_array_handler(self, ImageMsg):
        dist_array = self.bridge.imgmsg_to_cv2(
            img_msg=ImageMsg, desired_encoding="passthrough"
        )
        self.dist_arrays = dist_array

    def get_lanemarking_position_and_angle(self, clusters):
        """Processes clusters and calculates line fitting for each cluster
        to extract lane markings."""
        center_coordinates = []
        angles = []
        deviations = []

        # Iterate through each cluster in the dictionary
        for label, cluster in clusters.items():
            # skip outliers
            if label == "outliers":
                continue

            y = cluster[:, 1]  # y-coordinates
            x = cluster[:, 0]  # x-coordinates

            # Skip clusters with fewer than 3 points
            if len(x) < 3:
                continue

            # Fit a line (linear regression) to the points in the cluster
            p, res = np.polyfit(x, y, deg=1, cov=True)

            # Calculate the standard deviation of the fit
            if np.ndim(res) == 2:
                std_dev = np.sqrt(
                    np.diag(res)
                )  # Standard deviation from the covariance matrix
            else:
                std_dev = np.sqrt(
                    res
                )  # If covariance is a vector, take the square root

            # Get the slope (m) of the line
            m = p[0]
            theta = np.arctan(m)

            # Calculate the center of the cluster (mean of x and y coordinates)
            center_y = np.mean(y)
            center_x = np.mean(x)

            # Check if this y-coordinate is within the tolerance of an existing
            # lanemarking
            if any(
                abs(center_y - existing_y) <= self.y_tolerance
                for _, existing_y in center_coordinates
            ):
                continue  # Skip this cluster if it is too close to an existing marking
            # Calculate the angle of the line relative to the x-axis (in radians)

            deviations.append(std_dev[0])  # Standard deviation of the slope
            # Store the results for lanemarkings, angles, and deviations
            angles.append(theta)
            center_coordinates.append([center_x, center_y])

        return center_coordinates, angles, deviations

    def cluster_points(self, points):
        """clusters the lidar points with the DBSCAN algorithm
        to separate the points according to lanemarkings

        Args:
            points: lidar points

        Returns:
            clusters: dict with all clusters (labels as keys)
        """
        labels = []
        try:
            clustering = DBSCAN(
                eps=self.epsilon, min_samples=self.min_samples, algorithm="ball_tree"
            ).fit(
                points
            )  # HDBSCAN
            labels = clustering.labels_
        except Exception as e:
            rospy.logwarn(
                f"could not cluster points for lane position calculation: {str(e)}"
            )

        # Find the unique cluster labels (excluding -1 for outliers)
        unique_labels = np.unique(labels)

        # Dictionary to store clusters
        clusters = {}

        # Group points by their labels
        for label in unique_labels:
            # Extract all points that belong to the current label
            cluster_points = points[labels == label]

            # If the cluster is not the outlier cluster (-1), store it
            if label != -1:
                clusters[label] = cluster_points
            else:
                clusters["outliers"] = cluster_points  # Store outliers separately

        return clusters

    # currently not used. Needs improvements in future
    def predict_lanemarkings(self, centers, angles, confidences, stamp):
        """predicts new lanemarkings if a lanemarking angle is too steep
        or a lanemarking is found close to the car,
        but the according marking on the other side was not detected

        Args:
            centers (_type_): array with all lanemarking centers
            angles (_type_): array with all lanemarking angles
            confidences (_type_): array with all lanemarking confidences
            stamp (_type_): timestamp of recieved lanemask

        Returns:
            predicted_lanemarkings: predicted lanemarking entities
        """
        new_centers = []
        new_angles = []
        new_confidences = []
        x_coords, y_coords = np.array(centers).T
        closest_lane_index = np.argmin(np.abs(y_coords))
        median_angle_deviations = self.get_median_angle(angles)
        for center, angle, angle_deviation, confidence in zip(
            centers, angles, median_angle_deviations, confidences
        ):
            x, y = center
            if np.abs(angle_deviation) > self.angle_prediction_threshold:
                new_centers.append(center)
                new_angles.append(angle - angle_deviation)
                new_confidences.append(confidence)
            if y == y_coords[closest_lane_index]:
                matching_indices = np.where(
                    (y_coords >= -y - 0.5) & (y_coords <= -y + 0.5)
                )[0]
                if len(matching_indices) == 0:
                    new_centers.append([x, -y])
                    new_angles.append(angle)
                    new_confidences.append(confidence)

        predicted_lanemarkings = self.lanemarking_from_coordinates(
            new_centers, new_angles, new_confidences, stamp, predicted=True
        )
        return predicted_lanemarkings


if __name__ == "__main__":
    roscomp.init("lane_position_node")
    node = lane_position("lane_position_node")
    node.run()

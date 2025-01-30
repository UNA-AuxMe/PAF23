#!/usr/bin/env python
from joblib import Parallel, delayed
import rospy
import ros_numpy
import numpy as np
from lidar_filter_utility import bounding_box, remove_field_name
from sensor_msgs.msg import PointCloud2, Image as ImageMsg
from sklearn.cluster import DBSCAN
from cv_bridge import CvBridge
from tf.transformations import quaternion_from_matrix
from visualization_msgs.msg import Marker, MarkerArray

# from mpl_toolkits.mplot3d import Axes3D
# from itertools import combinations
# from matplotlib.colors import LinearSegmentedColormap


class LidarDistance:
    """See doc/perception/lidar_distance_utility.md on
    how to configute this node
    """

    def __init__(self):
        self.cluster_buffer = []

    def callback(self, data):
        """
        Callback function that processes LiDAR point cloud data.

        Executes clustering and image calculations for the provided point cloud.

        :param data: LiDAR point cloud as a ROS PointCloud2 message.
        """
        self.start_clustering(data)
        self.start_image_calculation(data)

    def listener(self):
        """
        Initializes the ROS node, creates publishers/subscribers, and keeps it active.
        """
        rospy.init_node("lidar_distance")
        self.bridge = CvBridge()  # OpenCV bridge for image conversions

        # Publisher for filtered point clouds
        self.pub_pointcloud = rospy.Publisher(
            rospy.get_param(
                "~point_cloud_topic",
                "/carla/hero/" + rospy.get_namespace() + "_filtered",
            ),
            PointCloud2,
            queue_size=1,
        )

        # Publishers for distance images in various directions
        self.dist_array_center_publisher = rospy.Publisher(
            rospy.get_param("~image_distance_topic", "/paf/hero/Center/dist_array"),
            ImageMsg,
            queue_size=10,
        )
        self.dist_array_back_publisher = rospy.Publisher(
            rospy.get_param("~image_distance_topic", "/paf/hero/Back/dist_array"),
            ImageMsg,
            queue_size=10,
        )
        self.dist_array_left_publisher = rospy.Publisher(
            rospy.get_param("~image_distance_topic", "/paf/hero/Left/dist_array"),
            ImageMsg,
            queue_size=10,
        )
        self.dist_array_right_publisher = rospy.Publisher(
            rospy.get_param("~image_distance_topic", "/paf/hero/Right/dist_array"),
            ImageMsg,
            queue_size=10,
        )
        self.dist_array_lidar_publisher = rospy.Publisher(
            rospy.get_param(
                "~image_distance_topic_cluster", "/paf/hero/dist_clustered"
            ),
            PointCloud2,
            queue_size=10,
        )
        self.marker_visualization_lidar_publisher = rospy.Publisher(
            rospy.get_param("~marker_topic", "/paf/hero/Lidar/Marker"),
            MarkerArray,
            queue_size=10,
        )

        # Subscriber for LiDAR data (point clouds)
        rospy.Subscriber(
            rospy.get_param("~source_topic", "/carla/hero/LIDAR"),
            PointCloud2,
            self.callback,
        )

        rospy.loginfo("Lidar Processor Node started.")
        rospy.spin()

    def start_clustering(self, data):
        """
        Filters LiDAR point clouds, performs clustering,
        generates bounding boxes, and publishes the results.

        :param data: LiDAR point clouds in ROS PointCloud2 format.
        """

        # Convert PointCloud2 data to a NumPy structured array
        coordinates = ros_numpy.point_cloud2.pointcloud2_to_array(data)

        # Filter the point clouds to exclude irrelevant data
        z_min = rospy.get_param("~clustering_lidar_z_min", -1.4)
        filtered_coordinates = coordinates[
            ~(
                (coordinates["x"] >= -2)
                & (coordinates["x"] <= 2)  # Exclude ego vehicle in x-axis
                & (coordinates["y"] >= -1)
                & (coordinates["y"] <= 1)  # Exclude ego vehicle in y-axis
            )
            & (
                coordinates["z"] > z_min
            )  # Exclude points below a certain height (street)
        ]

        # Perform clustering on the filtered coordinates
        eps = rospy.get_param("~dbscan_eps", 0.4)
        min_samples = rospy.get_param("~dbscan_min_samples", 10)
        clustered_points, cluster_labels = cluster_lidar_data_from_pointcloud(
            filtered_coordinates, eps, min_samples
        )

        # Extract x, y, z coordinates into a separate array
        filtered_xyz = np.column_stack(
            (
                filtered_coordinates["x"],
                filtered_coordinates["y"],
                filtered_coordinates["z"],
            )
        )

        # Combine coordinates with their cluster labels
        cluster_labels = cluster_labels.reshape(-1, 1)
        points_with_labels = np.hstack((filtered_xyz, cluster_labels))

        bounding_boxes = generate_bounding_boxes(points_with_labels)

        # Create a MarkerArray for visualization
        marker_array = MarkerArray()
        for label, bbox in bounding_boxes:
            if label != -1:  # Ignore noise points (label = -1)
                marker = create_bounding_box_marker(label, bbox)
                marker_array.markers.append(marker)

        # Publish the MarkerArray for visualization
        self.marker_visualization_lidar_publisher.publish(marker_array)

        # Store valid cluster data for combining
        if clustered_points:
            self.cluster_buffer.append(clustered_points)
        else:
            rospy.logwarn("No cluster data generated.")

        # Combine clusters from the buffer
        combined_clusters = combine_clusters(self.cluster_buffer)
        self.cluster_buffer = []

        self.publish_clusters(combined_clusters, data.header)

    def publish_clusters(self, combined_clusters, data_header):
        """
        Publishes combined clusters as a ROS PointCloud2 message.

        :param combined_clusters: Combined point clouds of the clusters as a structured
         NumPy array.
        :param data_header: Header information for the ROS message.
        """
        # Convert to a PointCloud2 message
        pointcloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(combined_clusters)
        pointcloud_msg.header = data_header
        pointcloud_msg.header.stamp = rospy.Time.now()
        # Publish the clusters
        self.dist_array_lidar_publisher.publish(pointcloud_msg)

    def start_image_calculation(self, data):
        """
        Computes distance images based on LiDAR data and publishes them.

        :param data: LiDAR point cloud in ROS PointCloud2 format.
        """
        coordinates = ros_numpy.point_cloud2.pointcloud2_to_array(data)

        # Directions to process
        directions = ["Center", "Back", "Left", "Right"]

        # Process images for all directions
        processed_images = {
            direction: self.calculate_image(coordinates, focus=direction)
            for direction in directions
        }

        # Publish the processed images
        self.publish_images(processed_images, data.header)

    def calculate_image(self, coordinates, focus):
        """
        Calculates a distance image for a specific focus (view direction) from
        LiDAR coordinates.

        :param coordinates: Filtered LiDAR coordinates as a NumPy array.
        :param focus: The focus direction ("Center", "Back", "Left", "Right").
        :return: Distance image as a 2D array.
        """
        # Define bounding box parameters based on focus direction
        bounding_box_params = {
            "Center": {"max_x": np.inf, "min_x": 0.0, "min_z": -np.inf},
            "Back": {"max_x": 0.0, "min_x": -np.inf, "min_z": -1.6},
            "Left": {"max_y": np.inf, "min_y": 0.0, "min_z": -1.6},
            "Right": {"max_y": -0.0, "min_y": -np.inf, "min_z": -1.6},
        }

        # Select parameters for the given focus
        params = bounding_box_params.get(focus)
        if not params:
            rospy.logwarn(f"Unknown focus: {focus}. Skipping image calculation.")
            return None

        # Apply bounding box filter
        reconstruct_bit_mask = bounding_box(coordinates, **params)
        reconstruct_coordinates = coordinates[reconstruct_bit_mask]

        # Remove the "intensity" field and convert to a NumPy array
        reconstruct_coordinates_xyz = np.array(
            remove_field_name(reconstruct_coordinates, "intensity").tolist()
        )

        # Reconstruct the image based on the focus
        return self.reconstruct_img_from_lidar(reconstruct_coordinates_xyz, focus=focus)

    def publish_images(self, processed_images, data_header):
        """
        Publishes distance images for various directions as ROS image messages.

        :param processed_images: Dictionary with directions ("Center", "Back", etc.)
         as keys and image arrays as values.
        :param data_header: Header for the ROS image messages.
        """
        # Process only valid NumPy arrays
        for direction, image_array in processed_images.items():
            if not isinstance(image_array, np.ndarray):
                continue

            # Convert the image into a ROS image message
            dist_array_msg = self.bridge.cv2_to_imgmsg(
                image_array, encoding="passthrough"
            )
            dist_array_msg.header = data_header

            if direction == "Center":
                self.dist_array_center_publisher.publish(dist_array_msg)
            if direction == "Back":
                self.dist_array_back_publisher.publish(dist_array_msg)
            if direction == "Left":
                self.dist_array_left_publisher.publish(dist_array_msg)
            if direction == "Right":
                self.dist_array_right_publisher.publish(dist_array_msg)

    def reconstruct_img_from_lidar(self, coordinates_xyz, focus):
        """
        Reconstructs a 2D image from 3D LiDAR data for a given camera perspective.

        :param coordinates_xyz: 3D coordinates of the filtered LiDAR points.
        :param focus: Camera view (e.g., "Center", "Back").
        :return: Reconstructed image as a 2D array.
        """

        # Create the intrinsic camera matrix based on image parameters (FOV, resolution)
        im = np.identity(3)
        im[0, 2] = 1280 / 2.0  # x-offset (image center)
        im[1, 2] = 720 / 2.0  # y-offset (image center)
        im[0, 0] = im[1, 1] = 1280 / (
            2.0 * np.tan(100 * np.pi / 360.0)
        )  # Scale factor based on FOV

        # Create the extrinsic camera matrix (identity matrix for no transformation)
        ex = np.zeros(shape=(3, 4))
        ex[0][0] = ex[1][1] = ex[2][2] = 1
        m = np.matmul(im, ex)  # Combine intrinsic and extrinsic matrices

        # Initialize empty images for reconstruction
        img = np.zeros(shape=(720, 1280), dtype=np.float32)
        dist_array = np.zeros(shape=(720, 1280, 3), dtype=np.float32)

        # Prepare points based on focus
        if focus in ["Center", "Back"]:
            points = np.column_stack(
                (
                    coordinates_xyz[:, 1],
                    coordinates_xyz[:, 2],
                    coordinates_xyz[:, 0],
                    np.ones(coordinates_xyz.shape[0]),
                )
            )
        elif focus in ["Left", "Right"]:
            points = np.column_stack(
                (
                    coordinates_xyz[:, 0],
                    coordinates_xyz[:, 2],
                    coordinates_xyz[:, 1],
                    np.ones(coordinates_xyz.shape[0]),
                )
            )
        else:
            rospy.logwarn(f"Unknown focus: {focus}. Skipping image calculation.")
            return None

        # Project 3D points to 2D image coordinates
        pixels = np.dot(m, points.T).T
        x = (pixels[:, 0] / pixels[:, 2]).astype(int)
        y = (pixels[:, 1] / pixels[:, 2]).astype(int)

        # Filter valid coordinates
        valid_indices = (x >= 0) & (x < 1280) & (y >= 0) & (y < 720)
        x = x[valid_indices]
        y = y[valid_indices]
        valid_coordinates = coordinates_xyz[valid_indices]

        if focus == "Center":
            img[719 - y, 1279 - x] = valid_coordinates[:, 0]
            dist_array[719 - y, 1279 - x] = valid_coordinates
        elif focus == "Back":
            img[y, 1279 - x] = -valid_coordinates[:, 0]
            dist_array[y, 1279 - x] = np.column_stack(
                (
                    -valid_coordinates[:, 0],
                    valid_coordinates[:, 1],
                    valid_coordinates[:, 2],
                )
            )
        elif focus == "Left":
            img[719 - y, 1279 - x] = valid_coordinates[:, 1]
            dist_array[719 - y, 1279 - x] = valid_coordinates
        elif focus == "Right":
            img[y, 1279 - x] = -valid_coordinates[:, 1]
            dist_array[y, 1279 - x] = np.column_stack(
                (
                    valid_coordinates[:, 0],
                    -valid_coordinates[:, 1],
                    valid_coordinates[:, 2],
                )
            )

        return dist_array


def generate_bounding_boxes(points_with_labels):
    """
    Generates axis-aligned bounding boxes (AABB) for clustered points.

    This function calculates bounding boxes for each unique
    cluster label in the input array.

    Args:
        points_with_labels (numpy.ndarray):
            A 2D array of shape (N, 4), where each row contains:
            - Coordinates (x, y, z) of a point
            - Cluster label in the last column
            Structure: [x, y, z, label]

    Returns:
        list:
            A list of tuples. Each tuple contains:
            - A cluster label (int/float)
            - A bounding box (tuple): (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    bounding_boxes = []

    # Identify unique cluster labels
    unique_labels = np.unique(points_with_labels[:, -1])

    # Process each cluster label
    for label in unique_labels:
        if label == -1:  # Skip noise points (label = -1)
            continue

        # Extract points belonging to the current cluster
        cluster_points = points_with_labels[points_with_labels[:, -1] == label, :3]

        # Calculate the bounding box for the cluster
        bbox = calculate_aabb(cluster_points)
        bounding_boxes.append((label, bbox))

    return bounding_boxes


def create_bounding_box_marker(label, bbox, bbox_type="aabb", frame_id="hero/LIDAR"):
    """
    Creates an RViz Marker for visualizing a 3D bounding box using Marker.CUBE.

    Args:
        label (int): Unique identifier for the cluster or object.
                     Used as the Marker ID.
        bbox (tuple): Bounding box dimensions.
                      For AABB: (x_min, x_max, y_min, y_max, z_min, z_max).
                      For OBB: (center, dimensions, eigenvectors).
        bbox_type (str): The type of bounding box ("aabb" or "obb").

    Returns:
        Marker: A CUBE Marker object that can be published to RViz.
    """
    # Initialize the Marker object
    marker = Marker()
    marker.header.frame_id = frame_id  # Reference frame for the marker
    marker.ns = "marker_lidar"  # Namespace to group related markers
    marker.id = int(label)  # Use the label as the unique marker ID
    marker.lifetime = rospy.Duration(0.1)  # Marker visibility duration in seconds
    marker.type = Marker.CUBE  # Use a cube for the bounding box
    marker.action = Marker.ADD  # Action to add or modify the marker

    # Set marker color and opacity
    marker.color.r = 1.0  # Red
    marker.color.g = 0.5  # Green
    marker.color.b = 0.5  # Blue
    marker.color.a = 0.8  # Opacity

    if bbox_type == "aabb":
        x_min, x_max, y_min, y_max, z_min, z_max = bbox

        # Calculate center and dimensions for AABB
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        center_z = (z_min + z_max) / 2.0

        size_x = x_max - x_min
        size_y = y_max - y_min
        size_z = z_max - z_min

        marker.pose.position.x = center_x
        marker.pose.position.y = center_y
        marker.pose.position.z = center_z

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = size_x
        marker.scale.y = size_y
        marker.scale.z = size_z

    elif bbox_type == "obb":
        center, dimensions, eigenvectors = bbox
        width, length, height = dimensions

        # Assign OBB parameters
        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = center[2]

        # Convert eigenvectors to quaternion
        quaternion = quaternion_from_matrix(np.vstack([eigenvectors.T, [0, 0, 0]]).T)
        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.pose.orientation.w = quaternion[3]

        marker.scale.x = width
        marker.scale.y = length
        marker.scale.z = height

    else:
        raise ValueError(f"Unsupported bbox_type: {bbox_type}")

    return marker


def calculate_aabb(cluster_points):
    """
    Calculates the axis-aligned bounding box (AABB) for a set of 3D points.

    This function computes the minimum and maximum values along each axis (x, y, z)
    for a given set of 3D points, which defines the bounding box that contains
    all points in the cluster.

    Args:
        cluster_points (numpy.ndarray):
        A 2D array where each row represents a 3D point (x, y, z).
        The array should have shape (N, 3) where N is the number of points.

    Returns:
        tuple: A tuple of the form (x_min, x_max, y_min, y_max, z_min, z_max),
        which represents the axis-aligned bounding box (AABB) for the given
        set of points. The values are the minimum and maximum coordinates
        along the x, y, and z axes.
    """

    # for 3d boxes
    x_min = np.min(cluster_points[:, 0])
    x_max = np.max(cluster_points[:, 0])
    y_min = np.min(cluster_points[:, 1])
    y_max = np.max(cluster_points[:, 1])
    z_min = np.min(cluster_points[:, 2])
    z_max = np.max(cluster_points[:, 2])
    return x_min, x_max, y_min, y_max, z_min, z_max


def array_to_pointcloud2(points, header="hero/Lidar"):
    """
    Converts an array of points into a ROS PointCloud2 message.

    :param points: Array of points with [x, y, z] coordinates.
    :param header: Header information for the ROS PointCloud2 message.
    :return: ROS PointCloud2 message.
    """
    # Ensure the input is a NumPy array
    points_array = np.array(points)

    # Convert the points into a structured array with fields "x", "y", "z"
    points_structured = np.array(
        [(p[0], p[1], p[2]) for p in points_array],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )

    # Create a PointCloud2 message from the structured array
    pointcloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(points_structured)

    # Set the timestamp and header for the message
    pointcloud_msg.header.stamp = rospy.Time.now()
    pointcloud_msg.header = header

    return pointcloud_msg


def combine_clusters(cluster_buffer):
    """
    Combines clusters from multiple point clouds into a structured NumPy array.

    :param cluster_buffer: List of dictionaries containing cluster IDs and point clouds.
    :return: Combined structured NumPy array with fields "x", "y", "z", "cluster_id".
    """
    points_list = []
    cluster_ids_list = []

    for clustered_points in cluster_buffer:
        for cluster_id, points in clustered_points.items():
            if points.size > 0:  # Ignore empty clusters
                points_list.append(points)
                # Create an array with the cluster ID for all points in the cluster
                cluster_ids_list.append(
                    np.full(points.shape[0], cluster_id, dtype=np.float32)
                )

    if not points_list:  # If no points are present
        return np.array(
            [], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("cluster_id", "f4")]
        )

    # Combine all points and cluster IDs into two separate arrays
    all_points = np.vstack(points_list)
    all_cluster_ids = np.concatenate(cluster_ids_list)

    # Create a structured array for the combined data
    combined_points = np.zeros(
        all_points.shape[0],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("cluster_id", "f4")],
    )
    combined_points["x"] = all_points[:, 0]
    combined_points["y"] = all_points[:, 1]
    combined_points["z"] = all_points[:, 2]
    combined_points["cluster_id"] = all_cluster_ids

    return combined_points


def cluster_lidar_data_from_pointcloud(coordinates, eps, min_samples):
    """
    Performs clustering on LiDAR data using DBSCAN and returns the clusters.

    :param coordinates: LiDAR point cloud as a NumPy array with "x", "y", "z".
    :param eps: Maximum distance between points to group them into a cluster.
    :param min_samples: Minimum number of points required to form a cluster.
    :return: Dictionary with cluster IDs and their corresponding point clouds.
    """
    if coordinates.shape[0] == 0:
        rospy.logerr("The input array 'coordinates' is empty.")
        return {}

    # Extract x, y, and z from the coordinates for DBSCAN clustering
    xyz = np.column_stack((coordinates["x"], coordinates["y"], coordinates["z"]))

    if xyz.shape[0] == 0:
        rospy.logwarn("No data points available for DBSCAN. Skipping clustering.")
        return {}

    # Apply DBSCAN to compute cluster labels for the point cloud
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
    labels = clustering.labels_

    # Remove noise (cluster ID: -1) and identify valid cluster IDs
    unique_labels = np.unique(labels)
    valid_labels = unique_labels[unique_labels != -1]

    # Create a dictionary with cluster IDs and their corresponding points
    clusters = Parallel(n_jobs=-1)(
        delayed(lambda cluster_label: (cluster_label, xyz[labels == cluster_label]))(
            label
        )
        for label in valid_labels
    )

    clusters = dict(clusters)

    return clusters, labels


if __name__ == "__main__":
    """
    Initialisiert die LidarDistance-Klasse und startet die Listener-Methode.
    """
    lidar_distance = LidarDistance()
    lidar_distance.listener()

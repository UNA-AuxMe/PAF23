from mapping.msg import ClusteredPointsArray
import rospy


def array_to_clustered_points(
    points,
    point_indices,
    object_speed_array=None,
    object_class_array=None,
    header_id="hero",
):
    """
    Convert the given points and point indices to a ClusteredPointsArray message.

    Args:
        points: numpy array with shape (N, 3)
        point_indices: numpy array with the shape (N,)
        object_speed_array: numpy array with the shape (N,)
        object_class_array: numpy array with the shape (N,)
        header_id: string

    Returns:
        ClusteredPointsArray message
    """
    # Create the ClusteredPointsArray message
    clustered_points = ClusteredPointsArray()
    clustered_points.header.frame_id = header_id
    clustered_points.header.stamp = rospy.Time.now()

    # Flatten the points array into a single list of [x1, y1, z1, x2, y2, z2, ...]
    clustered_points.clusterPointsArray = points.flatten().tolist()

    # Populate the indexArray
    clustered_points.indexArray = point_indices.astype(int).tolist()

    # Populate the motionArray if object_speed_array is provided
    if object_speed_array is not None:
        clustered_points.motionArray = object_speed_array
        # rospy.logerr("Motion2D is not implemented")

    # Populate the object_class if object_class_array is provided
    if object_class_array is not None:
        clustered_points.object_class = object_class_array

    return clustered_points

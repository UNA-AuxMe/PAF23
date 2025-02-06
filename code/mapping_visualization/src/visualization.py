#!/usr/bin/env python


import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from visualization_msgs.msg import Marker, MarkerArray
from mapping.msg import Map as MapMsg

from rospy import Publisher
from rospy import Duration

from mapping_common.map import Map

MARKER_NAMESPACE: str = "map"


class Visualization(CompatibleNode):
    """The visualization for the intermediate layer.

    This Node will publish the marker array composed of the different entities.
    """

    def __init__(self, args):
        super().__init__("mapping_visualization")

        self.marker_publisher: Publisher = self.new_publisher(
            MarkerArray, "/paf/hero/mapping/marker_array", qos_profile=1
        )

        self.filter_is_collider = False
        self.filter_is_tracked = False
        self.filter_is_stopmark = False
        self.filter_is_lanemark = False
        self.filter_is_ignored = False
        self.show_meta_markers = True

        self.new_service(SetBool, "vis/set_is_collider", self.set_is_collider_filter)
        self.new_service(SetBool, "vis/set_is_tracked", self.set_is_tracked_filter)
        self.new_service(SetBool, "vis/set_is_stopmark", self.set_is_stopmark_filter)
        self.new_service(SetBool, "vis/set_is_lanemark", self.set_is_lanemark_filter)
        self.new_service(SetBool, "vis/set_is_ignored", self.set_is_ignored_filter)

        self.new_subscription(
            topic=self.get_param("~map_topic", "/paf/hero/mapping/init_data"),
            msg_type=MapMsg,
            callback=self.map_callback,
            qos_profile=1,
        )

    def map_callback(self, data: MapMsg):
        map = Map.from_ros_msg(data)
        marker_array = MarkerArray()
        marker_array.markers.append(self.create_deleteall_marker())

        marker_timestamp = roscomp.ros_timestamp(self.get_time(), from_sec=True)
        markers = []
        for entity in map.entities:
            # TODO: filtering based on flags
            markers.append(entity.to_marker())
            if not self.show_meta_markers:
                continue
            markers.extend(entity.get_meta_markers())

        for id, marker in enumerate(markers):
            marker.header.stamp = marker_timestamp
            marker.header.frame_id = "hero"
            marker.ns = MARKER_NAMESPACE
            marker.id = id + 1000
            marker.lifetime = Duration.from_sec(2 / 20.0)
            marker_array.markers.append(marker)

        self.marker_publisher.publish(marker_array)

    def create_deleteall_marker(self) -> Marker:
        """Creates a marker that deletes all markers in RViz for this topic.

        Prepend this to the MarkerArray to delete all markers
        before the new ones are displayed.
        """
        return Marker(ns=MARKER_NAMESPACE, action=Marker.DELETEALL)

    def set_is_collider_filter(self, request: SetBoolRequest):
        self.filter_is_collider = request.data
        self.loginfo(f"Called collider {request.data}")

        response = SetBoolResponse()
        response.success = True
        return response

    def set_is_tracked_filter(self, request: SetBoolRequest):
        self.filter_is_tracked = request.data
        self.loginfo(f"Called tracked {request.data}")

        response = SetBoolResponse()
        response.success = True
        return response

    def set_is_stopmark_filter(self, request: SetBoolRequest):
        self.filter_is_stopmark = request.data
        self.loginfo(f"Called stopmark {request.data}")
        response = SetBoolResponse()
        response.success = True
        return response

    def set_is_lanemark_filter(self, request: SetBoolRequest):
        self.filter_is_lanemark = request.data

        self.loginfo(f"Called lanemark {request.data}")
        response = SetBoolResponse()
        response.success = True
        return response

    def set_is_ignored_filter(self, request: SetBoolRequest):
        self.filter_is_ignored = request.data

        self.loginfo(f"Called ignored {request.data}")
        response = SetBoolResponse()
        response.success = True
        return response


def main(args=None):
    """Main function to start the node.

    Args:
        args (_type_, optional): Runtime args, do not get processed. Defaults to None.
    """

    roscomp.init("mapping_visualization")

    try:
        node = Visualization(args)
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        roscomp.shutdown()


if __name__ == "__main__":
    main()

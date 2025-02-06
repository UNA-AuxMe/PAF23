from typing import List, Optional, Dict, Tuple
from enum import Enum
from dataclasses import dataclass, field

import shapely

from uuid import UUID, uuid4
from genpy.rostime import Time, Duration
from std_msgs.msg import Header
import uuid_msgs.msg as uuid_msgs
from visualization_msgs.msg import Marker, MarkerArray
import rospy

from mapping_common.shape import Shape2D, MarkerStyle
from mapping_common.transform import Vector2, Transform2D, Point2

from mapping import msg


@dataclass
class Motion2D:
    """Motion of an entity

    Speeds are global (not relative to hero car).
    """

    linear_motion: Vector2 = field(default_factory=lambda: Vector2.zero())
    """Linear motion

    Direction vector based on the x-axis/heading of the entity
    (which is furthermore based on the entity's transform)

    The length of the vector is the velocity in m/s
    """
    angular_velocity: float = 0.0
    """Angular velocity in radians/s

    - angle > 0: CCW
    - angle < 0: CW
    """

    @staticmethod
    def from_ros_msg(m: msg.Motion2D) -> "Motion2D":
        motion = Vector2.from_ros_msg(m.linear_motion)
        return Motion2D(linear_motion=motion, angular_velocity=m.angular_velocity)

    def to_ros_msg(self) -> msg.Motion2D:
        motion = self.linear_motion.to_ros_msg()
        return msg.Motion2D(
            linear_motion=motion, angular_velocity=self.angular_velocity
        )


@dataclass(init=False)
class Flags:
    """Dedicated flags an entity can have.

    Look into the FlagFilter class for an explanation fo the individual flags.

    Note that attributes should not be accessed directly,
    but only the matches_filter function should be used
    """

    _is_collider: bool = False
    _is_stopmark: bool = False
    _is_lanemark: bool = False
    _is_ignored: bool = False
    _is_hero: bool = False

    def __init__(
        self,
        is_collider: bool = False,
        is_stopmark: bool = False,
        is_lanemark: bool = False,
        is_ignored: bool = False,
        is_hero: bool = False,
    ):
        self._is_collider = is_collider
        self._is_stopmark = is_stopmark
        self._is_lanemark = is_lanemark
        self._is_ignored = is_ignored
        self._is_hero = is_hero

    def matches_filter(self, f: "FlagFilter") -> bool:
        """Returns if these Flags match the filter mask f

        Args:
            f (FlagFilter): Filter mask to filter with

        Returns:
            bool: If f matches self
        """
        if f.is_ignored is not None:
            if self._is_ignored is not f.is_ignored:
                return False
        if f.is_collider is not None:
            if self._is_collider is not f.is_collider:
                return False
        if f.is_stopmark is not None:
            if self._is_stopmark is not f.is_stopmark:
                return False
        if f.is_lanemark is not None:
            if self._is_lanemark is not f.is_lanemark:
                return False
        if f.is_hero is not None:
            if self._is_hero is not f.is_hero:
                return False
        return True

    @staticmethod
    def from_ros_msg(m: msg.Flags) -> "Flags":
        return Flags(
            is_collider=m.is_collider,
            is_stopmark=m.is_stopmark,
            is_lanemark=m.is_lanemark,
            is_ignored=m.is_ignored,
            is_hero=m.is_hero,
        )

    def to_ros_msg(self) -> msg.Flags:
        return msg.Flags(
            is_collider=self._is_collider,
            is_stopmark=self._is_stopmark,
            is_lanemark=self._is_lanemark,
            is_ignored=self._is_ignored,
            is_hero=self._is_hero,
        )


@dataclass
class FlagFilter:
    """Filter mask to filter entities by their flags

    Setting a flag to None means it is ignored when filtering
    """

    has_motion: Optional[bool] = None
    """motion is not None"""
    is_collider: Optional[bool] = None
    """hero can collide with this entity"""
    is_tracked: Optional[bool] = None
    """tracking_info is not None"""
    is_stopmark: Optional[bool] = None
    """entity is a stopmark/-line where hero has to stop and wait"""
    is_lanemark: Optional[bool] = None
    """entity is a lanemark"""
    is_ignored: Optional[bool] = None
    """entity should be ignored in all downstream algorithms"""
    is_hero: Optional[bool] = None
    """this entity is the SssssssssssssssssssssuperCar"""


@dataclass
class TrackingInfo:
    """Information that might be required to consistently track entities"""

    visibility_time: Duration = field(default_factory=Duration)
    """How long the entity has been visible for. Never gets reset"""
    invisibility_time: Duration = field(default_factory=Duration)
    """How long the entity has been uninterruptedly not visible.
    Reset when the entity is visible again"""
    visibility_frame_count: int = 0
    """In how many data frames the entity was visible. Never gets reset"""
    invisibility_frame_count: int = 0
    """In how many consecutive data frames the entity was not visible.
    Reset when the entity is visible again"""
    moving_time: Duration = field(default_factory=Duration)
    """How long an entity was moving continuously. Reset when standing"""
    standing_time: Duration = field(default_factory=Duration)
    """How long an entity stood still continuously. Reset when moving"""
    moving_time_sum: Duration = field(default_factory=Duration)
    """Sums of all the time the entity was moving. Never gets reset"""
    standing_time_sum: Duration = field(default_factory=Duration)
    """Sums of all the time the entity was standing still. Never gets reset"""
    min_linear_speed: float = 0.0
    """Minimum linear speed of this entity ever recorded"""
    max_linear_speed: float = 0.0
    """Maximum linear speed of this entity ever recorded"""

    @staticmethod
    def from_ros_msg(m: msg.TrackingInfo) -> "TrackingInfo":
        return TrackingInfo(
            visibility_time=m.visibility_time,
            invisibility_time=m.invisibility_time,
            visibility_frame_count=m.visibility_frame_count,
            invisibility_frame_count=m.invisibility_frame_count,
            moving_time=m.moving_time,
            standing_time=m.standing_time,
            moving_time_sum=m.moving_time_sum,
            standing_time_sum=m.standing_time_sum,
            min_linear_speed=m.min_linear_speed,
            max_linear_speed=m.max_linear_speed,
        )

    def to_ros_msg(self) -> msg.TrackingInfo:
        return msg.TrackingInfo(
            visibility_time=self.visibility_time,
            invisibility_time=self.invisibility_time,
            visibility_frame_count=self.visibility_frame_count,
            invisibility_frame_count=self.invisibility_frame_count,
            moving_time=self.moving_time,
            standing_time=self.standing_time,
            moving_time_sum=self.moving_time_sum,
            standing_time_sum=self.standing_time_sum,
            min_linear_speed=self.min_linear_speed,
            max_linear_speed=self.max_linear_speed,
        )


@dataclass
class Entity:
    """A thing of interest around the hero car that has a location and a shape"""

    confidence: float
    """The sensor's confidence that this entity is correct and actually exists."""
    priority: float
    """This entity's priority over others. Mostly relevant for the merging filter"""
    shape: Shape2D
    """Shape2D for collision calculations"""
    transform: Transform2D
    """Transform2D based on the map origin (hero car)"""
    timestamp: Time = field(default_factory=Time)
    """When adding the entity its timestamp is the timestamp
    of the associated sensor data
    (might slightly differ to the timestamp of the Map)

    In the filtering steps the entity will be predicted to the map creation timestamp.
    All entities of the output map have the same timestamp as the map.
    """
    flags: Flags = field(default_factory=Flags)
    """Flags (Categories) this entity is in

    Filter for them with the matches_filter(f) function and a FlagFilter
    """
    uuid: UUID = field(default_factory=uuid4)
    """Unique id of the entity in the map, set by the map

    If the entity is tracked, this id is persistent across map data frames
    """
    sensor_id: List[str] = field(default_factory=list)
    """Sort-of unique id(s) set by the sensor

    Used to aid the tracking.

    Prefixed with the sensor node name to make sure ids stay unique between sensors.

    Mainly set from the tracking id outputs of the yolov11 model.
    """
    motion: Optional[Motion2D] = None
    """The motion of this entity

    If unset the motion is unknown (Entity not properly tracked yet),
    but not necessarily zero.
    Some algorithms might assume the motion is zero when unset.
    """
    tracking_info: Optional[TrackingInfo] = None
    """Hold the tracking information for this entity

    If this entity is supposed to be tracked, tracking_info must not be None"""

    def matches_filter(self, f: FlagFilter) -> bool:
        """Returns if this entity matches the filter mask f

        Args:
            f (FlagFilter): Filter mask to filter with

        Returns:
            bool: If f matches self
        """
        if not self.flags.matches_filter(f):
            return False
        if f.has_motion is not None:
            if (self.motion is not None) is not f.has_motion:
                return False
        if f.is_tracked is not None:
            if (self.tracking_info is not None) is not f.is_tracked:
                return False
        return True

    @staticmethod
    def from_ros_msg(m: msg.Entity) -> "Entity":
        """Creates an entity from m

        Note that the returned entity might be a subclass of Entity
        """
        entity_type = None
        msg_type_lower = m.type_name.lower()
        if msg_type_lower in _entity_supported_classes_dict:
            entity_type = _entity_supported_classes_dict[msg_type_lower]
        if entity_type is None:
            rospy.logerr(
                f"Received entity type '{m.type_name}' is not supported."
                f"Base class 'Entity' will be used instead."
                f"The type must be one of {_entity_supported_classes_dict.keys()}"
            )
            entity_type = Entity

        kwargs = entity_type._extract_kwargs(m)
        return entity_type(**kwargs)

    @staticmethod
    def _extract_kwargs(m: msg.Entity) -> Dict:
        """Extracts all attributes for the __init__ of this type from m

        Args:
            m (msg.Entity): ROS message to extract data from

        Returns:
            Dict: arguments for the __init__ function
        """
        motion = (
            None if m.flags.has_motion is False else Motion2D.from_ros_msg(m.motion)
        )
        tracking_info = (
            None
            if m.flags.is_tracked is False
            else TrackingInfo.from_ros_msg(m.tracking_info)
        )
        return {
            "confidence": m.confidence,
            "priority": m.priority,
            "shape": Shape2D.from_ros_msg(m.shape),
            "transform": Transform2D.from_ros_msg(m.transform),
            "timestamp": m.header.stamp,
            "flags": Flags.from_ros_msg(m.flags),
            "uuid": UUID(bytes=m.uuid.uuid),
            "sensor_id": m.sensor_id,
            "motion": motion,
            "tracking_info": tracking_info,
        }

    def to_ros_msg(self) -> msg.Entity:
        type_name = type(self).__name__
        flags = self.flags.to_ros_msg()
        flags.has_motion = self.motion is not None
        flags.is_tracked = self.tracking_info is not None
        motion = self.motion.to_ros_msg() if self.motion is not None else msg.Motion2D()
        tracking_info = (
            self.tracking_info.to_ros_msg()
            if self.tracking_info is not None
            else msg.TrackingInfo()
        )
        return msg.Entity(
            confidence=self.confidence,
            priority=self.priority,
            shape=self.shape.to_ros_msg(),
            transform=self.transform.to_ros_msg(),
            header=Header(stamp=self.timestamp),
            flags=flags,
            uuid=uuid_msgs.UniqueID(uuid=self.uuid.bytes),
            sensor_id=self.sensor_id,
            motion=motion,
            tracking_info=tracking_info,
            type_name=type_name,
        )

    def to_marker(self) -> Marker:
        """Creates an ROS marker based on the entity

        Returns:
            Marker: ROS marker message
        """
        m = self.shape.to_marker(self.transform)

        m.color.r = 1
        m.color.g = 1
        m.color.b = 1
        m.color.a = 0.5
        m.pose.position.z = m.scale.z / 2.0
        return m

    def get_meta_markers(self) -> List[Marker]:
        """Creates additional meta markers for the entity

        Returns:
            List[Marker]: List of ROS marker messages
        """
        meta_markers = []
        if self.motion is not None:
            meta_markers.append(self.to_motion_marker())
            speed_in_ms = self.motion.linear_motion.length()
            speed_in_kmh = speed_in_ms * 3.6
            motion_text = f"{speed_in_kmh:.2f} km/h"
            meta_markers.append(self.get_text_marker(motion_text))
        return meta_markers

    def to_motion_marker(self) -> Marker:
        assert self.motion is not None
        m = Marker()
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.lifetime = Duration.from_sec(2 / 20.0)
        m.pose.position.x = self.transform.translation().x()
        m.pose.position.y = self.transform.translation().y()
        m.pose.position.z = 0.0
        m.scale.x = 0.1
        m.scale.y = 0.3
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
        m.points.append(Point2.zero().to_ros_msg())
        m.points.append(
            (self.transform * self.motion.linear_motion)
            .point()  # type: ignore
            .to_ros_msg()
        )
        return m

    def get_text_marker(self, text: str, offset: Optional[Vector2] = None) -> Marker:
        if offset is None:
            offset = Vector2.zero()
        text_marker = Marker()
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.lifetime = Duration.from_sec(2 / 20.0)
        text_marker.pose.position.x = self.transform.translation().x() + offset.x()
        text_marker.pose.position.y = self.transform.translation().y() + offset.y()
        text_marker.pose.position.z = 1.5
        text_marker.scale.z = 0.3
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.text = text
        return text_marker

    def to_shapely(self) -> "ShapelyEntity":
        return ShapelyEntity(self, self.shape.to_shapely(self.transform))

    def is_mergeable_with(self, other: "Entity") -> bool:
        """Returns if self should be merged/combined with other at all

        Mainly used in the filtering steps of the intermediate layer map

        Args:
            other (Entity): Other entity to merge with

        Returns:
            bool: If self and other should be merged at all
        """
        if self.flags._is_collider is not other.flags._is_collider:
            return False
        if self.flags._is_lanemark is not other.flags._is_lanemark:
            return False
        if self.flags._is_stopmark is not other.flags._is_stopmark:
            return False
        if self.flags._is_hero is not other.flags._is_hero:
            # This limitation might be removed later if there are no rectangular
            # clusters used anymore that might falsely overlap with the hero.
            return False
        if (
            isinstance(self, Car)
            and isinstance(other, Pedestrian)
            or isinstance(self, Pedestrian)
            and isinstance(other, Car)
        ):
            # Cars and pedestrians must get merged as a bicycle is detected
            # as a car and a pedestrian at the same time.
            return True
        if not (isinstance(self, type(other)) or isinstance(other, type(self))):
            return False
        return True

    def get_global_x_velocity(self) -> Optional[float]:
        """
        Returns the global x velocity of the entity in in m/s.

        Returns:
        - Optional[float]: Velocity of the entity in front in m/s.
        """

        if self.motion is None:
            return None

        motion = self.motion.linear_motion
        global_motion: Vector2 = self.transform * motion

        velocity = global_motion.x()
        return velocity

    def get_delta_forward_velocity_of(self, other: "Entity") -> Optional[float]:
        """Calculates the delta velocity compared to other in the heading of self

        - result > 0: other moves away from self
        - result < 0: other moves nearer to self

        Args:
            other (Entity)

        Returns:
            Optional[float]: Delta velocity if both entities have one.
        """
        if self.motion is None or other.motion is None:
            return None

        into_self_local = self.transform.inverse() * other.transform
        other_motion_in_self_coords: Vector2 = (
            into_self_local * other.motion.linear_motion
        )
        relative_motion = other_motion_in_self_coords - self.motion.linear_motion
        return relative_motion.x()

    def get_width(self) -> float:
        """Returns the local width (y-bounds) of the entity

        Returns:
            float: width
        """
        local_poly = self.shape.to_shapely()
        min_x, min_y, max_x, max_y = local_poly.bounds
        return max_y - min_y


@dataclass(init=False)
class Car(Entity):
    brake_light: "Car.BrakeLightState"
    indicator: "Car.IndicatorState"

    class BrakeLightState(Enum):
        OFF = 0
        ON = 1

    class IndicatorState(Enum):
        OFF = 0
        LEFT = 1
        RIGHT = 2

    def __init__(
        self,
        brake_light: "Car.BrakeLightState" = BrakeLightState.OFF,
        indicator: "Car.IndicatorState" = IndicatorState.OFF,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.brake_light = brake_light
        self.indicator = indicator

    @staticmethod
    def _extract_kwargs(m: msg.Entity) -> Dict:
        kwargs = super(Car, Car)._extract_kwargs(m)
        kwargs["brake_light"] = Car.BrakeLightState(m.type_car.brake_light)
        kwargs["indicator"] = Car.IndicatorState(m.type_car.indicator)
        return kwargs

    def to_ros_msg(self, base_msg: Optional[msg.Entity] = None) -> msg.Entity:
        m = super().to_ros_msg()
        m.type_car = msg.TypeCar(
            brake_light=self.brake_light.value, indicator=self.indicator.value
        )
        return m

    def to_marker(self) -> Marker:
        m = super().to_marker()
        # [0, 0, 255],  # 10: Vehicles
        m.color.r = 0
        m.color.g = 0
        m.color.b = 255 / 255
        return m


@dataclass(init=False)
class Lanemarking(Entity):
    style: "Lanemarking.Style"
    position_index: int
    predicted: bool

    class Style(Enum):
        SOLID = 0
        DASHED = 1

    def __init__(
        self, style: "Lanemarking.Style", position_index: int, predicted: bool, **kwargs
    ):
        super().__init__(**kwargs)
        self.style = style
        self.position_index = position_index
        self.predicted = predicted

    @staticmethod
    def _extract_kwargs(m: msg.Entity) -> Dict:
        kwargs = super(Lanemarking, Lanemarking)._extract_kwargs(m)
        kwargs["style"] = Lanemarking.Style(m.type_lanemarking.style)
        kwargs["position_index"] = m.type_lanemarking.position_index
        kwargs["predicted"] = m.type_lanemarking.predicted
        return kwargs

    def to_ros_msg(self, base_msg: Optional[msg.Entity] = None) -> msg.Entity:
        m = super().to_ros_msg()
        m.type_lanemarking = msg.TypeLanemarking(
            style=self.style.value,
            position_index=self.position_index,
            predicted=self.predicted,
        )
        return m

    def to_marker(self) -> Marker:
        """Creates an ROS marker based on the entity

        Returns:
            Marker: ROS marker message
        """
        m = self.shape.to_marker(self.transform)

        if self.predicted:
            m.color.a = 0.5
            m.color.r = int(255 * self.confidence)
            m.color.g = 0
            m.color.b = 0
        else:
            m.color.a = 0.5
            m.color.r = 0
            m.color.g = 0
            m.color.b = int(255 * self.confidence)

        m.scale.z = 0.1
        m.pose.position.z = m.scale.z / 2.0

        return m

    def get_meta_markers(self) -> List[Marker]:
        common_meta_markers = super().get_meta_markers()
        common_meta_markers.append(self.get_text_marker(f"{self.position_index}"))
        return common_meta_markers


@dataclass(init=False)
class TrafficLight(Entity):
    """Traffic light or stop sign

    Note: Class may be split up later

    TrafficLight and StopSign add only their stop line to the map.
    They set the *is_stopmark* flag only if the car has to stop there.
    """

    state: "TrafficLight.State"

    class State(Enum):
        RED = 0
        GREEN = 1
        YELLOW = 2

    def __init__(self, state: "TrafficLight.State", **kwargs):
        super().__init__(**kwargs)
        self.state = state

    @staticmethod
    def _extract_kwargs(m: msg.Entity) -> Dict:
        kwargs = super(TrafficLight, TrafficLight)._extract_kwargs(m)
        kwargs["state"] = TrafficLight.State(m.type_traffic_light.state)
        return kwargs

    def to_ros_msg(self, base_msg: Optional[msg.Entity] = None) -> msg.Entity:
        m = super().to_ros_msg()
        m.type_traffic_light = msg.TypeTrafficLight(state=self.state.value)
        return m


@dataclass(init=False)
class Pedestrian(Entity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_marker(self) -> Marker:
        m = super().to_marker()
        # [220, 20, 60],  # 4: Pedestrians
        m.color.r = 220 / 255
        m.color.g = 20 / 255
        m.color.b = 60 / 255
        return m


_entity_supported_classes = [Entity, Car, Lanemarking, TrafficLight, Pedestrian]
_entity_supported_classes_dict = {}
for t in _entity_supported_classes:
    t_name = t.__name__.lower()
    _entity_supported_classes_dict[t_name] = t


@dataclass
class ShapelyEntity:
    """A Container containing both an entity and a shapely.Polygon
    based on the entity's shape and transform

    **IMPORTANT** If the entity is modified, the Polygon will
    not automatically update itself and will contain outdated information.
    """

    entity: Entity
    poly: shapely.Polygon

    def get_distance_to(self, other: "ShapelyEntity") -> float:
        """Returns the distance to other in m.

        Args:
            other (ShapelyEntity)

        Returns:
            float: distance
        """

        return shapely.distance(self.poly, other.poly)


def shape_debug_marker_array(
    namespace: str,
    timestamp: Optional[rospy.Time] = None,
    lifetime: Optional[rospy.Duration] = None,
    entities: Optional[List[Tuple[Entity, Tuple[float, float, float, float]]]] = None,
    shapes: Optional[List[Tuple[Shape2D, Tuple[float, float, float, float]]]] = None,
    markers: Optional[List[Tuple[Marker, Tuple[float, float, float, float]]]] = None,
) -> MarkerArray:
    """Build a MarkerArray for debugging based on several mapping_common types

    All inputs are a list of tuples:
        - Each Tuple contains an object and a color
        - The color is a Tuple of (r, g, b, a)

    Args:
        namespace (str): Namespace of the markers
        timestamp (Optional[rospy.Time], optional): Timestamp of the markers.
            Defaults to None.
        lifetime (Optional[rospy.Duration], optional): Lifetime of the markers.
            Defaults to rospy.Duration.from_sec(0.5).
        entities (Optional[List[Tuple
            [Entity, Tuple[float, float, float, float]]]], optional):
            Entities to visualize. Defaults to None.
        shapes (Optional[List[Tuple
            [Shape2D, Tuple[float, float, float, float]]]], optional):
            Shapes to visualize. Defaults to None.
        markers (Optional[List[Tuple[
            Marker, Tuple[float, float, float, float]]]], optional):
            Markers to visualize . Defaults to None.

    Returns:
        MarkerArray: _description_
    """
    if lifetime is None:
        lifetime = rospy.Duration.from_sec(0.5)
    if entities is None:
        entities = []
    if shapes is None:
        shapes = []
    if markers is None:
        markers = []
    if timestamp is None:
        timestamp = rospy.get_rostime()

    for entity, color in entities:
        marker = entity.to_marker()
        markers.append((marker, color))
    for shape, color in shapes:
        marker = shape.to_marker(marker_style=MarkerStyle.LINESTRING)
        markers.append((marker, color))
    marker_array = MarkerArray(markers=[Marker(ns=namespace, action=Marker.DELETEALL)])
    for id, marker_color in enumerate(markers):
        marker, color = marker_color
        marker.header.frame_id = "hero"
        marker.header.stamp = timestamp
        marker.ns = namespace
        marker.id = id
        marker.lifetime = lifetime
        marker.scale.z = 0.3
        r, g, b, a = color
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a
        marker_array.markers.append(marker)

    return marker_array

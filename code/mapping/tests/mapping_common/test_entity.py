from mapping_common import entity, shape, transform


def get_shape():
    return shape.Rectangle(1.0, 2.0)


def get_transform():
    return transform.Transform2D.new_rotation(1.0)


def get_car():
    flags = entity.Flags(is_collider=True)
    return entity.Car(
        indicator=entity.Car.IndicatorState.LEFT,
        confidence=1.0,
        priority=1.0,
        shape=get_shape(),
        transform=get_transform(),
        motion=get_motion(),
        flags=flags,
    )


def get_motion():
    return entity.Motion2D(linear_motion=transform.Vector2.new(5.0, 0.0))


def test_entity_conversion():
    e = entity.Entity(
        confidence=1.0,
        priority=1.0,
        shape=get_shape(),
        transform=get_transform(),
    )
    msg = e.to_ros_msg()
    e_conv = entity.Entity.from_ros_msg(msg)
    assert e == e_conv


def test_car_conversion():
    c = get_car()
    msg = c.to_ros_msg()
    c_conv = entity.Entity.from_ros_msg(msg)
    assert c == c_conv


def test_lanemark_conversion():
    e = entity.Lanemarking(
        confidence=1.0,
        priority=1.0,
        shape=get_shape(),
        transform=get_transform(),
        style=entity.Lanemarking.Style.SOLID,
        position_index=1,
        predicted=False,
    )
    msg = e.to_ros_msg()
    e_conv = entity.Entity.from_ros_msg(msg)
    assert e == e_conv


def test_traffic_light_conversion():
    e = entity.TrafficLight(
        confidence=1.0,
        priority=1.0,
        shape=get_shape(),
        transform=get_transform(),
        state=entity.TrafficLight.State.RED,
    )
    msg = e.to_ros_msg()
    e_conv = entity.Entity.from_ros_msg(msg)
    assert e == e_conv


def test_pedestrian_conversion():
    e = entity.Pedestrian(
        confidence=1.0,
        priority=1.0,
        shape=get_shape(),
        transform=get_transform(),
    )
    msg = e.to_ros_msg()
    e_conv = entity.Entity.from_ros_msg(msg)
    assert e == e_conv


def test_matches_flags():
    e = get_car()
    flags = entity.Flags(is_collider=True)
    e.flags = flags
    filter = entity.FlagFilter(has_motion=True, is_collider=True, is_lanemark=False)

    assert e.matches_filter(filter)


def test_not_matches_flags():
    e = get_car()
    flags = entity.Flags(is_collider=True)
    e.flags = flags
    filter = entity.FlagFilter(has_motion=False, is_collider=True)

    assert not e.matches_filter(filter)

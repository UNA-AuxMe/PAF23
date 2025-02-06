from mapping_common.transform import Transform2D, Vector2, Point2
import math

from test_shape import get_polygon


def test_point_conversion():
    p = Point2.new(1.0, 25.0)
    msg = p.to_ros_msg()
    p_conv = Point2.from_ros_msg(msg)

    assert p == p_conv


def test_vector_conversion():
    v = Vector2.new(1.0, 25.0)
    msg = v.to_ros_msg()
    v_conv = Vector2.from_ros_msg(msg)

    assert v == v_conv


def test_transform_conversion():
    p = Point2.new(1.0, 25.0)
    msg = p.to_ros_msg()
    p_conv = Point2.from_ros_msg(msg)

    assert p == p_conv


def test_translation():
    transl_v = Vector2.new(1.0, 2.0)
    transl = Transform2D.new_translation(transl_v)
    zero_p = Point2.zero()
    zero_v = Vector2.zero()

    assert transl * zero_p == Point2.from_vector(transl_v)
    assert transl * zero_v == Vector2.zero()


def test_translation2():
    transl_v = Vector2.new(1.0, 2.0)
    transl = Transform2D.new_translation(transl_v)

    assert transl.translation() == transl_v


def test_rotation():
    rot = Transform2D.new_rotation(0.0)
    assert math.isclose(rot.rotation(), 0.0)


def test_rotation2():
    rot = Transform2D.new_rotation(math.pi / 2)
    assert math.isclose(rot.rotation(), math.pi / 2)


def test_rotation3():
    rot = Transform2D.new_rotation(-math.pi / 2)
    assert math.isclose(rot.rotation(), -math.pi / 2)


def test_rotation4():
    rot = Transform2D.new_rotation(-math.pi / 2)
    v = Vector2.new(1.0, 0.0)
    v_rot: Vector2 = rot * v
    assert math.isclose(v_rot.x(), 0.0, abs_tol=1e-15)
    assert math.isclose(v_rot.y(), -1.0, abs_tol=1e-15)


def test_rotation5():
    rot = Transform2D.new_rotation(math.pi / 2)
    v = Vector2.new(1.0, 0.0)
    v_rot: Vector2 = rot * v
    assert math.isclose(v_rot.x(), 0.0, abs_tol=1e-15)
    assert math.isclose(v_rot.y(), 1.0, abs_tol=1e-15)


def test_rotation_translation_vector():
    transl = Vector2.new(1.0, 2.0)
    transf = Transform2D.new_rotation_translation(math.pi / 2, transl)
    v = Vector2.new(1.0, 0.0)
    v_rot: Vector2 = transf * v
    # Vector ignores translation
    assert math.isclose(v_rot.x(), 0.0, abs_tol=1e-15)
    assert math.isclose(v_rot.y(), 1.0, abs_tol=1e-15)


def test_rotation_translation_point():
    transl = Vector2.new(1.0, 2.0)
    transf = Transform2D.new_rotation_translation(math.pi / 2, transl)
    p = Point2.new(1.0, 0.0)
    p_trans: Point2 = transf * p
    assert math.isclose(p_trans.x(), 1.0, abs_tol=1e-15)
    assert math.isclose(p_trans.y(), 3.0, abs_tol=1e-15)


def test_shapely_transform():
    shape = get_polygon()
    shapely_poly = shape.to_shapely()
    center = shapely_poly.centroid

    transl_v = Vector2.new(1.0, 2.0)
    transl = Transform2D.new_translation(transl_v)
    transformed = transl * shapely_poly
    t_center = transformed.centroid

    assert center.x + 1.0 == t_center.x and center.y + 2.0 == t_center.y


def test_normalize():
    v = Vector2.new(2.0, 2.0)
    v_norm = v.normalized()

    assert math.isclose(v_norm.length(), 1.0)


def test_vector_scalar_mul():
    v = Vector2.new(2.0, 2.0)
    v_norm = v.normalized()
    v_long = v_norm * 5.0

    assert math.isclose(v_long.length(), 5.0)


def test_vector_scalar_mul_int():
    v = Vector2.new(2.0, 2.0)
    v_norm = v.normalized()
    v_long = v_norm * 5

    assert math.isclose(v_long.length(), 5.0)


def test_vector_scalar_rmul():
    v = Vector2.new(2.0, 2.0)
    v_norm = v.normalized()
    v_long = 5.0 * v_norm

    assert math.isclose(v_long.length(), 5.0)


def test_zero_vector_normalization():
    v = Vector2.zero()
    assert v.length() == 0.0

    assert v.normalized() == v


def test_vector_add():
    v = Vector2.new(2.0, 2.0)
    v2 = Vector2.new(1.0, 4.0)
    v_sum = v + v2
    v_result = Vector2.new(3.0, 6.0)

    assert v_sum == v_result


def test_inverse():
    transl = Vector2.new(1.0, 2.0)
    transf = Transform2D.new_rotation_translation(math.pi / 2, transl)
    inv = transf.inverse()

    p = Point2.new(1.0, 0.0)
    p_trans: Point2 = transf * p
    p_ret: Point2 = inv * p_trans

    assert math.isclose(p_ret.x(), p.x(), abs_tol=1e-15)
    assert math.isclose(p_ret.y(), p.y(), abs_tol=1e-15)

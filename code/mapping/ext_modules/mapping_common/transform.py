from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import math
import shapely

from geometry_msgs import msg as geometry_msgs
from mapping import msg


@dataclass(init=False, eq=False)
class _Coord2:
    """Homogenous 2 dimensional coordinate

    This base class should be abstract,
    but cython does not support the ABC superclass and decorators
    """

    # Matrix with shape (3)
    _matrix: npt.NDArray[np.float64]

    def __init__(self, matrix: npt.NDArray[np.float64]) -> None:
        assert matrix.shape == (
            3,
        ), f"{type(self).__name__} matrix must have shape (3,)"
        self._matrix = matrix

    def x(self) -> float:
        return self._matrix[0]

    def y(self) -> float:
        return self._matrix[1]

    def __eq__(self, value) -> bool:
        if type(self) is type(value):
            return (self._matrix == value._matrix).all()
        return False


@dataclass(init=False, eq=False)
class Point2(_Coord2):
    """2 dimensional point.

    Receives both rotation and translation when transformed with a Transform2D"""

    def vector(self) -> "Vector2":
        return Vector2(self._matrix)

    def vector_to(self, other: "Point2") -> "Vector2":
        start = self.vector()
        end = other.vector()
        return end - start

    @staticmethod
    def new(x: float, y: float) -> "Point2":
        m = np.array([x, y, 1.0], dtype=np.float64)
        return Point2(m)

    @staticmethod
    def zero() -> "Point2":
        return Point2.new(0.0, 0.0)

    @staticmethod
    def from_vector(v: "Vector2") -> "Point2":
        return Point2(v._matrix)

    @staticmethod
    def from_ros_msg(m: geometry_msgs.Point) -> "Point2":
        return Point2.new(m.x, m.y)

    def to_ros_msg(self) -> geometry_msgs.Point:
        return geometry_msgs.Point(x=self.x(), y=self.y(), z=0.0)

    def to_shapely(self) -> shapely.Point:
        return shapely.Point(self._matrix[:2])

    def __add__(self, other):
        if isinstance(other, Vector2):
            matrix = self._matrix + other._matrix
            matrix[2] = 1.0
            return Point2(matrix)
        raise TypeError(
            f"Unsupported operand types for +: '{type(self)}' and '{type(other)}'"
        )

    def __sub__(self, other):
        if isinstance(other, Vector2):
            matrix = self._matrix - other._matrix
            matrix[2] = 1.0
            return Point2(matrix)
        raise TypeError(
            f"Unsupported operand types for -: '{type(self)}' and '{type(other)}'"
        )


@dataclass(init=False, eq=False)
class Vector2(_Coord2):
    """2 dimensional direction vector.

    Receives only the rotation when transformed with a Transform2D"""

    def length(self) -> float:
        """Calculates the length of this vector

        Returns:
            float: length of this vector
        """
        # This implementation was chosen because (at the time of writing this) it was
        # about 2 times faster than `np.linalg.norm(self._matrix[:2])`
        return math.sqrt(self._matrix[0] ** 2 + self._matrix[1] ** 2)

    def normalized(self) -> "Vector2":
        """Returns this direction Vector with length 1.0

        If the vector is the zero vector, the result will be zero as well

        Returns:
            Vector2: Vector with length 1.0
        """
        zero = Vector2.zero()
        if self == zero:
            return zero
        return self / self.length()

    def angle_to(self, other: "Vector2") -> float:
        """Calculates the angle to *other*

        Args:
            other (Vector2): Vector to calculate the angle to

        Returns:
            float: signed angle in radians. Always in interval [-pi,pi].

            - angle > 0: CCW
            - angle < 0: CW
        """
        # Based on https://stackoverflow.com/questions/21483999/
        # using-atan2-to-find-angle-between-two-vectors/21486462#21486462
        # A·B = |A| |B| COS(θ) = x
        # A×B = |A| |B| SIN(θ) = y (in 2 dimensional space only)
        # Where θ is the (unsigned) angle between A and B
        # => The implementation is very similar Transform2D.rotation(self)
        # => |A| |B| can be ignored because it is just a scalar multiplication
        #  of the vector (does not change direction)
        cross = np.cross(self._matrix[:2], other._matrix[:2])
        dot = np.dot(self._matrix[:2], other._matrix[:2])
        return math.atan2(cross, dot)

    def point(self) -> Point2:
        return Point2(self._matrix)

    @staticmethod
    def new(x: float, y: float) -> "Vector2":
        m = np.array([x, y, 1.0], dtype=np.float64)
        return Vector2(m)

    @staticmethod
    def zero() -> "Vector2":
        return Vector2.new(0.0, 0.0)

    @staticmethod
    def forward() -> "Vector2":
        return Vector2.new(1.0, 0.0)

    @staticmethod
    def backward() -> "Vector2":
        return Vector2.new(-1.0, 0.0)

    @staticmethod
    def left() -> "Vector2":
        return Vector2.new(0.0, 1.0)

    @staticmethod
    def right() -> "Vector2":
        return Vector2.new(0.0, -1.0)

    @staticmethod
    def from_point(p: Point2) -> "Vector2":
        return Vector2(p._matrix)

    @staticmethod
    def from_ros_msg(m: geometry_msgs.Vector3) -> "Vector2":
        return Vector2.new(m.x, m.y)

    def to_ros_msg(self) -> geometry_msgs.Vector3:
        return geometry_msgs.Vector3(x=self.x(), y=self.y(), z=0.0)

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            matrix = self._matrix * other
            matrix[2] = 1.0
            return Vector2(matrix)
        raise TypeError(
            f"Unsupported operand types for *: '{type(self)}' and '{type(other)}'"
        )

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return self.__mul__(other)
        raise TypeError(
            f"Unsupported operand types for *: '{type(self)}' and '{type(other)}'"
        )

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            matrix = self._matrix / other
            matrix[2] = 1.0
            return Vector2(matrix)
        raise TypeError(
            f"Unsupported operand types for /: '{type(self)}' and '{type(other)}'"
        )

    def __add__(self, other):
        if isinstance(other, Vector2):
            matrix = self._matrix + other._matrix
            matrix[2] = 1.0
            return Vector2(matrix)
        if isinstance(other, Point2):
            matrix = self._matrix + other._matrix
            matrix[2] = 1.0
            return Point2(matrix)
        raise TypeError(
            f"Unsupported operand types for +: '{type(self)}' and '{type(other)}'"
        )

    def __sub__(self, other):
        if isinstance(other, Vector2):
            matrix = self._matrix - other._matrix
            matrix[2] = 1.0
            return Vector2(matrix)
        raise TypeError(
            f"Unsupported operand types for -: '{type(self)}' and '{type(other)}'"
        )

    def __neg__(self) -> "Vector2":
        matrix = -self._matrix
        matrix[2] = 1.0
        return Vector2(matrix)


@dataclass(init=False, eq=False)
class Transform2D:
    """Homogeneous 2 dimensional transformation matrix

    Based on https://alexsm.com/homogeneous-transforms/

    ## Examples:
    ### Transform a Vector2
    ```python
    v = Vector2.new(1.0, 0.0)
    t = Transform2D.new_rotation(math.pi/2.0)
    v_transformed = t * v
    ```
    v_transformed is (0.0, 1.0)

    Note that Vectors are only directions/offsets and ignore translations.
    """

    # Matrix with shape (3, 3)
    _matrix: npt.NDArray[np.float64]

    def __init__(self, matrix: npt.NDArray[np.float64]) -> None:
        assert matrix.shape == (
            3,
            3,
        ), "Transformation matrix must be a homogenous 3x3 matrix"
        self._matrix = matrix

    def translation(self) -> Vector2:
        """Returns only the translation that this Transform applies

        Returns:
            Vector2: translation
        """
        m = self._matrix[:, 2]
        m = m / m[2]
        return Vector2(m)

    def rotation(self) -> float:
        """Returns only the rotation that this Transform applies

        Returns:
            float: rotation angle in radians

            - angle > 0: CCW
            - angle < 0: CW
        """
        # Atan2(y, x) is angle to Vector(x, y)
        # => matrix[1, 0] = sin(a) = y; self._matrix[0, 0] = cos(a) = x
        return np.arctan2(self._matrix[1, 0], self._matrix[0, 0])

    def inverse(self) -> "Transform2D":
        """Returns an inverted Transformation matrix

        Returns:
            Transform2D: Inverted Transformation matrix
        """
        return Transform2D(np.linalg.inv(self._matrix))

    @staticmethod
    def identity() -> "Transform2D":
        """Returns the identity transform (no transformation)

        Returns:
            Transform2D: Identity transform
        """
        return Transform2D(matrix=np.eye(3, dtype=np.float64))

    @staticmethod
    def new_rotation(angle: float) -> "Transform2D":
        """Returns a transformation matrix consisting of a rotation around `angle`

        Args:
            angle (float): Rotation angle in radians

            - angle > 0: CCW
            - angle < 0: CW
        """
        c = np.cos(angle)
        s = np.sin(angle)

        rot_matrix = np.array([[c, -s], [s, c]])
        transform = Transform2D.identity()
        transform._matrix[:2, :2] = rot_matrix
        return transform

    @staticmethod
    def new_translation(v: Vector2) -> "Transform2D":
        """Returns a transformation matrix consisting of a translation along `v`

        Args:
            v (Vector2): Translation vector
        """
        transform = Transform2D.identity()
        transform._matrix[:2, 2] = v._matrix[:2]
        return transform

    @staticmethod
    def new_rotation_translation(angle: float, v: Vector2) -> "Transform2D":
        """Returns a transformation matrix consisting of first a rotation around `angle`
        and then a translation along `v`.

        Args:
            angle (float): Rotation angle in radians
            - angle > 0: CCW
            - angle < 0: CW

            v (Vector2): Translation vector
        """
        transform = Transform2D.new_rotation(angle)
        transform._matrix[:2, 2] = v._matrix[:2]
        return transform

    @staticmethod
    def from_ros_msg(m: msg.Transform2D) -> "Transform2D":
        matrix = np.array(m.matrix, dtype=np.float64).reshape((3, 3), order="C")
        return Transform2D(matrix)

    def to_ros_msg(self) -> msg.Transform2D:
        t = tuple(self._matrix.flatten(order="C"))
        return msg.Transform2D(matrix=t)

    def __mul__(self, other):
        if isinstance(other, Transform2D):
            return Transform2D(np.matmul(self._matrix, other._matrix))
        if isinstance(other, Point2):
            m = np.matmul(self._matrix, other._matrix)
            m = m / m[2]
            return Point2(m)
        if isinstance(other, Vector2):
            matrix = self._matrix.copy()
            matrix[:2, 2] = 0.0
            m = np.matmul(matrix, other._matrix)
            m = m / m[2]
            return Vector2(m)
        if isinstance(other, shapely.Geometry):

            def transform_shapely_point(
                points: npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
                assert (
                    len(points.shape) == 2 and points.shape[1] == 2
                ), "Coords must be 2 dimensional"
                num_points = points.shape[0]
                ones = np.ones(shape=(num_points, 1), dtype=np.float64)
                points_homo = np.concatenate(
                    (points, ones),
                    axis=1,
                )
                points_homo = points_homo[:, None, :]
                points_homo = np.transpose(points_homo, axes=(0, 2, 1))
                transformed = np.matmul(self._matrix, points_homo)
                transformed = np.transpose(transformed, axes=(0, 2, 1))
                transformed = np.squeeze(transformed, 1)
                transformed = transformed / transformed[:, 2, None]
                return transformed[:, :2]

            return shapely.transform(other, transformation=transform_shapely_point)
        raise TypeError(
            f"Unsupported operand types for *: '{type(self)}' and '{type(other)}'"
        )

    def __eq__(self, value) -> bool:
        if type(self) is type(value):
            return (self._matrix == value._matrix).all()
        return False

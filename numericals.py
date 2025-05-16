# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Visual/numerical representation of geometry by assigning concrete coordinates to points.
Methods to find things such as intersections between lines, tangents to circles; as well as drawing utils.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Union

import geometry as gm
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from numpy.random import uniform as unif  # pylint: disable=g-importing-member
import itertools
import random
from matplotlib.patches import Arc
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import re
import sympy as sp
ATOM = 1e-12


# Some variables are there for better code reading.
# pylint: disable=unused-assignment
# pylint: disable=unused-argument
# pylint: disable=unused-variable

# Naming in geometry is a little different
# we stick to geometry naming to better read the code.
# pylint: disable=invalid-name


class Point:
  """Numerical point."""

  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __lt__(self, other: Point) -> bool:
    return (self.x, self.y) < (other.x, other.y)

  def __gt__(self, other: Point) -> bool:
    return (self.x, self.y) > (other.x, other.y)

  def __add__(self, p: Point) -> Point:
    return Point(self.x + p.x, self.y + p.y)

  def __sub__(self, p: Point) -> Point:
    return Point(self.x - p.x, self.y - p.y)

  def __mul__(self, f: float) -> Point:
    return Point(self.x * f, self.y * f)

  def __rmul__(self, f: float) -> Point:
    return self * f

  def __truediv__(self, f: float) -> Point:
    return Point(self.x / f, self.y / f)

  def __floordiv__(self, f: float) -> Point:
    div = self / f  # true div
    return Point(int(div.x), int(div.y))

  def __str__(self) -> str:
    return 'P({},{})'.format(self.x, self.y)

  def close(self, point: Point, tol: float = 1e-12) -> bool:
    return abs(self.x - point.x) < tol and abs(self.y - point.y) < tol

  def midpoint(self, p: Point) -> Point:
    return Point(0.5 * (self.x + p.x), 0.5 * (self.y + p.y))

  def distance(self, p: Union[Point, Line, Circle]) -> float:
    if isinstance(p, Line):
      return p.distance(self)
    if isinstance(p, Circle):
      return abs(p.radius - self.distance(p.center))
    dx = self.x - p.x
    dy = self.y - p.y
    return np.sqrt(dx * dx + dy * dy)

  def distance2(self, p: Point) -> float:
    if isinstance(p, Line):
      return p.distance(self)
    dx = self.x - p.x
    dy = self.y - p.y
    return dx * dx + dy * dy

  def rotatea(self, ang: float) -> Point:
    sinb, cosb = np.sin(ang), np.cos(ang)
    return self.rotate(sinb, cosb)

  def rotate(self, sinb: float, cosb: float) -> Point:
    x, y = self.x, self.y
    return Point(x * cosb - y * sinb, x * sinb + y * cosb)

  def flip(self) -> Point:
    return Point(-self.x, self.y)

  def perpendicular_line(self, line: Line) -> Line:
    return line.perpendicular_line(self)

  def foot(self, line: Line) -> Point:
    if isinstance(line, Line):
      l = line.perpendicular_line(self)
      return line_line_intersection(l, line)
    elif isinstance(line, Circle):
      c, r = line.center, line.radius
      return c + (self - c) * r / self.distance(c)
    raise ValueError('Dropping foot to weird type {}'.format(type(line)))

  def parallel_line(self, line: Line) -> Line:
    return line.parallel_line(self)

  def norm(self) -> float:
    return np.sqrt(self.x**2 + self.y**2)

  def cos(self, other: Point) -> float:
    x, y = self.x, self.y
    a, b = other.x, other.y
    return (x * a + y * b) / self.norm() / other.norm()

  def dot(self, other: Point) -> float:
    return self.x * other.x + self.y * other.y

  def sign(self, line: Line) -> int:
    return line.sign(self)

  def is_same(self, other: Point) -> bool:
    return self.distance(other) <= ATOM


class Line:
  """Numerical line."""

  def __init__(
      self,
      p1: Point = None,
      p2: Point = None,
      coefficients: tuple[int, int, int] = None,
  ):
    if p1 is None and p2 is None and coefficients is None:
      self.coefficients = None, None, None
      return

    a, b, c = coefficients or (
        p1.y - p2.y,
        p2.x - p1.x,
        p1.x * p2.y - p2.x * p1.y,
    )

    # Make sure a is always positive (or always negative for that matter)
    # With a == 0, Assuming a = +epsilon > 0
    # Then b such that ax + by = 0 with y>0 should be negative.
    if a < 0.0 or a == 0.0 and b > 0.0:
      a, b, c = -a, -b, -c

    self.coefficients = a, b, c

  def parallel_line(self, p: Point) -> Line:
    a, b, _ = self.coefficients
    return Line(coefficients=(a, b, -a * p.x - b * p.y))  # pylint: disable=invalid-unary-operand-type

  def perpendicular_line(self, p: Point) -> Line:
    a, b, _ = self.coefficients
    return Line(p, p + Point(a, b))

  def greater_than(self, other: Line) -> bool:
    a, b, _ = self.coefficients
    x, y, _ = other.coefficients
    # b/a > y/x
    return b * x > a * y

  def __gt__(self, other: Line) -> bool:
    return self.greater_than(other)

  def __lt__(self, other: Line) -> bool:
    return other.greater_than(self)

  def same(self, other: Line) -> bool:
    a, b, c = self.coefficients
    x, y, z = other.coefficients
    return close_enough(a * y, b * x) and close_enough(b * z, c * y)

  def equal(self, other: Line) -> bool:
    a, b, _ = self.coefficients
    x, y, _ = other.coefficients
    # b/a == y/x
    return b * x == a * y

  def less_than(self, other: Line) -> bool:
    a, b, _ = self.coefficients
    x, y, _ = other.coefficients
    # b/a > y/x
    return b * x < a * y

  def intersect(self, obj: Union[Line, Circle]) -> tuple[Point, ...]:
    if isinstance(obj, Line):
      return line_line_intersection(self, obj)
    if isinstance(obj, Circle):
      return line_circle_intersection(self, obj)

  def distance(self, p: Point) -> float:
    a, b, c = self.coefficients
    return abs(self(p.x, p.y)) / math.sqrt(a * a + b * b)

  def __call__(self, x: Point, y: Point = None) -> float:
    if isinstance(x, Point) and y is None:
      return self(x.x, x.y)
    a, b, c = self.coefficients
    return x * a + y * b + c

  def is_parallel(self, other: Line) -> bool:
    a, b, _ = self.coefficients
    x, y, _ = other.coefficients
    return abs(a * y - b * x) < ATOM

  def is_perp(self, other: Line) -> bool:
    a, b, _ = self.coefficients
    x, y, _ = other.coefficients
    return abs(a * x + b * y) < ATOM

  def cross(self, other: Line) -> float:
    a, b, _ = self.coefficients
    x, y, _ = other.coefficients
    return a * y - b * x

  def dot(self, other: Line) -> float:
    a, b, _ = self.coefficients
    x, y, _ = other.coefficients
    return a * x + b * y

  def point_at(self, x: float = None, y: float = None) -> Optional[Point]:
    """Get a point on line closest to (x, y)."""
    a, b, c = self.coefficients
    # ax + by + c = 0
    if x is None and y is not None:
      if a != 0:
        return Point((-c - b * y) / a, y)  # pylint: disable=invalid-unary-operand-type
      else:
        return None
    elif x is not None and y is None:
      if b != 0:
        return Point(x, (-c - a * x) / b)  # pylint: disable=invalid-unary-operand-type
      else:
        return None
    elif x is not None and y is not None:
      if a * x + b * y + c == 0.0:
        return Point(x, y)
    return None

  def diff_side(self, p1: Point, p2: Point) -> Optional[bool]:
    d1 = self(p1.x, p1.y)
    d2 = self(p2.x, p2.y)
    if d1 == 0 or d2 == 0:
      return None
    return d1 * d2 < 0

  def same_side(self, p1: Point, p2: Point) -> Optional[bool]:
    d1 = self(p1.x, p1.y)
    d2 = self(p2.x, p2.y)
    if d1 == 0 or d2 == 0:
      return None
    return d1 * d2 > 0

  def sign(self, point: Point) -> int:
    s = self(point.x, point.y)
    if s > 0:
      return 1
    elif s < 0:
      return -1
    return 0

  def is_same(self, other: Line) -> bool:
    a, b, c = self.coefficients
    x, y, z = other.coefficients
    return abs(a * y - b * x) <= ATOM and abs(b * z - c * y) <= ATOM

  def sample_within(self, points: list[Point], n: int = 5) -> list[Point]:
    """Sample a point within the boundary of points."""
    center = sum(points, Point(0.0, 0.0)) * (1.0 / len(points))
    radius = max([p.distance(center) for p in points])
    if close_enough(center.distance(self), radius):
      center = center.foot(self)
    a, b = line_circle_intersection(self, Circle(center.foot(self), radius))

    result = None
    best = -1.0
    for _ in range(n):
      rand = unif(0.0, 1.0)
      x = a + (b - a) * rand
      mind = min([x.distance(p) for p in points])
      if mind > best:
        best = mind
        result = x

    return [result]


class InvalidLineIntersectError(Exception):
  pass


class HalfLine(Line):
  """Numerical ray."""

  def __init__(self, tail: Point, head: Point):  # pylint: disable=super-init-not-called
    self.line = Line(tail, head)
    self.coefficients = self.line.coefficients
    self.tail = tail
    self.head = head

  def intersect(self, obj: Union[Line, HalfLine, Circle, HoleCircle]) -> Point:
    if isinstance(obj, (HalfLine, Line)):
      return line_line_intersection(self.line, obj)

    exclude = [self.tail]
    if isinstance(obj, HoleCircle):
      exclude += [obj.hole]

    a, b = line_circle_intersection(self.line, obj)
    if any([a.close(x) for x in exclude]):
      return b
    if any([b.close(x) for x in exclude]):
      return a

    v = self.head - self.tail
    va = a - self.tail
    vb = b - self.tail
    if v.dot(va) > 0:
      return a
    if v.dot(vb) > 0:
      return b
    raise InvalidLineIntersectError()

  def sample_within(self, points: list[Point], n: int = 5) -> list[Point]:
    center = sum(points, Point(0.0, 0.0)) * (1.0 / len(points))
    radius = max([p.distance(center) for p in points])
    if close_enough(center.distance(self.line), radius):
      center = center.foot(self)
    a, b = line_circle_intersection(self, Circle(center.foot(self), radius))

    if (a - self.tail).dot(self.head - self.tail) > 0:
      a, b = self.tail, a
    else:
      a, b = self.tail, b  # pylint: disable=self-assigning-variable

    result = None
    best = -1.0
    for _ in range(n):
      x = a + (b - a) * unif(0.0, 1.0)
      mind = min([x.distance(p) for p in points])
      if mind > best:
        best = mind
        result = x

    return [result]


def _perpendicular_bisector(p1: Point, p2: Point) -> Line:
  midpoint = (p1 + p2) * 0.5
  return Line(midpoint, midpoint + Point(p2.y - p1.y, p1.x - p2.x))


def same_sign(
    a: Point, b: Point, c: Point, d: Point, e: Point, f: Point
) -> bool:
  a, b, c, d, e, f = map(lambda p: p.sym, [a, b, c, d, e, f])
  ab, cb = a - b, c - b
  de, fe = d - e, f - e
  return (ab.x * cb.y - ab.y * cb.x) * (de.x * fe.y - de.y * fe.x) > 0


class Circle:
  """Numerical circle."""

  def __init__(
      self,
      center: Optional[Point] = None,
      radius: Optional[float] = None,
      p1: Optional[Point] = None,
      p2: Optional[Point] = None,
      p3: Optional[Point] = None,
  ):
    if not center:
      if not (p1 and p2 and p3):
        self.center = self.radius = self.r2 = None
        return
        # raise ValueError('Circle without center need p1 p2 p3')

      l12 = _perpendicular_bisector(p1, p2)
      l23 = _perpendicular_bisector(p2, p3)
      center = line_line_intersection(l12, l23)

    self.center = center
    self.a, self.b = center.x, center.y

    if not radius:
      if not (p1 or p2 or p3):
        raise ValueError('Circle needs radius or p1 or p2 or p3')
      p = p1 or p2 or p3
      self.r2 = (self.a - p.x) ** 2 + (self.b - p.y) ** 2
      self.radius = math.sqrt(self.r2)
    else:
      self.radius = radius
      self.r2 = radius * radius

  def intersect(self, obj: Union[Line, Circle]) -> tuple[Point, ...]:
    if isinstance(obj, Line):
      return obj.intersect(self)
    if isinstance(obj, Circle):
      return circle_circle_intersection(self, obj)

  def sample_within(self, points: list[Point], n: int = 5) -> list[Point]:
    """Sample a point within the boundary of points."""
    result = None
    best = -1.0
    for _ in range(n):
      ang = unif(0.0, 2.0) * np.pi
      x = self.center + Point(np.cos(ang), np.sin(ang)) * self.radius
      mind = min([x.distance(p) for p in points])
      if mind > best:
        best = mind
        result = x

    return [result]


class HoleCircle(Circle):
  """Numerical circle with a missing point."""

  def __init__(self, center: Point, radius: float, hole: Point):
    super().__init__(center, radius)
    self.hole = hole

  def intersect(self, obj: Union[Line, HalfLine, Circle, HoleCircle]) -> Point:
    if isinstance(obj, Line):
      a, b = line_circle_intersection(obj, self)
      if a.close(self.hole):
        return b
      return a
    if isinstance(obj, HalfLine):
      return obj.intersect(self)
    if isinstance(obj, Circle):
      a, b = circle_circle_intersection(obj, self)
      if a.close(self.hole):
        return b
      return a
    if isinstance(obj, HoleCircle):
      a, b = circle_circle_intersection(obj, self)
      if a.close(self.hole) or a.close(obj.hole):
        return b
      return a


def solve_quad(a: float, b: float, c: float) -> tuple[float, float]:
  """Solve a x^2 + bx + c = 0."""
  a = 2 * a
  d = b * b - 2 * a * c
  if d < 0:
    return None  # the caller should expect this result.

  y = math.sqrt(d)
  return (-b - y) / a, (-b + y) / a


def circle_circle_intersection(c1: Circle, c2: Circle) -> tuple[Point, Point]:
  """Returns a pair of Points as intersections of c1 and c2."""
  # circle 1: (x0, y0), radius r0
  # circle 2: (x1, y1), radius r1
  x0, y0, r0 = c1.a, c1.b, c1.radius
  x1, y1, r1 = c2.a, c2.b, c2.radius

  d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
  if d == 0:
    raise InvalidQuadSolveError()

  a = (r0**2 - r1**2 + d**2) / (2 * d)
  h = r0**2 - a**2
  if h < 0:
    raise InvalidQuadSolveError()
  h = np.sqrt(h)
  x2 = x0 + a * (x1 - x0) / d
  y2 = y0 + a * (y1 - y0) / d
  x3 = x2 + h * (y1 - y0) / d
  y3 = y2 - h * (x1 - x0) / d
  x4 = x2 - h * (y1 - y0) / d
  y4 = y2 + h * (x1 - x0) / d

  return Point(x3, y3), Point(x4, y4)


class InvalidQuadSolveError(Exception):
  pass


def line_circle_intersection(line: Line, circle: Circle) -> tuple[Point, Point]:
  """Returns a pair of points as intersections of line and circle."""
  a, b, c = line.coefficients
  r = float(circle.radius)
  center = circle.center
  p, q = center.x, center.y

  if b == 0:
    x = -c / a
    x_p = x - p
    x_p2 = x_p * x_p
    y = solve_quad(1, -2 * q, q * q + x_p2 - r * r)
    if y is None:
      raise InvalidQuadSolveError()
    y1, y2 = y
    return (Point(x, y1), Point(x, y2))

  if a == 0:
    y = -c / b
    y_q = y - q
    y_q2 = y_q * y_q
    x = solve_quad(1, -2 * p, p * p + y_q2 - r * r)
    if x is None:
      raise InvalidQuadSolveError()
    x1, x2 = x
    return (Point(x1, y), Point(x2, y))

  c_ap = c + a * p
  a2 = a * a
  y = solve_quad(
      a2 + b * b, 2 * (b * c_ap - a2 * q), c_ap * c_ap + a2 * (q * q - r * r)
  )
  if y is None:
    raise InvalidQuadSolveError()
  y1, y2 = y

  return Point(-(b * y1 + c) / a, y1), Point(-(b * y2 + c) / a, y2)


def _check_between(a: Point, b: Point, c: Point) -> bool:
  """Whether a is between b & c."""
  return (a - b).dot(c - b) > 0 and (a - c).dot(b - c) > 0


def circle_segment_intersect(
    circle: Circle, p1: Point, p2: Point
) -> list[Point]:
  l = Line(p1, p2)
  px, py = line_circle_intersection(l, circle)

  result = []
  if _check_between(px, p1, p2):
    result.append(px)
  if _check_between(py, p1, p2):
    result.append(py)
  return result


def line_segment_intersection(l: Line, A: Point, B: Point) -> Point:  # pylint: disable=invalid-name
  a, b, c = l.coefficients
  x1, y1, x2, y2 = A.x, A.y, B.x, B.y
  dx, dy = x2 - x1, y2 - y1
  alpha = (-c - a * x1 - b * y1) / (a * dx + b * dy)
  return Point(x1 + alpha * dx, y1 + alpha * dy)


def line_line_intersection(l1: Line, l2: Line) -> Point:
  a1, b1, c1 = l1.coefficients
  a2, b2, c2 = l2.coefficients
  # a1x + b1y + c1 = 0
  # a2x + b2y + c2 = 0
  d = a1 * b2 - a2 * b1
  if d == 0:
    raise InvalidLineIntersectError
  return Point((c2 * b1 - c1 * b2) / d, (c1 * a2 - c2 * a1) / d)


def check_too_close(
    newpoints: list[Point], points: list[Point], tol: int = 0.1
) -> bool:
  if not points:
    return False
  avg = sum(points, Point(0.0, 0.0)) * 1.0 / len(points)
  mindist = min([p.distance(avg) for p in points])
  for p0 in newpoints:
    for p1 in points:
      if p0.distance(p1) < tol * mindist:
        return True
  return False


def check_too_far(
    newpoints: list[Point], points: list[Point], tol: int = 4
) -> bool:
  if len(points) < 2:
    return False
  avg = sum(points, Point(0.0, 0.0)) * 1.0 / len(points)
  maxdist = max([p.distance(avg) for p in points])
  for p in newpoints:
    if p.distance(avg) > maxdist * tol:
      return True
  return False


def check_aconst(args: list[Point]) -> bool:
  a, b, c, d, num, den = args
  d = d + a - c
  ang = ang_between(a, b, d)
  if ang < 0:
    ang += np.pi
  return close_enough(ang, num * np.pi / den)


def check(name: str, args: list[Union[gm.Point, Point]]) -> bool:
  """Numerical check that a constraint is satisfied, e.g.,
  equal angle by computing angle with point coordinates substituted.
  """
  if name == 'eqangle6':
    name = 'eqangle'
  elif name == 'eqratio6':
    name = 'eqratio'
  elif name in ['simtri2', 'simtri*']:
    name = 'simtri'
  elif name in ['contri2', 'contri*']:
    name = 'contri'
  elif name == 'para':
    name = 'para_or_coll'
  elif name == 'on_line':
    name = 'coll'
  elif name in ['rcompute', 'acompute']:
    return True
  elif name in ['fixl', 'fixc', 'fixb', 'fixt', 'fixp']:
    return True

  fn_name = 'check_' + name
  if fn_name not in globals():
    return None

  fun = globals()['check_' + name]
  args = [p.num if isinstance(p, gm.Point) else p for p in args]
  return fun(args)


def check_circle(points: list[Point]) -> bool:
  if len(points) != 4:
    return False
  o, a, b, c = points
  oa, ob, oc = o.distance(a), o.distance(b), o.distance(c)
  return close_enough(oa, ob) and close_enough(ob, oc)


def check_coll(points: list[Point]) -> bool:
  a, b = points[:2]
  l = Line(a, b)
  for p in points[2:]:
    if abs(l(p.x, p.y)) > ATOM:
      return False
  return True


def check_ncoll(points: list[Point]) -> bool:
  return not check_coll(points)


def check_sameside(points: list[Point]) -> bool:
  b, a, c, y, x, z = points
  # whether b is to the same side of a & c as y is to x & z
  ba = b - a
  bc = b - c
  yx = y - x
  yz = y - z
  return ba.dot(bc) * yx.dot(yz) > 0


def check_para_or_coll(points: list[Point]) -> bool:
  return check_para(points) or check_coll(points)


def check_para(points: list[Point]) -> bool:
  a, b, c, d = points
  ab = Line(a, b)
  cd = Line(c, d)
  if ab.same(cd):
    return False
  return ab.is_parallel(cd)


def check_perp(points: list[Point]) -> bool:
  a, b, c, d = points
  ab = Line(a, b)
  cd = Line(c, d)
  return ab.is_perp(cd)


def check_cyclic(points: list[Point]) -> bool:
  points = list(set(points))
  (a, b, c), *ps = points
  circle = Circle(p1=a, p2=b, p3=c)
  for d in ps:
    if not close_enough(d.distance(circle.center), circle.radius):
      return False
  return True


def bring_together(
    a: Point, b: Point, c: Point, d: Point
) -> tuple[Point, Point, Point, Point]:
  ab = Line(a, b)
  cd = Line(c, d)
  x = line_line_intersection(ab, cd)
  unit = Circle(center=x, radius=1.0)
  y, _ = line_circle_intersection(ab, unit)
  z, _ = line_circle_intersection(cd, unit)
  return x, y, x, z


def same_clock(
    a: Point, b: Point, c: Point, d: Point, e: Point, f: Point
) -> bool:
  ba = b - a
  cb = c - b
  ed = e - d
  fe = f - e
  return (ba.x * cb.y - ba.y * cb.x) * (ed.x * fe.y - ed.y * fe.x) > 0


def check_const_angle(points: list[Point]) -> bool:
  """Check if the angle is equal to the given constant."""
  a, b, c, d, m, n = points
  a, b, c, d = bring_together(a, b, c, d)
  ba = b - a
  dc = d - c

  a3 = np.arctan2(ba.y, ba.x)
  a4 = np.arctan2(dc.y, dc.x)
  y = a3 - a4

  return close_enough(m / n % 1, y / np.pi % 1)


def check_eqangle(points: list[Point]) -> bool:
  """Check if 8 points make 2 equal angles."""
  a, b, c, d, e, f, g, h = points

  ab = Line(a, b)
  cd = Line(c, d)
  ef = Line(e, f)
  gh = Line(g, h)

  if ab.is_parallel(cd):
    return ef.is_parallel(gh)
  if ef.is_parallel(gh):
    return ab.is_parallel(cd)

  a, b, c, d = bring_together(a, b, c, d)
  e, f, g, h = bring_together(e, f, g, h)

  ba = b - a
  dc = d - c
  fe = f - e
  hg = h - g

  sameclock = (ba.x * dc.y - ba.y * dc.x) * (fe.x * hg.y - fe.y * hg.x) > 0
  if not sameclock:
    ba = ba * -1.0

  a1 = np.arctan2(fe.y, fe.x)
  a2 = np.arctan2(hg.y, hg.x)
  x = a1 - a2

  a3 = np.arctan2(ba.y, ba.x)
  a4 = np.arctan2(dc.y, dc.x)
  y = a3 - a4

  xy = (x - y) % (2 * np.pi)
  return close_enough(xy, 0, tol=1e-11) or close_enough(
      xy, 2 * np.pi, tol=1e-11
  )


def check_eqratio(points: list[Point]) -> bool:
  a, b, c, d, e, f, g, h = points
  ab = a.distance(b)
  cd = c.distance(d)
  ef = e.distance(f)
  gh = g.distance(h)
  return close_enough(ab * gh, cd * ef)


def check_cong(points: list[Point]) -> bool:
  a, b, c, d = points
  return close_enough(a.distance(b), c.distance(d))


def check_midp(points: list[Point]) -> bool:
  a, b, c = points
  return check_coll(points) and close_enough(a.distance(b), a.distance(c))


def check_simtri(points: list[Point]) -> bool:
  """Check if 6 points make a pair of similar triangles."""
  a, b, c, x, y, z = points
  ab = a.distance(b)
  bc = b.distance(c)
  ca = c.distance(a)
  xy = x.distance(y)
  yz = y.distance(z)
  zx = z.distance(x)
  tol = 1e-9
  return close_enough(ab * yz, bc * xy, tol) and close_enough(
      bc * zx, ca * yz, tol
  )


def check_contri(points: list[Point]) -> bool:
  a, b, c, x, y, z = points
  ab = a.distance(b)
  bc = b.distance(c)
  ca = c.distance(a)
  xy = x.distance(y)
  yz = y.distance(z)
  zx = z.distance(x)
  tol = 1e-9
  return (
      close_enough(ab, xy, tol)
      and close_enough(bc, yz, tol)
      and close_enough(ca, zx, tol)
  )


def check_ratio(points: list[Point]) -> bool:
  a, b, c, d, m, n = points
  ab = a.distance(b)
  cd = c.distance(d)
  return close_enough(ab * n, cd * m)


def draw_angle(
    ax: matplotlib.axes.Axes,
    head: Point,
    p1: Point,
    p2: Point,
    color: Any = 'red',
    alpha: float = 0.5,
    frac: float = 1.0,
) -> None:
  """Draw an angle on plt ax."""
  d1 = p1 - head
  d2 = p2 - head

  a1 = np.arctan2(float(d1.y), float(d1.x))
  a2 = np.arctan2(float(d2.y), float(d2.x))
  a1, a2 = a1 * 180 / np.pi, a2 * 180 / np.pi
  a1, a2 = a1 % 360, a2 % 360

  if a1 > a2:
    a1, a2 = a2, a1

  if a2 - a1 > 180:
    a1, a2 = a2, a1

  b1, b2 = a1, a2
  if b1 > b2:
    b2 += 360
  d = b2 - b1
  # if d >= 90:
  #   return

  scale = min(2.0, 90 / d)
  scale = max(scale, 0.4)
  fov = matplotlib.patches.Wedge(
      (float(head.x), float(head.y)),
      unif(0.075, 0.125) * scale * frac,
      a1,
      a2,
      color=color,
      alpha=alpha,
  )
  ax.add_artist(fov)

def annotate_angle(ax, angle, color='blue', offset=0.2, arc_radius=0.02, arc_width=0.6, show_degree=False, linestyle='-'):
    """
    在给定的 ax 上，在角 head（顶点，由 p1 和 p2 构成）处标注角度值，并绘制角度符号（弧线）。
    :param ax: Matplotlib 轴对象
    :param angle: 角的三个点，格式为 {"head": p0, "p1": p1, "p2": p2}，其中点包含 x, y 属性
    :param color: 标注文字和弧线的颜色
    :param offset: 文字相对于顶点的偏移距离
    :param arc_radius: 角度弧线的半径
    :param arc_width: 角度弧线的线宽
    """
    
    p1 = angle["p1"]
    head = angle["head"]
    p2 = angle["p2"]
    # print(dir(p1))
    v1 = np.array([p1.x - head.x, p1.y - head.y])
    v2 = np.array([p2.x - head.x, p2.y - head.y])
    
    # 
    len_v1 = np.linalg.norm(v1)
    len_v2 = np.linalg.norm(v2)
    
    # 
    angle_rad = math.acos(np.dot(v1, v2) / (len_v1 * len_v2))
    angle_deg = np.degrees(angle_rad)

    # 
    a1 = np.degrees(math.atan2(v1[1], v1[0])) % 360
    a2 = np.degrees(math.atan2(v2[1], v2[0])) % 360

    # 
    start_angle = min(a1, a2)
    angle_span = abs(a1 - a2)
    if angle_span > 180:
        start_angle = max(a1, a2)
        angle_span = 360 - angle_span


    if abs(angle_deg - 90) < 1e-1:
        # 单位向量方向
        u1 = v1 / len_v1
        u2 = v2 / len_v2
        
        # 计算正方形边长：取较短边的一半或原来的计算方法中的较小值
        original_side_len = (arc_radius / np.sqrt(2)) * 1.0
        half_shorter_side = min(len_v1, len_v2) / 3
        side_len = min(original_side_len, half_shorter_side)

        # 直角框的三个角点
        p0 = np.array([head.x, head.y])
        p1 = p0 + u1 * side_len
        p2 = p0 + u2 * side_len
        p3 = p1 + u2 * side_len  # p3 completes the square

        # 画小正方形
        ax.plot([p1[0], p3[0]], [p1[1], p3[1]], color=color, linewidth=0.6)
        ax.plot([p3[0], p2[0]], [p3[1], p2[1]], color=color, linewidth=0.6)

    else:
      if show_degree:
        # 正常画圆弧
            # 计算适应线段长度的半径：取较短边的一半或原来的计算方法中的较小值
        half_shorter_side = min(len_v1, len_v2) / 3
        arc_radius = max(arc_radius, half_shorter_side)
        arc = Arc((head.x, head.y), arc_radius*2, arc_radius*2,
                  angle=0, theta1=start_angle, theta2=start_angle + angle_span,
                  color=color, linewidth=arc_width)
        ax.add_patch(arc)

        # 计算文本位置（放在弧线或角附近）
        mid_angle = start_angle + angle_span / 2
        text_x = head.x + offset * math.cos(math.radians(mid_angle))
        text_y = head.y + offset * math.sin(math.radians(mid_angle))
        
        ax.text(text_x, text_y, f"{angle_deg:.1f}°", color=color, fontsize=8, ha='center')
      else:
                # 画第一个圆弧（内层）
        arc1 = Arc((head.x, head.y), arc_radius * 2, arc_radius * 2,
                  angle=0, theta1=start_angle, theta2=start_angle + angle_span,
                  color=color, linewidth=arc_width, linestyle = linestyle)
        ax.add_patch(arc1)

        # 画第二个圆弧（外层，半径稍大避免重叠）
        arc2_radius = arc_radius + 0.03  # 外层圆弧半径
        arc2 = Arc((head.x, head.y), arc2_radius * 2, arc2_radius * 2,
                  angle=0, theta1=start_angle, theta2=start_angle + angle_span,
                  color=color, linewidth=arc_width, linestyle = linestyle)
        ax.add_patch(arc2)
        # Get coordinates of start and end points for the inner arc
        start_x1 = head.x + arc_radius * np.cos(np.radians(start_angle))
        start_y1 = head.y + arc_radius * np.sin(np.radians(start_angle))
        end_x1 = head.x + arc_radius * np.cos(np.radians(start_angle + angle_span))
        end_y1 = head.y + arc_radius * np.sin(np.radians(start_angle + angle_span))

        # Get coordinates of start and end points for the outer arc
        start_x2 = head.x + arc2_radius * np.cos(np.radians(start_angle))
        start_y2 = head.y + arc2_radius * np.sin(np.radians(start_angle))
        end_x2 = head.x + arc2_radius * np.cos(np.radians(start_angle + angle_span))
        end_y2 = head.y + arc2_radius * np.sin(np.radians(start_angle + angle_span))

        # Plot the points
        ax.plot(start_x1, start_y1, 'o', color=color, markersize=1.2)
        ax.plot(end_x1, end_y1, 'o', color=color, markersize=1.2)
        ax.plot(start_x2, start_y2, 'o', color=color, markersize=1.2)
        ax.plot(end_x2, end_y2, 'o', color=color, markersize=1.2)
from matplotlib.lines import Line2D

# 计算垂线的函数
def draw_perpendicular_line(p1, p2, ax, color, scale=0.06):
    # 计算向量 AB 和单位垂直向量（顺时针旋转90度）
    vec_AB = np.array([p2.x - p1.x, p2.y - p1.y])
    normal_AB = np.array([vec_AB[1], -vec_AB[0]])
    normal_AB = normal_AB / np.linalg.norm(normal_AB) * scale  # 缩放控制短线的长度

    # 绘制垂线
    line_segments = []
    # for i in range(1, 3):  # 在1/3和2/3的位置绘制垂线
    fraction = 1 / 3  # 均匀分布位置
    point_position = (p1.x + fraction * vec_AB[0], p1.y + fraction * vec_AB[1])

    # 获取垂线的两端点坐标
    mid1 = np.array(point_position) + normal_AB
    mid2 = np.array(point_position) - normal_AB

    # 将垂线的端点坐标存储在列表中
    line_segments.append((mid1, mid2))

    # 绘制所有存储的垂线
    for (mid1, mid2) in line_segments:
        ax.plot([mid1[0], mid2[0]], [mid1[1], mid2[1]], color=color, lw=0.4)



# 计算垂线的函数
def draw_perpendicular_lines( p1, p2,  result, ax, colors, result_to_color):
    if result != []:    
      # print("result:", result)
      # 根据result值确定颜色
      group_id = result[0]  # 假设result的第一个元素是组ID
      # 根据result值确定颜色
      if group_id not in result_to_color:
          result_to_color[group_id] = colors[group_id % len(colors)]
      color = result_to_color[group_id]  # 使用组对应的颜色
      
      vec_AB = np.array([p2.x - p1.x, p2.y - p1.y])
      third_AB = vec_AB / 3  # 计算线段的1/3向量
      # 计算单位垂直向量（顺时针旋转90度）
      normal_AB = np.array([vec_AB[1], -vec_AB[0]])
      normal_AB = normal_AB / np.linalg.norm(normal_AB) * 0.08  # 缩放控制短线的长度
      # 1/3位置点
      one_third_point = (p1.x + third_AB[0], p1.y + third_AB[1])
      num_lines = (result[0] + 1) * 3
      # num_lines = 3
      print()
      line_segments = []
      for i in range(result[0] + 1):
          # 计算垂线的位置 - 平均分布在线段上
          fraction = (i + 1) / (num_lines + 1)  # 把线段均匀分为 num_lines+1 份
          point_position = (p1.x + fraction * vec_AB[0], p1.y + fraction * vec_AB[1])

          # 获取垂线的两端点坐标
          mid1 = np.array(point_position) + normal_AB
          mid2 = np.array(point_position) - normal_AB

          # 将垂线的端点坐标存储在列表中
          line_segments.append((mid1, mid2))

      # 绘制所有存储的垂线
      for (mid1, mid2) in line_segments:
          ax.plot([mid1[0], mid2[0]], [mid1[1], mid2[1]], color=color, lw=0.4)


def naming_position(
    ax: matplotlib.axes.Axes, p: Point, lines: list[Line], circles: list[Circle]
) -> tuple[float, float]:
    """Figure out a good naming position on the drawing."""
    r = 0.12
    c = Circle(center=p, radius=r)
    avoid = []
    for p1, p2 in lines:
        try:
            avoid.extend(circle_segment_intersect(c, p1, p2))
        except InvalidQuadSolveError:
            continue
    for x in circles:
        try:
            avoid.extend(circle_circle_intersection(c, x))
        except InvalidQuadSolveError:
            continue

    if not avoid:
        # 使用简单偏移
        name_pos = [p.x + 0.015, p.y + 0.015]
        # ang = np.pi/4
    else:
        # 计算角度
        angs = sorted([ang_of(p, a) for a in avoid])
        angs += [angs[0] + 2 * np.pi]
        angs = [(angs[i + 1] - a, a) for i, a in enumerate(angs[:-1])]

        d, a = max(angs)
        ang = a + d / 2

        # 使用角度计算位置
        name_pos = p + Point(np.cos(ang), np.sin(ang)) * r
        name_pos = [name_pos.x - r / 1.5, name_pos.y - r / 1.5]
    
    # 获取当前坐标轴范围并确保位置在范围内
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # 添加边距，确保标签不会紧贴边缘
    margin = 0.10
    x = max(min(name_pos[0], xlim[1] - margin), xlim[0] + margin)
    y = max(min(name_pos[1], ylim[1] - margin), ylim[0] + margin)
    
    return x, y


# def draw_point(
#     ax: matplotlib.axes.Axes,
#     p: Point,
#     name: str,
#     lines: list[Line],
#     circles: list[Circle],
#     color: Any = 'white',
#     size: float = 10,
# ) -> None:
#   """draw a point."""
#   ax.scatter(p.x, p.y, color=color, s=size)

#   if color == 'white':
#     color = 'lightgreen'
#   else:
#     color = 'grey'

#   name = name.upper()
#   if len(name) > 1:
#     name = name[0] + '_' + name[1:]

#   ax.annotate(
#       name, naming_position(ax, p, lines, circles), color=color, fontsize=3
#   )

def draw_point_reinforce(
    ax: matplotlib.axes.Axes,
    p: Point,
    name: str,
    lines: list[Line],
    circles: list[Circle],
    ncolor: Any = 'white',
    pcolor: Any = 'white',
    size: float = 6,
) -> None:
  """draw a point."""
  ax.scatter(p.x, p.y, color=pcolor, s=size)

  name = name.upper()
  if len(name) > 1:
    name = name[0] + '_' + name[1:]

  ax.annotate(
      name, naming_position(ax, p, lines, circles), color=ncolor, fontsize=5
  )

# 实际绘制线
def _draw_line(
    ax: matplotlib.axes.Axes,
    p1: Point,
    p2: Point,
    color: Any = 'white',
    lw: float = 0.8,
    alpha: float = 0.8,
    ls = '-'
) -> None:
  """Draw a line in matplotlib."""
  # ls = '-'
  # if color == '--':
  #   color = 'black'
  #   ls = '--'

  lx, ly = (p1.x, p2.x), (p1.y, p2.y)
  # print(p1, p2)
  # print("lx", lx)
  # print("ly", ly)
  color = 'black'
  ax.plot(lx, ly, color=color, lw=lw, alpha=alpha, ls=ls)

def check_line_name(line_name, segment_indices):
      
    # Check if line_name matches any extracted segment
    for segment, indices in segment_indices.items():
        if sorted(line_name) == sorted(segment):
            # print(f"Found {line_name} in sublist(s) {indices}")
            return indices
    return []

def draw_line(
    ax: matplotlib.axes.Axes, line: Line, color: Any = 'white',  label_color = Any, show_len: bool = True,  ls = '-',segment_indices =  Any
) -> tuple[Point, Point]:
  """Draw a line."""
  points_all = line.neighbors(gm.Point)
  
  line_name=""
  for p in points_all:
    line_name+=p.name.upper()
  # print(len(points))
  if len(points_all) <= 1:
    return
    # return None, None, None, None  # 返回默认值
  # poinnamets = [p.name for p in points]
  # print(poinnamets)
  # if len(line_name) == 3:
    # return None, None, None, None  # 返回默认值
  # print(points[0])
  points = [p.num for p in points_all]
  p1 = points[0]
  p2 = points[1]

  if len(points) ==2:
    # p1, p2 = points[:2]
    _draw_line(ax, p1, p2, color, ls=ls)

  # p1, p2 = pmin[0], pmax[0]
  if len(points) >=3:
    pmin, pmax = (p1, 0.0), (p2, (p2 - p1).dot(p2 - p1))
    points_dict = dict(zip([p.name for p in points_all], points))
    for p in points:
      v = (p - p1).dot(p2 - p1)
      if v < pmin[1]:
        pmin = p, v
      if v > pmax[1]:
        pmax = p, v
    ls = "--"
    # third_point = [p for p in points if p is not p1 and p is not p2][0]
    _draw_line(ax, pmin[0], pmax[0], color, ls=ls)

 # 定义一组鲜明的颜色
  colors = list(mcolors.TABLEAU_COLORS.values())

  # 创建一个结果值到颜色的映射
  result_to_color = {}
  if segment_indices:
    line_segments = []
    if len(points) >=3:
            # 获取字典中的所有键
      keys = list(points_dict.keys())

      # 使用itertools.combinations生成所有两两组合
      line_segments = {f'{key1.upper()}{key2.upper()}': (points_dict[key1], points_dict[key2]) 
                 for key1, key2 in itertools.combinations(keys, 2)}
      for line_name in line_segments:
        print("line_name:", line_name)
        result = check_line_name(line_name, segment_indices)
        print("result:", result)
        p1 = line_segments[line_name][0]
        p2 = line_segments[line_name][1]
        draw_perpendicular_lines( p1, p2, result, ax, colors, result_to_color)
    elif len(points) ==2:# if any(sorted(line_name) == sorted(x) for x in all_segments):
      result = check_line_name(line_name, segment_indices)

      draw_perpendicular_lines( p1, p2,  result, ax, colors, result_to_color)




  if show_len:
      print(line_name)
      if line_name not in {"AB", "BC", "AE", "CE", "BF", "BA","CB", "EA", "EC", "FB", "AD", "DA", "CD", "DC"}:
          return p1, p2, None, None



      dx = p2.x - p1.x
      dy = p2.y - p1.y
      line_len = math.sqrt(dx ** 2 + dy ** 2)
      # Store AB's length as a property of the function itself
      if line_name in {"AB", "BA"}:
          draw_line.AB_length = line_len
      # Store AB's length as a property of the function itself
      if line_name in {"AD", "DA"}:
          draw_line.AD_length = line_len         
            # Check if this is CD/DC and compare with AB's length
      if line_name in {"CD", "DC"} and hasattr(draw_line, 'AB_length'):
          if abs(line_len - draw_line.AB_length) < 1e-10:
              return p1, p2, None, None
      if line_name in {"CB", "BC"} and hasattr(draw_line, 'AD_length'):
          if abs(line_len - draw_line.AD_length) < 1e-10:
              return p1, p2, None, None      

      
      name = line_name + " = " + "{:.2f}".format(line_len)
      name = line_name + " = " + "{:.2f}".format(line_len)

      # 线段中点
      mid_x = (p1.x + p2.x) / 2
      mid_y = (p1.y + p2.y) / 2

      # 单位方向向量
      dir_len = math.sqrt(dx ** 2 + dy ** 2)
      dx_unit = dx / dir_len
      dy_unit = dy / dir_len

      # 法向量方向（垂直线段）
      nx = -dy_unit
      ny = dx_unit

      scale_factor = 0.8
      # Calculate adaptive font size
      # You can adjust these parameters to suit your needs
      min_fontsize = 3
      max_fontsize = 5
      fontsize = min(max(min_fontsize, line_len * scale_factor), max_fontsize)
      # 偏移量
      offset = 3  # points
      angle = math.degrees(math.atan2(dy, dx))

      # ✅ 自动修正旋转角度：确保文字不颠倒
      if angle > 90 or angle < -90:
          angle += 180
      color_choice = ['red', 'green', 'blue', 'magenta', 'purple']
      ax.annotate(
          name,
          (mid_x, mid_y),
          xytext=(nx * offset, ny * offset),
          textcoords='offset points',
          fontsize=fontsize,
          color= random.choice([length_color for length_color in color_choice if length_color != color]),
          ha='center',
          va='center',
          rotation=angle,
          rotation_mode='anchor',
      )      
      return p1, p2, line_len, line_name
  else:
    return p1, p2, None, None


def _draw_circle(
    ax: matplotlib.axes.Axes, c: Circle, color: Any = 'cyan', lw: float = 0.8
) -> None:
  ls = '-'
  if color == '--':
    color = 'black'
    ls = '--'

  ax.add_patch(
      plt.Circle(
          (c.center.x, c.center.y),
          c.radius,
          color=color,
          alpha=0.8,
          fill=False,
          lw=lw,
          ls=ls,
      )
  )


def draw_circle(
    ax: matplotlib.axes.Axes, circle: Circle, color: Any = 'cyan'
) -> Circle:
  """Draw a circle."""
  if circle.num is not None:
    circle = circle.num
  else:
    points = circle.neighbors(gm.Point)
    if len(points) <= 2:
      return
    points = [p.num for p in points]
    p1, p2, p3 = points[:3]
    circle = Circle(p1=p1, p2=p2, p3=p3)

  _draw_circle(ax, circle, color)
  return circle


def mark_segment(
    ax: matplotlib.axes.Axes, p1: Point, p2: Point, color: Any, alpha: float
) -> None:
  _ = alpha
  x, y = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
  ax.scatter(x, y, color=color, alpha=1.0, marker='o', s=50)


def highlight_angle(
    ax: matplotlib.axes.Axes,
    a: Point,
    b: Point,
    c: Point,
    d: Point,
    color: Any,
    alpha: float,
) -> None:
  """Highlight an angle between ab and cd with (color, alpha)."""
  try:
    a, b, c, d = bring_together(a, b, c, d)
  except:  # pylint: disable=bare-except
    return
  draw_angle(ax, a, b, d, color=color, alpha=alpha, frac=1.0)


def highlight(
    ax: matplotlib.axes.Axes,
    name: str,
    args: list[gm.Point],
    lcolor: Any,
    color1: Any,
    color2: Any,
) -> None:
  """Draw highlights."""
  args = list(map(lambda x: x.num if isinstance(x, gm.Point) else x, args))

  if name == 'cyclic':
    a, b, c, d = args
    _draw_circle(ax, Circle(p1=a, p2=b, p3=c), color=color1, lw=2.0)
  if name == 'coll':
    a, b, c = args
    a, b = max(a, b, c), min(a, b, c)
    _draw_line(ax, a, b, color=color1, lw=2.0)
  if name == 'para':
    a, b, c, d = args
    _draw_line(ax, a, b, color=color1, lw=2.0)
    _draw_line(ax, c, d, color=color2, lw=2.0)
  if name == 'eqangle':
    a, b, c, d, e, f, g, h = args

    x = line_line_intersection(Line(a, b), Line(c, d))
    if b.distance(x) > a.distance(x):
      a, b = b, a
    if d.distance(x) > c.distance(x):
      c, d = d, c
    a, b, d = x, a, c

    y = line_line_intersection(Line(e, f), Line(g, h))
    if f.distance(y) > e.distance(y):
      e, f = f, e
    if h.distance(y) > g.distance(y):
      g, h = h, g
    e, f, h = y, e, g

    _draw_line(ax, a, b, color=lcolor, lw=2.0)
    _draw_line(ax, a, d, color=lcolor, lw=2.0)
    _draw_line(ax, e, f, color=lcolor, lw=2.0)
    _draw_line(ax, e, h, color=lcolor, lw=2.0)
    if color1 == '--':
      color1 = 'red'
    draw_angle(ax, a, b, d, color=color1, alpha=0.5)
    if color2 == '--':
      color2 = 'red'
    draw_angle(ax, e, f, h, color=color2, alpha=0.5)
  if name == 'perp':
    a, b, c, d = args
    _draw_line(ax, a, b, color=color1, lw=2.0)
    _draw_line(ax, c, d, color=color1, lw=2.0)
  if name == 'ratio':
    a, b, c, d, m, n = args
    _draw_line(ax, a, b, color=color1, lw=2.0)
    _draw_line(ax, c, d, color=color2, lw=2.0)
  if name == 'cong':
    a, b, c, d = args
    _draw_line(ax, a, b, color=color1, lw=2.0)
    _draw_line(ax, c, d, color=color2, lw=2.0)
  if name == 'midp':
    m, a, b = args
    _draw_line(ax, a, m, color=color1, lw=2.0, alpha=0.5)
    _draw_line(ax, b, m, color=color2, lw=2.0, alpha=0.5)
  if name == 'eqratio':
    a, b, c, d, m, n, p, q = args
    _draw_line(ax, a, b, color=color1, lw=2.0, alpha=0.5)
    _draw_line(ax, c, d, color=color2, lw=2.0, alpha=0.5)
    _draw_line(ax, m, n, color=color1, lw=2.0, alpha=0.5)
    _draw_line(ax, p, q, color=color2, lw=2.0, alpha=0.5)


HCOLORS = None

def split_to_pairs(vertical_line):
    result = []
    for sublist in vertical_line:
        if len(sublist) > 2:
            # 如果子列表大于2，生成两两组合
            result.extend([list(pair) for pair in itertools.combinations(sublist, 2)])
        else:
            # 如果子列表正好是2个元素，直接添加
            result.append(sublist)
    return result

def get_midpoint(start, end):
    return Point((start.x + end.x) / 2, (start.y + end.y) / 2)

# 获取线段的等分点
def get_segment_points(start, end, num_points):
    x_vals = np.linspace(start.x, end.x, num_points + 1)  # 等分点的 x 坐标
    y_vals = np.linspace(start.y, end.y, num_points + 1)  # 等分点的 y 坐标
    return [Point(x, y) for x, y in zip(x_vals, y_vals)]

# 绘制线段并标记倍数关系
def plot_segment_with_marks(ax, point_dict, seg1_name, seg2_name, factor, color):
    # 从 line name 获取对应的点
    # 将线段名称转换为小写后，提取对应的点名
    seg1_start_key, seg1_end_key = seg1_name[0].lower(), seg1_name[1].lower()
    seg2_start_key, seg2_end_key = seg2_name[0].lower(), seg2_name[1].lower()
    
    # 从 point_dict 获取线段的端点
    seg1_start, seg1_end = point_dict[seg1_start_key], point_dict[seg1_end_key]
    seg2_start, seg2_end = point_dict[seg2_start_key], point_dict[seg2_end_key]
    
    # 计算 AB 的中点
    # midpoint_seg2 = get_midpoint(seg2_start, seg2_end)

    # 获取 AC 的等分点
    seg1_points = get_segment_points(seg1_start, seg1_end, factor)


    # 在 AB 中点添加标记
    # ax.plot(midpoint_seg2.x, midpoint_seg2.y, marker='o', color=color, markerfacecolor=color)
    # 在 AC 的每个等分点添加标记
    for point in seg1_points[1:-1]:  # 标记每个等分点
        ax.plot(point.x, point.y,marker='o', markeredgecolor = color, markerfacecolor='none', markersize=3)  # 绿色圆点标记

    draw_perpendicular_line(seg2_start, seg2_end, ax, color)  # 绘制垂线，使用点1与点0
  
    for i in range(1, len(seg1_points)):  # 从第二个点开始，避免标记起点
        draw_perpendicular_line(seg1_points[i-1], seg1_points[i], ax, color)  # 绘制相邻点之间的垂线

def check_ratio_relationships(segment_lengths, tolerance=1e-6):
    segments = list(segment_lengths.keys())
    relationships = []
    
    # 比较所有可能的线段对
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            seg1 = segments[i]
            seg2 = segments[j]
            len1 = segment_lengths[seg1]
            len2 = segment_lengths[seg2]
            
            # 确保分母不为零
            if len2 == 0 or len1 == 0:
                continue
                
            # 计算比值
            ratio = len1 / len2
            
            # 检查比值是否接近整数
            nearest_int_ratio = round(ratio)
            if abs(ratio - nearest_int_ratio) < tolerance:
                relationships.append((seg1, seg2, nearest_int_ratio, ratio))
                continue
            
            # 检查倒数是否接近整数
            inv_ratio = len2 / len1
            nearest_int_inv_ratio = round(inv_ratio)
            if abs(inv_ratio - nearest_int_inv_ratio) < tolerance:
                relationships.append((seg2, seg1, nearest_int_inv_ratio, inv_ratio))
    
    return relationships

def calculate_segment_length(segment_name, point_dict):
    # 假设线段名称由两个字符组成，分别代表起点和终点
    start_point_name = segment_name[0].lower()  # 取第一个字符作为起点名称
    end_point_name = segment_name[1].lower()    # 取第二个字符作为终点名称
    
    # 从点字典中获取起点和终点对象
    start_point = point_dict[start_point_name]
    end_point = point_dict[end_point_name]
    
    # 获取点的坐标
    x1, y1 = start_point.x, start_point.y  # 假设Point对象有x和y属性
    x2, y2 = end_point.x, end_point.y
    
    # 计算欧几里得距离
    length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    return length

def merge_and_deduplicate(sublist):
    # 展平子列表
    flat_list = [point for pair in sublist for point in pair]
    
    # 找到重复的元素
    unique_elements = []
    duplicates = []
    
    for point in flat_list:
        if flat_list.count(point) == 1:
            unique_elements.append(point)
        elif point not in duplicates:
            duplicates.append(point)
    
    # 将重复元素放到中间
    middle_index = len(unique_elements) // 2
    result = unique_elements[:middle_index] + duplicates + unique_elements[middle_index:]
    
    return result

def is_perpendicular(v1, v2, epsilon=1e-10):
    # 计算点积
    dot_product = v1['x'] * v2['x'] + v1['y'] * v2['y']
    # 使用一个小的阈值来判断是否接近零
    return abs(dot_product) < epsilon

def process_angle_lists(angle_lists, points, point_dict):
    result = []
    
    # print(f"开始处理角度列表: {angle_lists}")
    # print(f"points列表包含 {len(points)} 个点")
    
    # 检查points中的每个元素是否都有name属性
    for i, p in enumerate(points):
        if not hasattr(p, 'name'):
            return []  # 提前返回空结果
    
    for i, sublist in enumerate(angle_lists):
        # print(f"处理子列表 {i}: {sublist}")
        
        # 检查是否是正确的类型
        if not isinstance(sublist, list):
            continue
            
        # 跳过包含'90'的子列表
        if '90' in sublist:
            continue
        
        angles_in_sublist = []
        for j, element in enumerate(sublist):
            if not isinstance(element, str):
                continue
                
            # 修改这里，允许更复杂的元素进行处理
            if len(element) >= 3:  # 允许更长的角度字符串
                p1_name = element[0]
                head_name = element[1]
                p2_name = element[2]
                
                p1 = point_dict.get(p1_name)
                head = point_dict.get(head_name)
                p2 = point_dict.get(p2_name)
                
                angle_dict = {
                    "p1": p1,
                    "head": head,
                    "p2": p2
                }
                
                # 
                if None not in angle_dict.values():
                    angles_in_sublist.append(angle_dict)
        
        if angles_in_sublist:
            result.append(angles_in_sublist)
            print(f"子列表 {i} 处理完成，添加了 {len(angles_in_sublist)} 个角度")
        else:
            print(f"子列表 {i} 没有生成有效的角度")
    
    return result if result else None  # 只有在有有效角度时才返回结果

def convert_angle_expression(expr):
    if 'pi' in expr:
        return None
        
    # 处理类似 m(d(ab)-d(bc)) 的表达式
    import re
    pattern = r'm\(d\((\w+)\)-d\((\w+)\)\)'
    match = re.match(pattern, expr)
    
    if match:
        first_part = match.group(1)
        second_part = match.group(2)
        
        # 找出两部分的共同字符作为中间字符
        common_chars = set(first_part) & set(second_part)
        if common_chars:
            middle = list(common_chars)[0]  # 假设只有一个共同字符
            
            # 找出第一部分中非共同字符作为左边字符
            left = next((c for c in first_part if c != middle), '')
            
            # 找出第二部分中非共同字符作为右边字符
            right = next((c for c in second_part if c != middle), '')
            
            result = left + middle + right
            return result
    
    return expr
 # 用于判断两个子列表是否等价（忽略字符顺序）
def are_equivalent_sublists(list1, list2):
    if len(list1) != len(list2):
        return False
    
    # 特殊处理90度角
    if '90' in list1 and '90' not in list2:
        return False
    if '90' in list2 and '90' not in list1:
        return False
    
    # 对于其他角度表示，检查是否可以通过重排字母匹配
    list1_non_90 = [item for item in list1 if item != None]
    list2_non_90 = [item for item in list2 if item != None]
    
    # 创建两个列表，每个列表包含字符串的排序字符
    sorted_list1 = [''.join(sorted(item)) for item in list1_non_90]
    sorted_list2 = [''.join(sorted(item)) for item in list2_non_90]
    
    # 检查排序后的字符串集合是否相同
    return set(sorted_list1) == set(sorted_list2)

def _draw_reinforce(
    ax: matplotlib.axes.Axes,
    points: list[gm.Point],
    lines: list[gm.Line],
    circles: list[gm.Circle],
    goal: Any,
    equals: list[tuple[Any, Any]],
    highlights: list[tuple[str, list[gm.Point]]],
    angle,
    equ_angle,
    segments,
    para
):
  """Draw everything."""
  # colors = ['red', 'green', 'blue', 'orange', 'magenta', 'purple', 'black', 'grey']
  # pcolor = 'black'
  # lcolor = 'black'
  # ccolor = 'grey'
  colors = []
  if get_theme() == 'dark':
    # pcolor, lcolor, ccolor = 'white', 'white', 'cyan'
    colors = ['red', 'green', 'blue', 'orange', 'magenta', 'purple', 'grey', 'white']
  elif get_theme() == 'light':
    # pcolor, lcolor, ccolor = 'black', 'black', 'blue'
    colors = ['red', 'green', 'blue', 'orange', 'magenta', 'purple', 'black', 'grey', 'cyan']
  elif get_theme() == 'grey':
    colors = ['red', 'green', 'blue', 'orange', 'magenta', 'purple', 'black', 'cyan']
    # pcolor, lcolor, ccolor = 'black', 'black', 'grey'
    # colors = ['red', 'green', 'blue', 'orange', 'magenta', 'purple', 'black']
    # colors = ['grey']

  pcolor = random.choice(colors)
  # # colors.remove(pcolor)
  lcolor = random.choice(colors)
  # # colors.remove(lcolor)
  ccolor = random.choice(colors)
  point_dict = {point.name: point.num for point in points}
  line_boundaries = []
  len_name_len = []
  label_color = random.choice(['red', 'green', 'blue', 'orange', 'magenta', 'purple'])

  segment_indices = None
  if segments:
    # print("segments:",[[n.obj for n in segment.equivs()] for segment in segments])
    all_segment = [[n.name for n in segment.equivs()] for segment in segments]
    # print("all_segment:",all_segment)
    segment_indices = {}
    special_segment = []
    for sublist_idx, sublist in enumerate(all_segment):
      # print("sublist:",sublist)
      if len(sublist) != 1: # 防止倍数关系
        for item in sublist:
            match = re.search(r'\((.*?)\)', item)
            if match:
                segment = match.group(1)
                if segment not in segment_indices:
                    segment_indices[segment] = []
                segment_indices[segment].append(sublist_idx)
      else:
        special_segment.append(re.search(r'\((.*?)\)', sublist[0]).group(1))
  # else:
  #   segment_indices = None
    # print("segment_indices:", segment_indices)
    # print("special_segment:", special_segment)
    # print("point_dict:", point_dict)
      # 计算所有特殊线段的长度
    segment_lengths = {}
    for segment in special_segment:
        segment_lengths[segment] = calculate_segment_length(segment, point_dict)

    # print(segment_lengths)
    color_choices = ['lightblue', 'pink', 'coral', 'indigo', 'salmon', 
        'xkcd:sky blue', 'xkcd:grass green', 'xkcd:midnight blue', 
        '#FF5733', '#00FF00', '#0000FF',  '0.5']
    ratio_relationships = check_ratio_relationships(segment_lengths)
    if ratio_relationships:
      # print("发现以下倍数关系：")
      for seg1, seg2, factor, exact_ratio in ratio_relationships:
          # print(f"{seg1} 是 {seg2} 的大约 {factor} 倍 (精确比值: {exact_ratio:.4f})")
          # print(type(seg1))

          random_color = random.choice(color_choices)
          plot_segment_with_marks(ax, point_dict, seg1, seg2, factor, random_color)
         # 颜色列表，确保有足够的颜色供每个子列表使用

  markcolors = plt.cm.get_cmap('Set1', len(para))  # 使用 Set1 颜色映射

  if para:
    for idx, sublist in enumerate(para):

        
        # 提取两个线段的点
        point1 = sublist[1]  # 第二个字母
        point2 = sublist[2]  # 第三个字母
        point3 = sublist[3]  # 第四个字母
        point4 = sublist[4]  # 第五个字母
        
        # 获取坐标
        num1 = point_dict[point1]
        num2 = point_dict[point2]
        num3 = point_dict[point3]
        num4 = point_dict[point4]
        
        # 计算线段的中心点
        mid1 = ((num1.x + num2.x) / 2, (num1.y + num2.y) / 2)
        mid2 = ((num3.x + num4.x) / 2, (num3.y + num4.y) / 2)
        
        # 为每个 sublist 分配不同的颜色
        markcolor = markcolors(idx)  # 使用颜色映射来获取颜色
        # 画出这两条线段
        # 绘制小三角形标记（表示平行） - 在两条线段的中点上画三角形
        ax.scatter(mid1[0], mid1[1], marker='^', color=markcolor, s=10)  # 第一条线段的中点
        ax.scatter(mid2[0], mid2[1], marker='^', color=markcolor, s=10)  # 第二条线段的中点
        

  for l in lines:
    p1, p2, line_len, line_name = draw_line(ax, l, color=lcolor, label_color=label_color, show_len = True, segment_indices = segment_indices,  )
    if p1 is not None and p2 is not None:
      line_boundaries.append((p1, p2))
    if line_len is not None and line_name is not None:
      len_name_len.append((line_len, line_name))

  circles = [draw_circle(ax, c, color=ccolor) for c in circles]

  ncolor=random.choice(colors)
  for p in points:
    draw_point_reinforce(ax, p.num, p.name, line_boundaries, circles, ncolor=ncolor, pcolor=pcolor)

  if angle:
    angle_point = [
        {
            "p1": point_dict[single_angle[0]],
            "head": point_dict[single_angle[1]],
            "p2": point_dict[single_angle[2]]
        }
        for single_angle in angle
    ]
      # 然后直接调用 annotate_angle 
    for angles in angle_point:
      annotate_angle(ax, angles, color=random.choice(['red', 'green', 'blue', 'orange', 'magenta', 'purple']), show_degree = True)

  if equ_angle:
          # 处理并输出结果
    # print("equ_angle:", equ_angle)
    # 转换所有表达式
    converted_lists = []
    for sublist in equ_angle:
        converted_sublist = [convert_angle_expression(item) for item in sublist]
        if converted_sublist is not None:
            converted_lists.append(converted_sublist)

    # 去除等价的子列表
    unique_lists = []
    for i, sublist in enumerate(converted_lists):
        # 检查这个子列表是否与已添加的某个子列表等价
        if not None in sublist:
          is_duplicate = False
          for unique_sublist in unique_lists:

              if are_equivalent_sublists(sublist, unique_sublist):
                  is_duplicate = True
                  break
          
          if not is_duplicate:
              unique_lists.append(sublist)

    # print("转换后的所有列表:", converted_lists)
    # print("去重后的列表:", unique_lists)
    angle_point = process_angle_lists(unique_lists, points, point_dict)
    # print(angle_point)
    # 假设这是你的颜色列表
    color_list = ['red', 'green', 'blue', 'orange', 'magenta', 'purple']
    # linestyle_list = ['-', '--', '-.', ':']
    arc_radius=0.15
    if angle_point is not None:
      for idx, angle_list in enumerate(angle_point):
      # 选择不同的颜色或样式
        # linestyle = linestyle_list[idx % len(linestyle_list)]  # 确保不会超出样式列表长度
        color = color_list[idx % len(color_list)] 
        arc_radius+=0.07 # 确保不会超出颜色列表的长度
        for angle in angle_list:
          annotate_angle(ax, angle, color=color, arc_radius = arc_radius, show_degree = False, linestyle='-')
    
    if any('m(1pi/2)' in sublist for sublist in equ_angle):
#       vertical_line = [[neighbor.name for neighbor in segment.neighbors(gm.Point)]
#     for segment in lines
#     if len([neighbor.name for neighbor in segment.neighbors(gm.Point)]) == 2
# ]
      vertical_line = [[neighbor.name for neighbor in segment.neighbors(gm.Point)]
    for segment in lines
]     
      vertical_line = split_to_pairs(vertical_line)
      # 用来存储相邻线段的列表
      # print("vertical_line:", vertical_line)
      adjacent_groups = []

      # 遍历每一对线段
      for i in range(len(vertical_line)):
          for j in range(i + 1, len(vertical_line)):
              # 如果两个线段有共同的点，则认为它们相邻
              if set(vertical_line[i]).intersection(set(vertical_line[j])):
                  adjacent_groups.append([vertical_line[i], vertical_line[j]])
      # print("adjacent_groups:", adjacent_groups)
      result = []

      for line_pair in adjacent_groups:
        # 获取两个点的坐标
        p1_name, p2_name = line_pair[0]
        p3_name, p4_name = line_pair[1]
        
        p1 = point_dict[p1_name]
        p2 = point_dict[p2_name]
        p3 = point_dict[p3_name]
        p4 = point_dict[p4_name]

        # 计算向量 p1->p2 和 p3->p4
        vector1 = {'x': p2.x - p1.x, 'y': p2.y - p1.y}
        vector2 = {'x': p4.x - p3.x, 'y': p4.y - p3.y}
        
        # 判断向量是否垂直
        result.append(is_perpendicular(vector1, vector2))

      # print(result)
            # 结果存储
      merged_results = []

      for adj_group, is_vertical in zip(adjacent_groups, result):
          if is_vertical:  # 如果是垂直
              merged_results.append(merge_and_deduplicate(adj_group))

      # print(merged_results)
      angle_point = [
        {
            "p1": point_dict[single_angle[0]],
            "head": point_dict[single_angle[1]],
            "p2": point_dict[single_angle[2]]
        }
        for single_angle in merged_results
    ]
      # print(angle_point)
      color = random.choice(['red', 'green', 'blue', 'orange', 'magenta', 'purple'])
      for angle in angle_point:
        annotate_angle(ax, angle, color=color, arc_radius = arc_radius, show_degree = False,)


  if highlights:
    global HCOLORS
    if HCOLORS is None:
      HCOLORS = [k for k in mcolors.TABLEAU_COLORS.keys() if 'red' not in k]

    for i, (name, args) in enumerate(highlights):
      color_i = HCOLORS[i % len(HCOLORS)]
      highlight(ax, name, args, 'black', color_i, color_i)

  if goal:
    name, args = goal
    lcolor = color1 = color2 = 'red'
    highlight(ax, name, args, lcolor, color1, color2)

  return len_name_len


THEME = 'dark'


def set_theme(theme) -> None:
  global THEME
  THEME = theme


def get_theme() -> str:
  return THEME


def draw(
    points: list[gm.Point],
    lines: list[gm.Line],
    circles: list[gm.Circle],
    segments: list[gm.Segment],
    goal: Any = None,
    highlights: list[tuple[str, list[gm.Point]]] = None,
    equals: list[tuple[Any, Any]] = None,
    block: bool = True,
    save_to: str = None,
    theme: str = 'dark',
    show_len: bool = False
) -> None:
  """Draw everything on the same canvas."""
  plt.close()
  imsize = 256 / 100
  fig, ax = plt.subplots(figsize=(imsize, imsize), dpi=100)

  set_theme(theme)

  if get_theme() == 'dark':
    ax.set_facecolor((0.0, 0.0, 0.0))
  else:
    ax.set_facecolor((1.0, 1.0, 1.0))

  len_dict={}
  len_dict=_draw(ax, points, lines, circles, goal, equals, highlights, show_len=show_len)

  plt.axis('equal')
  fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
  if points:
    xmin = min([p.num.x for p in points])-3
    xmax = max([p.num.x for p in points])+3
    ymin = min([p.num.y for p in points])-3
    ymax = max([p.num.y for p in points])+3
    plt.margins((xmax - xmin) * 0.1, (ymax - ymin) * 0.1)

  plt.show(block=block)
  plt.savefig(fname = save_to, dpi=300)
  if len_dict:
    return len_dict



def draw_reinforce(
    points: list[gm.Point],
    lines: list[gm.Line],
    circles: list[gm.Circle],
    segments: list[gm.Length],
    goal: Any = None,
    highlights: list[tuple[str, list[gm.Point]]] = None,
    equals: list[tuple[Any, Any]] = None,
    block: bool = True,
    save_to: str = None,
    angle: list[tuple[Any]] = None,
    equ_angle: list[tuple[Any]] = None,
    para: list[tuple[Any]] = None,
    theme: str = 'white',

) -> None:
  """Draw everything on the same canvas."""
  plt.close()
  imsize = 256 / 100
  fig, ax = plt.subplots(figsize=(imsize, imsize), dpi=100)

  theme=random.choice(['dark', 'light', 'grey'])
  theme = 'light'
  set_theme(theme)

  if get_theme() == 'dark':
      ax.set_facecolor((0.0, 0.0, 0.0))
  elif get_theme() == 'grey':
      ax.set_facecolor((0.5, 0.5, 0.5))
  else:
      ax.set_facecolor((1.0, 1.0, 1.0))

  len_name_len = _draw_reinforce(ax, points, lines, circles, goal, equals, highlights, angle, equ_angle, segments, para)
  # print(len_name_len)
  plt.axis('equal')
  fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
  # if points:
  #   for p in points:
  #     print(p.name,p.)
  xmin = min([p.num.x for p in points])-3
  xmax = max([p.num.x for p in points])+3
  ymin = min([p.num.y for p in points])-3
  ymax = max([p.num.y for p in points])+3
  plt.margins((xmax - xmin) * 0.1, (ymax - ymin) * 0.1)

  plt.show(block=block)
  
  plt.savefig(fname = save_to, dpi=300)
  
  return len_name_len



def close_enough(a: float, b: float, tol: float = 1e-12) -> bool:
  return abs(a - b) < tol


def assert_close_enough(a: float, b: float, tol: float = 1e-12) -> None:
  assert close_enough(a, b, tol), f'|{a}-{b}| = {abs(a-b)} >= {tol}'


def ang_of(tail: Point, head: Point) -> float:
  vector = head - tail
  arctan = np.arctan2(vector.y, vector.x) % (2 * np.pi)
  return arctan


def ang_between(tail: Point, head1: Point, head2: Point) -> float:
  ang1 = ang_of(tail, head1)
  ang2 = ang_of(tail, head2)
  diff = ang1 - ang2
  # return diff % (2*np.pi)
  if diff > np.pi:
    return diff - 2 * np.pi
  if diff < -np.pi:
    return 2 * np.pi + diff
  return diff


def head_from(tail: Point, ang: float, length: float = 1) -> Point:
  vector = Point(np.cos(ang) * length, np.sin(ang) * length)
  return tail + vector


def random_points(n: int = 3) -> list[Point]:
  return [Point(unif(-1, 1), unif(-1, 1)) for _ in range(n)]


def random_rfss(*points: list[Point]) -> list[Point]:
  """Random rotate-flip-scale-shift a point cloud."""
  # center point cloud.
  average = sum(points, Point(0.0, 0.0)) * (1.0 / len(points))
  points = [p - average for p in points]

  # rotate
  ang = unif(0.0, 2 * np.pi)
  sin, cos = np.sin(ang), np.cos(ang)
  # scale and shift
  scale = unif(0.5, 2.0)
  shift = Point(unif(-1, 1), unif(-1, 1))
  points = [p.rotate(sin, cos) * scale + shift for p in points]

  # randomly flip
  if np.random.rand() < 0.5:
    points = [p.flip() for p in points]

  return points


def reduce(
    objs: list[Union[Point, Line, Circle, HalfLine, HoleCircle]],
    existing_points: list[Point],
) -> list[Point]:
  """Reduce intersecting objects into one point of intersections."""
  if all(isinstance(o, Point) for o in objs):
    return objs

  elif len(objs) == 1:
    return objs[0].sample_within(existing_points)

  elif len(objs) == 2:
    a, b = objs
    result = a.intersect(b)
    if isinstance(result, Point):
      return [result]
    a, b = result
    a_close = any([a.close(x) for x in existing_points])
    if a_close:
      return [b]
    b_close = any([b.close(x) for x in existing_points])
    if b_close:
      return [a]
    return [np.random.choice([a, b])]

  else:
    raise ValueError(f'Cannot reduce {objs}')


def sketch(
    name: str, args: list[Union[Point, gm.Point]]
) -> list[Union[Point, Line, Circle, HalfLine, HoleCircle]]:
  fun = globals()['sketch_' + name]
  args = [p.num if isinstance(p, gm.Point) else p for p in args]
  # print(args)
  out = fun(args)
  # print("-------\n", out, "\n-------")
  # out can be one or multiple {Point/Line/HalfLine}
  if isinstance(out, (tuple, list)):
    return list(out)
  return [out]


def sketch_on_opline(args: tuple[gm.Point, ...]) -> HalfLine:
  a, b = args
  return HalfLine(a, a + a - b)


def sketch_on_hline(args: tuple[gm.Point, ...]) -> HalfLine:
  a, b = args
  return HalfLine(a, b)


def sketch_ieq_triangle(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a = Point(0.0, 0.0)
  b = Point(1.0, 0.0)

  c, _ = Circle(a, p1=b).intersect(Circle(b, p1=a))
  return a, b, c


def sketch_incenter2(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a, b, c = args
  l1 = sketch_bisect([b, a, c])
  l2 = sketch_bisect([a, b, c])
  i = line_line_intersection(l1, l2)
  x = i.foot(Line(b, c))
  y = i.foot(Line(c, a))
  z = i.foot(Line(a, b))
  return x, y, z, i


def sketch_excenter2(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a, b, c = args
  l1 = sketch_bisect([b, a, c])
  l2 = sketch_exbisect([a, b, c])
  i = line_line_intersection(l1, l2)
  x = i.foot(Line(b, c))
  y = i.foot(Line(c, a))
  z = i.foot(Line(a, b))
  return x, y, z, i


def sketch_centroid(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a, b, c = args
  x = (b + c) * 0.5
  y = (c + a) * 0.5
  z = (a + b) * 0.5
  i = line_line_intersection(Line(a, x), Line(b, y))
  return x, y, z, i


def sketch_ninepoints(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a, b, c = args
  x = (b + c) * 0.5
  y = (c + a) * 0.5
  z = (a + b) * 0.5
  c = Circle(p1=x, p2=y, p3=z)
  return x, y, z, c.center


def sketch_2l1c(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  """Sketch a circle touching two lines and another circle."""
  a, b, c, p = args
  bc, ac = Line(b, c), Line(a, c)
  circle = Circle(p, p1=a)

  d, d_ = line_circle_intersection(p.perpendicular_line(bc), circle)
  if bc.diff_side(d_, a):
    d = d_

  e, e_ = line_circle_intersection(p.perpendicular_line(ac), circle)
  if ac.diff_side(e_, b):
    e = e_

  df = d.perpendicular_line(Line(p, d))
  ef = e.perpendicular_line(Line(p, e))
  f = line_line_intersection(df, ef)

  g, g_ = line_circle_intersection(Line(c, f), circle)
  if bc.same_side(g_, a):
    g = g_

  b_ = c + (b - c) / b.distance(c)
  a_ = c + (a - c) / a.distance(c)
  m = (a_ + b_) * 0.5
  x = line_line_intersection(Line(c, m), Line(p, g))
  return x.foot(ac), x.foot(bc), g, x


def sketch_3peq(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a, b, c = args
  ab, bc, ca = Line(a, b), Line(b, c), Line(c, a)

  z = b + (c - b) * np.random.uniform(-0.5, 1.5)

  z_ = z * 2 - c
  l = z_.parallel_line(ca)
  x = line_line_intersection(l, ab)
  y = z * 2 - x
  return x, y, z


def try_to_sketch_intersect(
    name1: str,
    args1: list[Union[gm.Point, Point]],
    name2: str,
    args2: list[Union[gm.Point, Point]],
    existing_points: list[Point],
) -> Optional[Point]:
  """Try to sketch an intersection between two objects."""
  obj1 = sketch(name1, args1)[0]
  obj2 = sketch(name2, args2)[0]

  if isinstance(obj1, Line) and isinstance(obj2, Line):
    fn = line_line_intersection
  elif isinstance(obj1, Circle) and isinstance(obj2, Circle):
    fn = circle_circle_intersection
  else:
    fn = line_circle_intersection
    if isinstance(obj2, Line) and isinstance(obj1, Circle):
      obj1, obj2 = obj2, obj1

  try:
    x = fn(obj1, obj2)
  except:  # pylint: disable=bare-except
    return None

  if isinstance(x, Point):
    return x

  x1, x2 = x

  close1 = check_too_close([x1], existing_points)
  far1 = check_too_far([x1], existing_points)
  if not close1 and not far1:
    return x1
  close2 = check_too_close([x2], existing_points)
  far2 = check_too_far([x2], existing_points)
  if not close2 and not far2:
    return x2

  return None


def sketch_acircle(args: tuple[gm.Point, ...]) -> Circle:
  a, b, c, d, f = args
  de = sketch_aline([c, a, b, f, d])
  fe = sketch_aline([a, c, b, d, f])
  e = line_line_intersection(de, fe)
  return Circle(p1=d, p2=e, p3=f)


def sketch_aline(args: tuple[gm.Point, ...]) -> HalfLine:
  """Sketch the construction aline."""
  A, B, C, D, E = args
  ab = A - B
  cb = C - B
  de = D - E

  dab = A.distance(B)
  ang_ab = np.arctan2(ab.y / dab, ab.x / dab)

  dcb = C.distance(B)
  ang_bc = np.arctan2(cb.y / dcb, cb.x / dcb)

  dde = D.distance(E)
  ang_de = np.arctan2(de.y / dde, de.x / dde)

  ang_ex = ang_de + ang_bc - ang_ab
  X = E + Point(np.cos(ang_ex), np.sin(ang_ex))
  return HalfLine(E, X)


def sketch_amirror(args: tuple[gm.Point, ...]) -> HalfLine:
  """Sketch the angle mirror."""
  A, B, C = args  # pylint: disable=invalid-name
  ab = A - B
  cb = C - B

  dab = A.distance(B)
  ang_ab = np.arctan2(ab.y / dab, ab.x / dab)
  dcb = C.distance(B)
  ang_bc = np.arctan2(cb.y / dcb, cb.x / dcb)

  ang_bx = 2 * ang_bc - ang_ab
  X = B + Point(np.cos(ang_bx), np.sin(ang_bx))  # pylint: disable=invalid-name
  return HalfLine(B, X)


def sketch_bisect(args: tuple[gm.Point, ...]) -> Line:
  a, b, c = args
  ab = a.distance(b)
  bc = b.distance(c)
  x = b + (c - b) * (ab / bc)
  m = (a + x) * 0.5
  return Line(b, m)


def sketch_exbisect(args: tuple[gm.Point, ...]) -> Line:
  a, b, c = args
  return sketch_bisect(args).perpendicular_line(b)


def sketch_bline(args: tuple[gm.Point, ...]) -> Line:
  a, b = args
  m = (a + b) * 0.5
  return m.perpendicular_line(Line(a, b))


def sketch_dia(args: tuple[gm.Point, ...]) -> Circle:
  a, b = args
  return Circle((a + b) * 0.5, p1=a)


def sketch_tangent(args: tuple[gm.Point, ...]) -> tuple[Point, Point]:
  a, o, b = args
  dia = sketch_dia([a, o])
  return circle_circle_intersection(Circle(o, p1=b), dia)


def sketch_circle(args: tuple[gm.Point, ...]) -> Circle:
  a, b, c = args
  return Circle(center=a, radius=b.distance(c))


def sketch_cc_tangent(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  """Sketch tangents to two circles."""
  o, a, w, b = args
  ra, rb = o.distance(a), w.distance(b)

  ow = Line(o, w)
  if close_enough(ra, rb):
    oo = ow.perpendicular_line(o)
    oa = Circle(o, ra)
    x, z = line_circle_intersection(oo, oa)
    y = x + w - o
    t = z + w - o
    return x, y, z, t

  swap = rb > ra
  if swap:
    o, a, w, b = w, b, o, a
    ra, rb = rb, ra

  oa = Circle(o, ra)
  q = o + (w - o) * ra / (ra - rb)

  x, z = circle_circle_intersection(sketch_dia([o, q]), oa)
  y = w.foot(Line(x, q))
  t = w.foot(Line(z, q))

  if swap:
    x, y, z, t = y, x, t, z

  return x, y, z, t


def sketch_hcircle(args: tuple[gm.Point, ...]) -> HoleCircle:
  a, b = args
  return HoleCircle(center=a, radius=a.distance(b), hole=b)


def sketch_e5128(args: tuple[gm.Point, ...]) -> tuple[Point, Point]:
  a, b, c, d = args
  ad = Line(a, d)

  g = (a + b) * 0.5
  de = Line(d, g)

  e, f = line_circle_intersection(de, Circle(c, p1=b))

  if e.distance(d) < f.distance(d):
    e = f
  return e, g


def sketch_eq_quadrangle(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  """Sketch quadrangle with two equal opposite sides."""
  a = Point(0.0, 0.0)
  b = Point(1.0, 0.0)

  length = np.random.uniform(0.5, 2.0)
  ang = np.random.uniform(np.pi / 3, np.pi * 2 / 3)
  d = head_from(a, ang, length)

  ang = ang_of(b, d)
  ang = np.random.uniform(ang / 10, ang / 9)
  c = head_from(b, ang, length)
  a, b, c, d = random_rfss(a, b, c, d)
  return a, b, c, d


def sketch_eq_trapezoid(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a = Point(0.0, 0.0)
  b = Point(1.0, 0.0)
  l = unif(0.5, 2.0)

  height = unif(0.5, 2.0)
  c = Point(0.5 + l / 2.0, height)
  d = Point(0.5 - l / 2.0, height)

  a, b, c, d = random_rfss(a, b, c, d)
  return a, b, c, d


def sketch_eqangle2(args: tuple[gm.Point, ...]) -> Point:
  """Sketch the def eqangle2."""
  a, b, c = args

  d = c * 2 - b

  ba = b.distance(a)
  bc = b.distance(c)
  l = ba * ba / bc

  if unif(0.0, 1.0) < 0.5:
    be = min(l, bc)
    be = unif(be * 0.1, be * 0.9)
  else:
    be = max(l, bc)
    be = unif(be * 1.1, be * 1.5)

  e = b + (c - b) * (be / bc)
  y = b + (a - b) * (be / l)
  return line_line_intersection(Line(c, y), Line(a, e))


def sketch_eqangle3(args: tuple[gm.Point, ...]) -> Circle:
  a, b, d, e, f = args
  de = d.distance(e)
  ef = e.distance(f)
  ab = b.distance(a)
  ang_ax = ang_of(a, b) + ang_between(e, d, f)
  x = head_from(a, ang_ax, length=de / ef * ab)
  return Circle(p1=a, p2=b, p3=x)


def sketch_eqdia_quadrangle(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  """Sketch quadrangle with two equal diagonals."""
  m = unif(0.3, 0.7)
  n = unif(0.3, 0.7)
  a = Point(-m, 0.0)
  c = Point(1 - m, 0.0)
  b = Point(0.0, -n)
  d = Point(0.0, 1 - n)

  ang = unif(-0.25 * np.pi, 0.25 * np.pi)
  sin, cos = np.sin(ang), np.cos(ang)
  b = b.rotate(sin, cos)
  d = d.rotate(sin, cos)
  a, b, c, d = random_rfss(a, b, c, d)
  return a, b, c, d


def sketch_free(args: tuple[gm.Point, ...]) -> Point:
  return random_points(1)[0]


def sketch_isos(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  base = unif(0.5, 1.5)
  height = unif(0.5, 1.5)

  b = Point(-base / 2, 0.0)
  c = Point(base / 2, 0.0)
  a = Point(0.0, height)
  a, b, c = random_rfss(a, b, c)
  return a, b, c


def sketch_line(args: tuple[gm.Point, ...]) -> Line:
  a, b = args
  return Line(a, b)


def sketch_cyclic(args: tuple[gm.Point, ...]) -> Circle:
  a, b, c = args
  return Circle(p1=a, p2=b, p3=c)


def sketch_hline(args: tuple[gm.Point, ...]) -> HalfLine:
  a, b = args
  return HalfLine(a, b)


def sketch_midp(args: tuple[gm.Point, ...]) -> Point:
  a, b = args
  return (a + b) * 0.5


def sketch_pentagon(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  points = [Point(1.0, 0.0)]
  ang = 0.0

  for i in range(4):
    ang += (2 * np.pi - ang) / (5 - i) * unif(0.5, 1.5)
    point = Point(np.cos(ang), np.sin(ang))
    points.append(point)

  a, b, c, d, e = points  # pylint: disable=unbalanced-tuple-unpacking
  a, b, c, d, e = random_rfss(a, b, c, d, e)
  return a, b, c, d, e


def sketch_pline(args: tuple[gm.Point, ...]) -> Line:
  a, b, c = args
  return a.parallel_line(Line(b, c))


def sketch_pmirror(args: tuple[gm.Point, ...]) -> Point:
  a, b = args
  return b * 2 - a


def sketch_quadrangle(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  """Sketch a random quadrangle."""
  m = unif(0.3, 0.7)
  n = unif(0.3, 0.7)

  a = Point(-m, 0.0)
  c = Point(1 - m, 0.0)
  b = Point(0.0, -unif(0.25, 0.75))
  d = Point(0.0, unif(0.25, 0.75))

  ang = unif(-0.25 * np.pi, 0.25 * np.pi)
  sin, cos = np.sin(ang), np.cos(ang)
  b = b.rotate(sin, cos)
  d = d.rotate(sin, cos)
  a, b, c, d = random_rfss(a, b, c, d)
  return a, b, c, d


def sketch_r_trapezoid(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a = Point(0.0, 1.0)
  d = Point(0.0, 0.0)
  b = Point(unif(0.5, 1.5), 1.0)
  c = Point(unif(0.5, 1.5), 0.0)
  a, b, c, d = random_rfss(a, b, c, d)
  return a, b, c, d


def sketch_r_triangle(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a = Point(0.0, 0.0)
  b = Point(0.0, unif(0.5, 2.0))
  c = Point(unif(0.5, 2.0), 0.0)
  a, b, c = random_rfss(a, b, c)
  return a, b, c


def sketch_rectangle(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a = Point(0.0, 0.0)
  b = Point(0.0, 1.0)
  l = unif(0.5, 2.0)
  c = Point(l, 1.0)
  d = Point(l, 0.0)
  a, b, c, d = random_rfss(a, b, c, d)
  return a, b, c, d


def sketch_reflect(args: tuple[gm.Point, ...]) -> Point:
  a, b, c = args
  m = a.foot(Line(b, c))
  return m * 2 - a


def sketch_risos(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a = Point(0.0, 0.0)
  b = Point(0.0, 1.0)
  c = Point(1.0, 0.0)
  a, b, c = random_rfss(a, b, c)
  return a, b, c


def sketch_rotaten90(args: tuple[gm.Point, ...]) -> Point:
  a, b = args
  ang = -np.pi / 2
  return a + (b - a).rotate(np.sin(ang), np.cos(ang))


def sketch_rotatep90(args: tuple[gm.Point, ...]) -> Point:
  a, b = args
  ang = np.pi / 2
  return a + (b - a).rotate(np.sin(ang), np.cos(ang))


def sketch_s_angle(args: tuple[gm.Point, ...]) -> HalfLine:
  a, b, y = args
  ang = y / 180 * np.pi
  x = b + (a - b).rotatea(ang)
  return HalfLine(b, x)


def sketch_segment(args: tuple[gm.Point, ...]) -> tuple[Point, Point]:
  a, b = random_points(2)
  return a, b


def sketch_shift(args: tuple[gm.Point, ...]) -> Point:
  a, b, c = args
  return c + (b - a)


def sketch_square(args: tuple[gm.Point, ...]) -> tuple[Point, Point]:
  a, b = args
  c = b + (a - b).rotatea(-np.pi / 2)
  d = a + (b - a).rotatea(np.pi / 2)
  return c, d


def sketch_isquare(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a = Point(0.0, 0.0)
  b = Point(1.0, 0.0)
  c = Point(1.0, 1.0)
  d = Point(0.0, 1.0)
  a, b, c, d = random_rfss(a, b, c, d)
  return a, b, c, d


def sketch_tline(args: tuple[gm.Point, ...]) -> Line:
  a, b, c = args
  return a.perpendicular_line(Line(b, c))


def sketch_trapezoid(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  d = Point(0.0, 0.0)
  c = Point(1.0, 0.0)

  base = unif(0.5, 2.0)
  height = unif(0.5, 2.0)
  a = Point(unif(0.2, 0.5), height)
  b = Point(a.x + base, height)
  a, b, c, d = random_rfss(a, b, c, d)
  return a, b, c, d


def sketch_triangle(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  a = Point(0.0, 0.0)
  b = Point(1.0, 0.0)
  ac = unif(0.5, 2.0)
  ang = unif(0.2, 0.8) * np.pi
  c = head_from(a, ang, ac)
  return a, b, c


def sketch_triangle12(args: tuple[gm.Point, ...]) -> tuple[Point, ...]:
  b = Point(0.0, 0.0)
  c = Point(unif(1.5, 2.5), 0.0)
  a, _ = circle_circle_intersection(Circle(b, 1.0), Circle(c, 2.0))
  a, b, c = random_rfss(a, b, c)
  return a, b, c


def sketch_trisect(args: tuple[gm.Point, ...]) -> tuple[Point, Point]:
  """Sketch two trisectors of an angle."""
  a, b, c = args
  ang1 = ang_of(b, a)
  ang2 = ang_of(b, c)

  swap = 0
  if ang1 > ang2:
    ang1, ang2 = ang2, ang1
    swap += 1

  if ang2 - ang1 > np.pi:
    ang1, ang2 = ang2, ang1 + 2 * np.pi
    swap += 1

  angx = ang1 + (ang2 - ang1) / 3
  angy = ang2 - (ang2 - ang1) / 3

  x = b + Point(np.cos(angx), np.sin(angx))
  y = b + Point(np.cos(angy), np.sin(angy))

  ac = Line(a, c)
  x = line_line_intersection(Line(b, x), ac)
  y = line_line_intersection(Line(b, y), ac)

  if swap == 1:
    return y, x
  return x, y


def sketch_trisegment(args: tuple[gm.Point, ...]) -> tuple[Point, Point]:
  a, b = args
  x, y = a + (b - a) * (1.0 / 3), a + (b - a) * (2.0 / 3)
  return x, y
#!/usr/bin/env python3
"""Gradient-descent evacuation simulator for small multi-floor buildings.

The simulator uses local attraction toward exits or ramp entrances, together with
repulsion from obstacles and other agents. Ramps get special handling once an
agent enters them, but ordinary floor navigation is purely local.

Supported floorplan schema
--------------------------------------
{
  "name": "Three story demo",
  "grid": {"cell_size": 0.35, "padding": 1.0},
  "floors": [
    {
      "name": "Ground",
      "z": 0.0,
      "thickness": 0.18,
      "polygon": [[0,0], [18,0], [18,12], [0,12]],
      "obstacles": [
        {"type": "box", "polygon": [[6,4], [7.4,4], [7.4,5.8], [6,5.8]], "height": 1.2},
        {"type": "column", "center": [12,8], "radius": 0.45, "height": 3.0}
      ]
    }
  ],
  "fixtures": [
    {"type": "wall", "polygon": [[0,0], [18,0], [18,0.25], [0,0.25]], "z": [0, 9]}
  ],
  "ramps": [
    {
      "name": "upper_to_mid",
      "polygon": [[14,7], [18,7], [18,9], [14,9]],
      "start": [18,8,6.09],
      "end": [14,8,3.09],
      "width": 2.0,
      "thickness": 0.16,
      "from_floor": 2,
      "to_floor": 1
    }
  ],
  "openings": [{"min": [13.8, 6.9, 5.8], "max": [18.2, 9.1, 6.5]}],
  "goal": {"min": [-0.5, 5.2, -0.1], "max": [0.8, 6.8, 1.0]}
}

Scenario schema (new style)
---------------------------
{
  "name": "Mixed agents",
  "simulation": {"dt": 0.05, "steps": 700, "seed": 7},
  "agent_defaults": {
    "radius": 0.22,
    "max_speed": 1.35,
    "nav_gain": 3.0,
    "wall_strength": 5.5,
    "wall_range": 1.0,
    "social_strength": 1.8,
    "social_range": 1.1,
    "ramp_gain": 3.6,
    "goal_gain": 2.2,
    "damping": 0.72
  },
  "agents": [
    {
      "count": 20,
      "floor": 2,
      "offset_range": {"x": [-0.75, 0.75], "y": [-0.65, 0.65]},
      "color": "royalblue",
      "repulsion": {
        "strength": 1.8,
        "range": 1.0,
        "anisotropy": {
          "enabled": true,
          "forward_strength": 1.7,
          "backward_strength": 0.55,
          "forward_angle_deg": 80
        }
      }
    }
  ],
  "obstacles": [
    {"type": "box", "floor": 1, "offset": [0.15, -0.2], "size": [1.1, 1.1], "height": 1.2}
  ]
}

Legacy fields from old_scratch.py are also accepted where practical.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Vector = np.ndarray
Polygon2D = np.ndarray

DATA_FOLDER = pathlib.Path(__file__).resolve().parent / "scenarios"
EPS = 1e-9
INF = 1e18


@dataclass
class Opening:
    min_corner: Vector
    max_corner: Vector


@dataclass
class Goal:
    name: str
    min_corner: Vector
    max_corner: Vector
    color: str = "green4"

    @property
    def center(self) -> Vector:
        return 0.5 * (self.min_corner + self.max_corner)


@dataclass
class Floor:
    index: int
    name: str
    z: float
    thickness: float
    polygon: Polygon2D
    color: str = "lightgray"
    obstacles: List["Fixture"] = field(default_factory=list)

    @property
    def surface_z(self) -> float:
        return self.z + 0.5 * self.thickness

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return polygon_bbox(self.polygon)


@dataclass
class Fixture:
    kind: str
    polygon: Optional[Polygon2D] = None
    center: Optional[Vector] = None
    radius: Optional[float] = None
    base_z: float = 0.0
    top_z: float = 0.0
    color: str = "slategray"
    alpha: float = 0.25

    def active_at_z(self, z: float) -> bool:
        return self.base_z - EPS <= z <= self.top_z + EPS


@dataclass
class Ramp:
    name: str
    polygon: Polygon2D
    start: Vector
    end: Vector
    width: float
    from_floor: int
    to_floor: int
    thickness: float = 0.16
    color: str = "orange5"

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def centerline_xy(self) -> Tuple[Vector, Vector]:
        return self.start[:2].copy(), self.end[:2].copy()

    @property
    def lower_floor(self) -> int:
        return self.from_floor if self.start[2] < self.end[2] else self.to_floor

    @property
    def upper_floor(self) -> int:
        return self.from_floor if self.start[2] > self.end[2] else self.to_floor


@dataclass
class RepulsionConfig:
    strength: float = 1.6
    range: float = 1.0
    anisotropy_enabled: bool = False
    forward_strength: float = 1.6
    backward_strength: float = 0.6
    forward_angle_deg: float = 80.0


@dataclass
class AgentConfig:
    radius: float = 0.22
    max_speed: float = 1.35
    nav_gain: float = 3.0
    wall_strength: float = 5.5
    wall_range: float = 1.0
    social_strength: float = 1.8
    social_range: float = 1.1
    ramp_gain: float = 3.6
    goal_gain: float = 2.2
    damping: float = 0.72


@dataclass
class SimulationConfig:
    dt: float = 0.05
    steps: int = 700
    seed: int = 7
    cell_size: float = 0.35
    offscreen: bool = False


@dataclass
class ScenarioObstacleSpec:
    raw: Dict


@dataclass
class AgentGroup:
    count: int
    floor: int
    color: str
    repulsion: RepulsionConfig
    placement: Dict


@dataclass
class HarasserGroup:
    count: int
    floor: int
    offset: List[float]
    color: str
    strength: float
    radius: float
    strategy: str = "intercept"


@dataclass
class Scenario:
    name: str
    simulation: SimulationConfig
    agent_defaults: AgentConfig
    groups: List[AgentGroup]
    obstacles: List[ScenarioObstacleSpec]
    harassers: List[HarasserGroup] = field(default_factory=list)


@dataclass
class Floorplan:
    name: str
    floors: List[Floor]
    fixtures: List[Fixture]
    ramps: List[Ramp]
    openings: List[Opening]
    goals: List[Goal]
    bounds_min: Vector
    bounds_max: Vector

    @property
    def primary_goal(self) -> Goal:
        return self.goals[0] if self.goals else Goal("default", vec([0, 0, 0]), vec([1, 1, 1]))

    def goal_center(self, goal_name: Optional[str] = None) -> Vector:
        if goal_name:
            for g in self.goals:
                if g.name == goal_name:
                    return g.center.copy()
        return self.primary_goal.center.copy()

    def goal_bounds(self, goal_name: Optional[str] = None) -> Tuple[Vector, Vector]:
        if goal_name:
            for g in self.goals:
                if g.name == goal_name:
                    return g.min_corner.copy(), g.max_corner.copy()
        return self.primary_goal.min_corner.copy(), self.primary_goal.max_corner.copy()


@dataclass
class Body:
    pos: Vector
    radius: float
    owner_id: int

    @property
    def x(self) -> float:
        return float(self.pos[0])

    @property
    def y(self) -> float:
        return float(self.pos[1])

    @property
    def z(self) -> float:
        return float(self.pos[2])


class Actor(ABC):
    _next_id: int = 0

    def __init__(self):
        self.id = Actor._next_id
        Actor._next_id += 1
        self._bodies: List[Body] = []
        self.active: bool = True
        self._reached_goal: bool = False

    @property
    def bodies(self) -> List[Body]:
        return self._bodies

    @property
    def reached_goal(self) -> bool:
        return self._reached_goal

    @reached_goal.setter
    def reached_goal(self, value: bool) -> None:
        self._reached_goal = value

    @abstractmethod
    def step(
        self,
        env: "Environment",
        cfg: AgentConfig,
        dt: float,
    ) -> None:
        pass

    @abstractmethod
    def force(
        self,
        env: "Environment",
        cfg: AgentConfig,
    ) -> Vector:
        pass

    def reset(self) -> None:
        self.active = True
        self._reached_goal = False
        for body in self._bodies:
            body.pos[2] = body.pos[2]  # placeholder for subclass reset


class Environment:
    def __init__(
        self,
        floorplan: Floorplan,
        field: NavigationField,
    ):
        self.floorplan = floorplan
        self.field = field
        self._actors: List[Actor] = []
        self._dynamic_obstacles: List[Body] = []

    def add_actor(self, actor: Actor) -> None:
        self._actors.append(actor)

    def add_dynamic_obstacle(self, body: Body) -> None:
        self._dynamic_obstacles.append(body)

    @property
    def actors(self) -> List[Actor]:
        return self._actors

    @property
    def evacuees(self) -> List["Evacuee"]:
        return [a for a in self._actors if isinstance(a, Evacuee)]

    @property
    def harassers(self) -> List["Harasser"]:
        return [a for a in self._actors if isinstance(a, Harasser)]

    @property
    def dynamic_obstacles(self) -> List[Body]:
        return self._dynamic_obstacles

    def all_bodies(self) -> List[Body]:
        bodies: List[Body] = []
        for actor in self._actors:
            bodies.extend(actor.bodies)
        bodies.extend(self._dynamic_obstacles)
        return bodies

    def bodies_of_type(self, actor_type: type) -> List[Body]:
        return [b for a in self._actors if isinstance(a, actor_type) for b in a.bodies]


class Evacuee(Actor):
    def __init__(
        self,
        pos: Vector,
        vel: Vector,
        floor_index: int,
        radius: float,
        color: str,
        repulsion: RepulsionConfig,
    ):
        super().__init__()
        self.pos = pos
        self.vel = vel
        self.floor_index = floor_index
        self.radius = radius
        self.color = color
        self.repulsion = repulsion
        self.ramp_name: Optional[str] = None
        self._bodies = [Body(pos=self.pos, radius=radius, owner_id=self.id)]

    @property
    def x(self) -> float:
        return float(self.pos[0])

    @property
    def y(self) -> float:
        return float(self.pos[1])

    @property
    def z(self) -> float:
        return float(self.pos[2])

    def force(self, env: Environment, cfg: AgentConfig) -> Vector:
        return compute_evacuee_force(
            self,
            env.actors,
            env.floorplan,
            env.field,
            env.dynamic_obstacles,
            cfg,
        )

    def step(self, env: Environment, cfg: AgentConfig, dt: float) -> None:
        if not self.active:
            return

        force_vec = self.force(env, cfg)
        self.vel = (1.0 - cfg.damping) * self.vel + cfg.damping * force_vec
        speed = norm(self.vel)
        if speed > cfg.max_speed:
            self.vel *= cfg.max_speed / speed

        self.pos[:2] += self.vel[:2] * dt
        self.pos[0] = clamp(self.pos[0], env.floorplan.bounds_min[0], env.floorplan.bounds_max[0])
        self.pos[1] = clamp(self.pos[1], env.floorplan.bounds_min[1], env.floorplan.bounds_max[1])

        _project_evacuee_to_walkable(self, env.floorplan, env.field)
        _update_evacuee_surface(self, env.floorplan, env.field)

        for goal in env.floorplan.goals:
            if distance_point_to_aabb(self.pos, goal.min_corner, goal.max_corner) <= self.radius:
                self.reached_goal = True
                self.active = False
                self.vel[:] = 0.0
                goal_center = goal.center.copy()
                goal_center[2] = max(goal_center[2], goal.min_corner[2] + self.radius)
                self.pos[:] = goal_center
                break

    def reset(self) -> None:
        super().reset()
        self.vel[:] = 0.0
        self.ramp_name = None


class Harasser(Actor):
    def __init__(
        self,
        pos: Vector,
        vel: Vector,
        floor_index: int,
        radius: float,
        color: str,
        strength: float = 5.0,
        strategy: str = "intercept",
    ):
        super().__init__()
        self.pos = pos.copy()
        self.vel = vel
        self.floor_index = floor_index
        self.radius = radius
        self.color = color
        self.strength = strength
        self.strategy = strategy
        self._bodies = [Body(pos=self.pos, radius=radius, owner_id=self.id)]
        self._target_pos = self.pos[:2].copy()

    @property
    def x(self) -> float:
        return float(self.pos[0])

    @property
    def y(self) -> float:
        return float(self.pos[1])

    @property
    def z(self) -> float:
        return float(self.pos[2])

    def _compute_target_nearest(self, evacuees: List["Evacuee"]) -> np.ndarray:
        nearest = min(evacuees, key=lambda e: norm(self.pos[:2] - e.pos[:2]))
        return nearest.pos[:2].copy()

    def _compute_target_density(self, evacuees: List["Evacuee"], env: Environment) -> np.ndarray:
        return _find_highest_density_point(evacuees, env.floorplan, self.pos)

    def _compute_target_flow(self, evacuees: List["Evacuee"], env: Environment) -> np.ndarray:
        return _compute_flow_target(evacuees, self.pos, env.floorplan, prediction_time=2.0)

    def _compute_target_intercept(self, evacuees: List["Evacuee"], env: Environment) -> np.ndarray:
        clusters = _find_evacuee_clusters(evacuees)
        if not clusters:
            return self._compute_target_nearest(evacuees)

        best_target = None
        best_score = -1.0

        for centroid, velocity, size in clusters:
            predicted_pos = centroid + velocity * 2.0
            min_dist = min(norm(predicted_pos - g.center[:2]) for g in env.floorplan.goals)
            urgency = 1.0 / max(min_dist, 0.5)
            score = float(size) * urgency

            if score > best_score:
                best_score = score
                best_target = predicted_pos

        return best_target if best_target is not None else self.pos[:2].copy()

    def _compute_target(self, evacuees: List["Evacuee"], env: Environment) -> np.ndarray:
        if not evacuees:
            return self.pos[:2].copy()

        if self.strategy == "nearest":
            return self._compute_target_nearest(evacuees)
        elif self.strategy == "density":
            return self._compute_target_density(evacuees, env)
        elif self.strategy == "flow":
            return self._compute_target_flow(evacuees, env)
        elif self.strategy == "intercept":
            return self._compute_target_intercept(evacuees, env)
        else:
            return self._compute_target_nearest(evacuees)

    def force(self, env: Environment, cfg: AgentConfig) -> Vector:
        evacuees = [a for a in env.actors if isinstance(a, Evacuee) and a.active and not a.reached_goal and a.floor_index == self.floor_index]
        if not evacuees:
            return np.zeros(3, dtype=float)

        self._target_pos = self._compute_target(evacuees, env)

        delta = self._target_pos - self.pos[:2]
        dist = norm(delta)
        if dist < EPS:
            return np.zeros(3, dtype=float)

        attraction = unit(delta) * self.strength

        ramp_repulsion = _compute_ramp_underside_repulsion(self.pos, env.floorplan)

        total = np.zeros(3, dtype=float)
        total[:2] = attraction + ramp_repulsion[:2]
        return total

    def step(self, env: Environment, cfg: AgentConfig, dt: float) -> None:
        if not self.active:
            return

        force_vec = self.force(env, cfg)
        self.vel = (1.0 - cfg.damping) * self.vel + cfg.damping * force_vec
        speed = norm(self.vel)
        if speed > cfg.max_speed:
            self.vel *= cfg.max_speed / speed

        self.pos[:2] += self.vel[:2] * dt
        self.pos[0] = clamp(self.pos[0], env.floorplan.bounds_min[0], env.floorplan.bounds_max[0])
        self.pos[1] = clamp(self.pos[1], env.floorplan.bounds_min[1], env.floorplan.bounds_max[1])

        if not is_xy_walkable_on_floor(env.floorplan, self.floor_index, self.pos[:2]):
            self.pos[:2] = nearest_walkable_xy(env.field, self.floor_index, self.pos[:2])
        current_floor = env.floorplan.floors[self.floor_index]
        self.pos[2] = current_floor.surface_z + self.radius

        self._bodies[0].pos[:] = self.pos

    def reset(self) -> None:
        super().reset()
        self.vel[:] = 0.0


@dataclass
class NavigationField:
    cell_size: float
    x_coords: np.ndarray
    y_coords: np.ndarray
    walkable: np.ndarray
    distance_to_block: np.ndarray
    goal_floor: int
    floor_lookup: Dict[int, Floor]


class Simulator:
    def __init__(
        self,
        floorplan: Floorplan,
        scenario: Scenario,
        env: Environment,
        field: Optional[NavigationField] = None,
    ):
        self.floorplan = floorplan
        self.scenario = scenario
        self._env = env
        self._field = field
        self.step_count = 0
        self._interrupted = False

    @property
    def env(self) -> Environment:
        return self._env

    @env.setter
    def env(self, value: Environment) -> None:
        self._env = value

    @property
    def field(self) -> Optional[NavigationField]:
        return self._field

    @field.setter
    def field(self, value: NavigationField) -> None:
        self._field = value

    @property
    def dt(self) -> float:
        return self.scenario.simulation.dt

    @property
    def total_steps(self) -> int:
        return self.scenario.simulation.steps

    @property
    def active_count(self) -> int:
        return sum(1 for a in self._env.actors if a.active)

    @property
    def reached_count(self) -> int:
        return sum(1 for a in self._env.actors if a.reached_goal)

    @property
    def is_finished(self) -> bool:
        return self.step_count >= self.total_steps

    def step(self) -> None:
        if self._interrupted or self.is_finished:
            return

        for actor in self._env.actors:
            actor.step(self._env, self.scenario.agent_defaults, self.dt)
        self.step_count += 1

    def run(self) -> None:
        while not self._interrupted and not self.is_finished:
            self.step()

    def interrupt(self) -> None:
        self._interrupted = True

    def reset(self) -> None:
        self.step_count = 0
        self._interrupted = False
        for actor in self._env.actors:
            actor.reset()


def vec(values: Sequence[float]) -> Vector:
    return np.array(values, dtype=float)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def norm(v: Vector) -> float:
    return float(np.linalg.norm(v))


def unit(v: Vector) -> Vector:
    n = norm(v)
    if n < EPS:
        return np.zeros_like(v, dtype=float)
    return v / n


def normalize_percent(value: float) -> float:
    if abs(value) > 1.0 and abs(value) <= 100.0:
        return value / 100.0
    return value


def polygon_bbox(polygon: Polygon2D) -> Tuple[float, float, float, float]:
    return (
        float(np.min(polygon[:, 0])),
        float(np.max(polygon[:, 0])),
        float(np.min(polygon[:, 1])),
        float(np.max(polygon[:, 1])),
    )


def polygon_area(polygon: Polygon2D) -> float:
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ensure_polygon(raw: Sequence[Sequence[float]]) -> Polygon2D:
    polygon = np.array(raw, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("Polygons must have shape (n, 2) with n >= 3.")
    if polygon_area(polygon) < 0:
        polygon = polygon[::-1].copy()
    return polygon


def point_in_polygon(point_xy: Sequence[float], polygon: Polygon2D) -> bool:
    x, y = float(point_xy[0]), float(point_xy[1])
    inside = False
    x0, y0 = polygon[-1]
    for x1, y1 in polygon:
        if ((y1 > y) != (y0 > y)) and (
            x < (x0 - x1) * (y - y1) / ((y0 - y1) + EPS) + x1
        ):
            inside = not inside
        x0, y0 = x1, y1
    return inside


def points_in_polygon(xs: np.ndarray, ys: np.ndarray, polygon: Polygon2D) -> np.ndarray:
    inside = np.zeros(xs.shape, dtype=bool)
    x0, y0 = polygon[-1]
    for x1, y1 in polygon:
        intersects = ((y1 > ys) != (y0 > ys)) & (
            xs < (x0 - x1) * (ys - y1) / ((y0 - y1) + EPS) + x1
        )
        inside ^= intersects
        x0, y0 = x1, y1
    return inside


def closest_point_on_segment_2d(point_xy: Vector, a_xy: Vector, b_xy: Vector) -> Tuple[Vector, float]:
    ab = b_xy - a_xy
    denom = float(np.dot(ab, ab))
    if denom < EPS:
        return a_xy.copy(), 0.0
    t = clamp(float(np.dot(point_xy - a_xy, ab)) / denom, 0.0, 1.0)
    return a_xy + t * ab, t


def distance_point_to_aabb(point: Vector, min_corner: Vector, max_corner: Vector) -> float:
    clamped = np.array(
        [
            clamp(point[0], min_corner[0], max_corner[0]),
            clamp(point[1], min_corner[1], max_corner[1]),
            clamp(point[2], min_corner[2], max_corner[2]),
        ],
        dtype=float,
    )
    return norm(point - clamped)


def aabb_contains(point: Vector, min_corner: Vector, max_corner: Vector) -> bool:
    return bool(np.all(point >= min_corner - EPS) and np.all(point <= max_corner + EPS))


def triangulate_polygon(polygon: Polygon2D) -> List[Tuple[int, int, int]]:
    polygon = ensure_polygon(polygon)
    remaining = list(range(len(polygon)))
    triangles: List[Tuple[int, int, int]] = []

    def is_convex(i0: int, i1: int, i2: int) -> bool:
        a = polygon[i1] - polygon[i0]
        b = polygon[i2] - polygon[i1]
        return float(a[0] * b[1] - a[1] * b[0]) > EPS

    def point_in_triangle(p: Vector, a: Vector, b: Vector, c: Vector) -> bool:
        v0 = c - a
        v1 = b - a
        v2 = p - a
        dot00 = float(np.dot(v0, v0))
        dot01 = float(np.dot(v0, v1))
        dot02 = float(np.dot(v0, v2))
        dot11 = float(np.dot(v1, v1))
        dot12 = float(np.dot(v1, v2))
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < EPS:
            return False
        inv = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv
        v = (dot00 * dot12 - dot01 * dot02) * inv
        return u >= -EPS and v >= -EPS and (u + v) <= 1.0 + EPS

    guard = 0
    while len(remaining) > 3 and guard < 10000:
        guard += 1
        ear_found = False
        for idx in range(len(remaining)):
            i0 = remaining[(idx - 1) % len(remaining)]
            i1 = remaining[idx]
            i2 = remaining[(idx + 1) % len(remaining)]
            if not is_convex(i0, i1, i2):
                continue
            a, b, c = polygon[i0], polygon[i1], polygon[i2]
            if any(
                point_in_triangle(polygon[j], a, b, c)
                for j in remaining
                if j not in (i0, i1, i2)
            ):
                continue
            triangles.append((i0, i1, i2))
            del remaining[idx]
            ear_found = True
            break
        if not ear_found:
            break
    if len(remaining) == 3:
        triangles.append((remaining[0], remaining[1], remaining[2]))
    return triangles


def build_prism_mesh(polygon: Polygon2D, z0: float, z1: float) -> Tuple[List[List[float]], List[List[int]]]:
    polygon = ensure_polygon(polygon)
    triangles = triangulate_polygon(polygon)
    n = len(polygon)
    vertices: List[List[float]] = []
    for x, y in polygon:
        vertices.append([float(x), float(y), float(z0)])
    for x, y in polygon:
        vertices.append([float(x), float(y), float(z1)])

    faces: List[List[int]] = []
    for a, b, c in triangles:
        faces.append([a, c, b])
        faces.append([a + n, b + n, c + n])
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    return vertices, faces


def build_surface_mesh(polygon: Polygon2D, z: float) -> Tuple[List[List[float]], List[List[int]]]:
    polygon = ensure_polygon(polygon)
    triangles = triangulate_polygon(polygon)
    vertices = [[float(x), float(y), float(z)] for x, y in polygon]
    faces = [[a, b, c] for a, b, c in triangles]
    return vertices, faces


def build_ramp_mesh(ramp: Ramp) -> Tuple[List[List[float]], List[List[int]]]:
    polygon = ensure_polygon(ramp.polygon)
    triangles = triangulate_polygon(polygon)
    top_vertices: List[List[float]] = []
    bottom_vertices: List[List[float]] = []
    for x, y in polygon:
        z = ramp_height_at_xy(ramp, np.array([x, y], dtype=float))
        top_vertices.append([float(x), float(y), float(z + 0.5 * ramp.thickness)])
        bottom_vertices.append([float(x), float(y), float(z - 0.5 * ramp.thickness)])
    vertices = bottom_vertices + top_vertices
    n = len(polygon)
    faces: List[List[int]] = []
    for a, b, c in triangles:
        faces.append([a, c, b])
        faces.append([a + n, b + n, c + n])
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    return vertices, faces


def make_rect_polygon(min_corner: Sequence[float], max_corner: Sequence[float]) -> Polygon2D:
    x0, y0 = float(min_corner[0]), float(min_corner[1])
    x1, y1 = float(max_corner[0]), float(max_corner[1])
    return ensure_polygon([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def rectangle_around_segment(start_xy: Vector, end_xy: Vector, width: float) -> Polygon2D:
    d = end_xy - start_xy
    d_unit = unit(d)
    if norm(d_unit) < EPS:
        d_unit = np.array([1.0, 0.0], dtype=float)
    normal = np.array([-d_unit[1], d_unit[0]], dtype=float)
    half_w = 0.5 * width
    return ensure_polygon(
        [
            start_xy - half_w * normal,
            end_xy - half_w * normal,
            end_xy + half_w * normal,
            start_xy + half_w * normal,
        ]
    )


def load_json_from_folder(folder: pathlib.Path, value: str) -> Dict:
    candidate = pathlib.Path(value)
    if candidate.suffix == ".json" and candidate.exists():
        path = candidate
    elif candidate.exists():
        path = candidate
    else:
        name = value[:-5] if value.endswith(".json") else value
        path = folder / f"{name}.json"
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def nearest_floor_index(z_value: float, floors: Sequence[Floor]) -> int:
    return min(range(len(floors)), key=lambda idx: abs(floors[idx].surface_z - z_value))


def parse_goal(data: Dict) -> Goal:
    name = data.get("name", "goal")
    if "min" in data and "max" in data:
        return Goal(name, vec(data["min"]), vec(data["max"]), color=data.get("color", "green4"))
    center = vec(data.get("center", [0.0, 0.0, 0.0]))
    radius = float(data.get("radius", 0.6))
    return Goal(name, center - radius, center + radius, color=data.get("color", "green4"))


def parse_opening(data: Dict) -> Opening:
    return Opening(vec(data["min"]), vec(data["max"]))


def parse_fixture(data: Dict, default_base: float, default_top: float) -> Fixture:
    kind = data.get("type", data.get("kind", "box")).lower()
    color = data.get("color", "slategray")
    alpha = float(data.get("alpha", 0.28))
    if "min" in data and "max" in data:
        min_corner = vec(data["min"])
        max_corner = vec(data["max"])
        return Fixture(
            kind=kind,
            polygon=make_rect_polygon(min_corner[:2], max_corner[:2]),
            base_z=float(min_corner[2]),
            top_z=float(max_corner[2]),
            color=color,
            alpha=alpha,
        )

    if kind in {"column", "pillar"}:
        center_xy = np.array(data.get("center", [0.0, 0.0])[:2], dtype=float)
        base_z = float(data.get("base_z", data.get("z", default_base)))
        top_z = float(data.get("top_z", base_z + data.get("height", default_top - default_base)))
        return Fixture(
            kind="column",
            center=center_xy,
            radius=float(data.get("radius", 0.5)),
            base_z=base_z,
            top_z=top_z,
            color=color,
            alpha=alpha,
        )

    if "polygon" in data:
        polygon = ensure_polygon(data["polygon"])
    elif "z" in data:
        z_range = data["z"]
        bounds = data.get("bounds")
        if bounds:
            polygon = ensure_polygon([
                [bounds["x"][0], bounds["y"][0]],
                [bounds["x"][1], bounds["y"][0]],
                [bounds["x"][1], bounds["y"][1]],
                [bounds["x"][0], bounds["y"][1]],
            ])
        else:
            polygon = None
    else:
        polygon = None

    z_range = data.get("z")
    if z_range is not None:
        base_z = float(z_range[0])
        top_z = float(z_range[1])
    else:
        base_z = float(data.get("base_z", data.get("z0", default_base)))
        top_z = float(data.get("top_z", data.get("z1", base_z + data.get("height", default_top - default_base))))
    return Fixture(
        kind=kind,
        polygon=polygon,
        base_z=base_z,
        top_z=top_z,
        color=color,
        alpha=alpha,
    )


def parse_ramp(data: Dict, floors: Sequence[Floor]) -> Ramp:
    if "start" in data and "end" in data:
        start = vec(data["start"])
        end = vec(data["end"])
    else:
        start_xy = vec(data["from_xy"])
        end_xy = vec(data["to_xy"])
        from_floor = int(data["from_floor"])
        to_floor = int(data["to_floor"])
        start = np.array([start_xy[0], start_xy[1], floors[from_floor].surface_z], dtype=float)
        end = np.array([end_xy[0], end_xy[1], floors[to_floor].surface_z], dtype=float)

    width = float(data.get("width", 2.0))
    polygon = ensure_polygon(data["polygon"]) if "polygon" in data else rectangle_around_segment(start[:2], end[:2], width)
    from_floor = int(data.get("from_floor", nearest_floor_index(start[2], floors)))
    to_floor = int(data.get("to_floor", nearest_floor_index(end[2], floors)))
    return Ramp(
        name=data.get("name", f"ramp_{from_floor}_{to_floor}"),
        polygon=polygon,
        start=start,
        end=end,
        width=width,
        from_floor=from_floor,
        to_floor=to_floor,
        thickness=float(data.get("thickness", 0.16)),
        color=data.get("color", "orange5"),
    )


def load_floorplan(spec: str) -> Floorplan:
    data = load_json_from_folder(DATA_FOLDER, spec)
    name = data.get("name", pathlib.Path(spec).stem)

    floors: List[Floor] = []
    raw_floors = data.get("floors", [])
    if not raw_floors:
        raise ValueError("Floorplan requires a non-empty 'floors' array.")
    for idx, raw_floor in enumerate(raw_floors):
        polygon = raw_floor.get("polygon")
        if polygon is None:
            bounds = data.get("bounds")
            if bounds is None:
                raise ValueError("Each floor needs a polygon unless global bounds are supplied.")
            polygon = [
                [bounds["x"][0], bounds["y"][0]],
                [bounds["x"][1], bounds["y"][0]],
                [bounds["x"][1], bounds["y"][1]],
                [bounds["x"][0], bounds["y"][1]],
            ]
        floor = Floor(
            index=idx,
            name=raw_floor.get("name", f"floor_{idx}"),
            z=float(raw_floor.get("z", idx * 3.0)),
            thickness=float(raw_floor.get("thickness", 0.18)),
            polygon=ensure_polygon(polygon),
            color=raw_floor.get("color", "lightgray"),
        )
        floors.append(floor)

    z_min = min(f.z for f in floors)
    z_max = max(f.z + f.thickness for f in floors)
    fixtures: List[Fixture] = []
    for raw in data.get("fixtures", []):
        fixtures.append(parse_fixture(raw, z_min, z_max + 3.0))
    for raw in data.get("walls", []):
        fixtures.append(parse_fixture({**raw, "type": raw.get("type", "wall")}, z_min, z_max + 3.0))
    for raw in data.get("obstacles", []):
        fixtures.append(parse_fixture(raw, z_min, z_max + 3.0))
    for floor, raw_floor in zip(floors, raw_floors):
        for raw in raw_floor.get("obstacles", []):
            floor.obstacles.append(
                parse_fixture(
                    raw,
                    floor.z,
                    floor.z + raw.get("height", floor.thickness + 1.5),
                )
            )

    openings = [parse_opening(raw) for raw in data.get("openings", data.get("repulsion_masks", []))]
    ramps = [parse_ramp(raw, floors) for raw in data.get("ramps", data.get("stairs", []))]

    raw_goals = data.get("goals")
    if raw_goals is not None:
        goals = [parse_goal(g) for g in raw_goals]
    else:
        legacy_goal = data.get("goal", {"center": [0.0, 0.0, floors[0].surface_z], "radius": 0.75})
        goals = [parse_goal(legacy_goal)]

    all_polygons = [floor.polygon for floor in floors]
    all_polygons.extend(fixture.polygon for fixture in fixtures if fixture.polygon is not None)
    all_polygons.extend(ramp.polygon for ramp in ramps)
    xmins, xmaxs, ymins, ymaxs = zip(*(polygon_bbox(poly) for poly in all_polygons))
    z_goals = [g.min_corner[2] for g in goals] + [g.max_corner[2] for g in goals]
    bounds_min = np.array([
        min(xmins) - 1.0,
        min(ymins) - 1.0,
        min([*z_goals, z_min]) - 0.5,
    ], dtype=float)
    bounds_max = np.array([
        max(xmaxs) + 1.0,
        max(ymaxs) + 1.0,
        max([*z_goals, z_max]) + 1.0,
    ], dtype=float)

    return Floorplan(
        name=name,
        floors=floors,
        fixtures=fixtures,
        ramps=ramps,
        openings=openings,
        goals=goals,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )


def parse_repulsion(data: Dict) -> RepulsionConfig:
    anis = data.get("anisotropy", {})
    mode = str(data.get("mode", "")).lower()
    return RepulsionConfig(
        strength=float(data.get("strength", 1.6)),
        range=float(data.get("range", 1.0)),
        anisotropy_enabled=bool(anis.get("enabled", mode == "anisotropic")),
        forward_strength=float(anis.get("forward_strength", 1.6)),
        backward_strength=float(anis.get("backward_strength", 0.6)),
        forward_angle_deg=float(anis.get("forward_angle_deg", 80.0)),
    )


def parse_agent_defaults(data: Dict) -> AgentConfig:
    if not data:
        return AgentConfig()
    legacy = dict(data)
    return AgentConfig(
        radius=float(legacy.get("radius", 0.22)),
        max_speed=float(legacy.get("max_speed", 1.35)),
        nav_gain=float(legacy.get("nav_gain", legacy.get("goal_attraction", 3.0))),
        wall_strength=float(legacy.get("wall_strength", legacy.get("wall_repulsion", 5.5))),
        wall_range=float(legacy.get("wall_range", 1.0)),
        social_strength=float(legacy.get("social_strength", 1.8)),
        social_range=float(legacy.get("social_range", 1.1)),
        ramp_gain=float(legacy.get("ramp_gain", legacy.get("ramp_attraction", 3.6))),
        goal_gain=float(legacy.get("goal_gain", 2.2)),
        damping=float(legacy.get("damping", 0.72)),
    )


def load_scenario(spec: str, floors: List[Floor]) -> Scenario:
    data = load_json_from_folder(DATA_FOLDER, spec)
    simulation_raw = data.get("simulation", {})
    grid_raw = data.get("grid", {})
    simulation = SimulationConfig(
        dt=float(simulation_raw.get("dt", 0.05)),
        steps=int(simulation_raw.get("steps", 700)),
        seed=int(simulation_raw.get("seed", 7)),
        cell_size=float(grid_raw.get("cell_size", simulation_raw.get("cell_size", 0.35))),
        offscreen=bool(simulation_raw.get("offscreen", False)),
    )
    defaults = parse_agent_defaults(data.get("agent_defaults", data.get("agent", {})))

    raw_groups = data.get("agents", data.get("evacuees", []))
    groups: List[AgentGroup] = []
    for raw in raw_groups:
        if "spawn" in raw and "box" in raw.get("spawn", {}):
            spawn_box = raw["spawn"]["box"]
            center = 0.5 * (vec(spawn_box["min"]) + vec(spawn_box["max"]))
            floor_idx = nearest_floor_index(center[2], floors)
            floor = floors[floor_idx]
            min_box = vec(spawn_box["min"])
            max_box = vec(spawn_box["max"])
            fx0, fx1, fy0, fy1 = floor.bbox
            placement = {
                "floor": floor_idx,
                "absolute_box": {"min": min_box.tolist(), "max": max_box.tolist()},
                "offset_range": {
                    "x": [2.0 * (min_box[0] - 0.5 * (fx0 + fx1)) / max(fx1 - fx0, EPS), 2.0 * (max_box[0] - 0.5 * (fx0 + fx1)) / max(fx1 - fx0, EPS)],
                    "y": [2.0 * (min_box[1] - 0.5 * (fy0 + fy1)) / max(fy1 - fy0, EPS), 2.0 * (max_box[1] - 0.5 * (fy0 + fy1)) / max(fy1 - fy0, EPS)],
                },
            }
        else:
            floor_idx = int(raw.get("floor", 0))
            placement = dict(raw)
            placement["floor"] = floor_idx
        groups.append(
            AgentGroup(
                count=int(raw.get("count", 1)),
                floor=floor_idx,
                color=raw.get("color", random.choice(["royalblue", "tomato", "gold", "orchid", "cyan4"])),
                repulsion=parse_repulsion(raw.get("repulsion", {})),
                placement=placement,
            )
        )

    obstacles = [ScenarioObstacleSpec(raw) for raw in data.get("obstacles", [])]

    raw_harassers = data.get("harassers", [])
    harassers: List[HarasserGroup] = []
    for raw in raw_harassers:
        strategy = raw.get("strategy", "intercept")
        if strategy not in HARASSER_STRATEGIES:
            strategy = "intercept"
        harassers.append(
            HarasserGroup(
                count=int(raw.get("count", 1)),
                floor=int(raw.get("floor", 0)),
                offset=raw.get("offset", [0.0, 0.0]),
                color=raw.get("color", "darkred"),
                strength=float(raw.get("strength", 5.0)),
                radius=float(raw.get("radius", 0.3)),
                strategy=strategy,
            )
        )

    return Scenario(
        name=data.get("name", pathlib.Path(spec).stem),
        simulation=simulation,
        agent_defaults=defaults,
        groups=groups,
        obstacles=obstacles,
        harassers=harassers,
    )


def scenario_obstacle_to_fixture(spec: ScenarioObstacleSpec, floorplan: Floorplan) -> Fixture:
    raw = spec.raw
    floor_index = int(raw.get("floor", 0))
    floor = floorplan.floors[floor_index]
    floor_center = np.array([
        0.5 * (floor.bbox[0] + floor.bbox[1]),
        0.5 * (floor.bbox[2] + floor.bbox[3]),
    ])
    extents = np.array([floor.bbox[1] - floor.bbox[0], floor.bbox[3] - floor.bbox[2]], dtype=float)

    if raw.get("type", "box").lower() in {"column", "pillar"}:
        offset = np.array([normalize_percent(v) for v in raw.get("offset", [0.0, 0.0])], dtype=float)
        center_xy = floor_center + 0.5 * extents * offset
        base_z = floor.z
        top_z = floor.z + float(raw.get("height", 2.5))
        return Fixture(
            kind="column",
            center=center_xy,
            radius=float(raw.get("radius", 0.45)),
            base_z=base_z,
            top_z=top_z,
            color=raw.get("color", "tan"),
            alpha=float(raw.get("alpha", 0.35)),
        )

    size = np.array(raw.get("size", [1.0, 1.0]), dtype=float)
    offset = np.array([normalize_percent(v) for v in raw.get("offset", [0.0, 0.0])], dtype=float)
    center_xy = floor_center + 0.5 * extents * offset
    min_xy = center_xy - 0.5 * size
    max_xy = center_xy + 0.5 * size
    base_z = floor.z
    top_z = floor.z + float(raw.get("height", 1.2))
    return Fixture(
        kind="box",
        polygon=make_rect_polygon(min_xy, max_xy),
        base_z=base_z,
        top_z=top_z,
        color=raw.get("color", "brown"),
        alpha=float(raw.get("alpha", 0.35)),
    )


def apply_scenario_obstacles(floorplan: Floorplan, scenario: Scenario) -> None:
    for spec in scenario.obstacles:
        floorplan.fixtures.append(scenario_obstacle_to_fixture(spec, floorplan))

# Cutoff
def opening_contains_xy(opening: Opening, point_xy: Vector, z_value: float) -> bool:
    return (
        opening.min_corner[0] - EPS <= point_xy[0] <= opening.max_corner[0] + EPS
        and opening.min_corner[1] - EPS <= point_xy[1] <= opening.max_corner[1] + EPS
        and opening.min_corner[2] - EPS <= z_value <= opening.max_corner[2] + EPS
    )


def point_blocked_by_fixture(point_xy: Vector, z_value: float, fixture: Fixture) -> bool:
    if not fixture.active_at_z(z_value):
        return False
    if fixture.kind == "column":
        assert fixture.center is not None and fixture.radius is not None
        return norm(point_xy - fixture.center) <= fixture.radius + EPS
    if fixture.polygon is None:
        return False
    return point_in_polygon(point_xy, fixture.polygon)


def is_xy_walkable_on_floor(floorplan: Floorplan, floor_index: int, point_xy: Vector) -> bool:
    floor = floorplan.floors[floor_index]
    if not point_in_polygon(point_xy, floor.polygon):
        return False

    sample_z = floor.surface_z
    fixtures = list(floorplan.fixtures) + list(floor.obstacles)
    for fixture in fixtures:
        if not point_blocked_by_fixture(point_xy, sample_z, fixture):
            continue
        if any(opening_contains_xy(opening, point_xy, sample_z) for opening in floorplan.openings):
            continue
        return False
    return True


def floor_walkable_mask(
    floor: Floor,
    floorplan: Floorplan,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> np.ndarray:
    xx, yy = np.meshgrid(x_coords, y_coords)
    walkable = points_in_polygon(xx, yy, floor.polygon)
    sample_z = floor.surface_z

    for fixture in list(floorplan.fixtures) + list(floor.obstacles):
        if fixture.kind == "column":
            if not fixture.active_at_z(sample_z):
                continue
            cx, cy = float(fixture.center[0]), float(fixture.center[1])
            rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            blocked = rr <= float(fixture.radius) + EPS
        else:
            if fixture.polygon is None or not fixture.active_at_z(sample_z):
                continue
            blocked = points_in_polygon(xx, yy, fixture.polygon)

        if floorplan.openings:
            opening_mask = np.zeros_like(blocked)
            for opening in floorplan.openings:
                opening_mask |= (
                    (xx >= opening.min_corner[0] - EPS)
                    & (xx <= opening.max_corner[0] + EPS)
                    & (yy >= opening.min_corner[1] - EPS)
                    & (yy <= opening.max_corner[1] + EPS)
                    & (sample_z >= opening.min_corner[2] - EPS)
                    & (sample_z <= opening.max_corner[2] + EPS)
                )
            blocked &= ~opening_mask
        walkable &= ~blocked
    return walkable


def nearest_walkable_index(mask: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, xy: Vector) -> Optional[Tuple[int, int]]:
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return None
    coords = np.column_stack((x_coords[indices[:, 1]], y_coords[indices[:, 0]]))
    distances = np.sum((coords - xy[None, :]) ** 2, axis=1)
    best = int(np.argmin(distances))
    return int(indices[best, 0]), int(indices[best, 1])


def nearest_walkable_xy(field: NavigationField, floor_index: int, xy: Vector) -> Vector:
    ij = nearest_walkable_index(field.walkable[floor_index], field.x_coords, field.y_coords, xy)
    if ij is None:
        return xy.copy()
    iy, ix = ij
    return np.array([field.x_coords[ix], field.y_coords[iy]], dtype=float)


def sample_scalar(grid: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, xy: Vector) -> float:
    x = clamp(float(xy[0]), float(x_coords[0]), float(x_coords[-1]))
    y = clamp(float(xy[1]), float(y_coords[0]), float(y_coords[-1]))
    ix_hi = int(np.searchsorted(x_coords, x))
    iy_hi = int(np.searchsorted(y_coords, y))
    ix1 = min(max(ix_hi, 1), len(x_coords) - 1)
    iy1 = min(max(iy_hi, 1), len(y_coords) - 1)
    ix0 = ix1 - 1
    iy0 = iy1 - 1
    x0, x1 = x_coords[ix0], x_coords[ix1]
    y0, y1 = y_coords[iy0], y_coords[iy1]
    tx = 0.0 if abs(x1 - x0) < EPS else (x - x0) / (x1 - x0)
    ty = 0.0 if abs(y1 - y0) < EPS else (y - y0) / (y1 - y0)
    v00 = float(grid[iy0, ix0])
    v10 = float(grid[iy0, ix1])
    v01 = float(grid[iy1, ix0])
    v11 = float(grid[iy1, ix1])
    return (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11


def sample_gradient(grid: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, xy: Vector, step: float) -> Vector:
    dx = np.array([step, 0.0], dtype=float)
    dy = np.array([0.0, step], dtype=float)
    gx = (sample_scalar(grid, x_coords, y_coords, xy + dx) - sample_scalar(grid, x_coords, y_coords, xy - dx)) / (2.0 * step)
    gy = (sample_scalar(grid, x_coords, y_coords, xy + dy) - sample_scalar(grid, x_coords, y_coords, xy - dy)) / (2.0 * step)
    return np.array([gx, gy], dtype=float)


def ramp_height_at_xy(ramp: Ramp, point_xy: Vector) -> float:
    _, t = closest_point_on_segment_2d(point_xy, ramp.start[:2], ramp.end[:2])
    return float(ramp.start[2] + t * (ramp.end[2] - ramp.start[2]))


def ramp_tangent_xy(ramp: Ramp, toward_lower: bool = True) -> Vector:
    high = ramp.start if ramp.start[2] >= ramp.end[2] else ramp.end
    low = ramp.end if ramp.start[2] >= ramp.end[2] else ramp.start
    tangent = low[:2] - high[:2] if toward_lower else high[:2] - low[:2]
    return unit(tangent)


def get_ramp_entry_exit(ramp: Ramp) -> Tuple[Tuple[int, Vector], Tuple[int, Vector]]:
    if ramp.start[2] >= ramp.end[2]:
        return (ramp.from_floor, ramp.start[:2]), (ramp.to_floor, ramp.end[:2])
    return (ramp.to_floor, ramp.end[:2]), (ramp.from_floor, ramp.start[:2])


def get_ramp_by_name(floorplan: Floorplan, ramp_name: Optional[str]) -> Optional[Ramp]:
    if not ramp_name:
        return None
    for ramp in floorplan.ramps:
        if ramp.name == ramp_name:
            return ramp
    return None


def ramp_endpoint_for_floor(ramp: Ramp, floor_index: int) -> Optional[Vector]:
    if floor_index == ramp.from_floor:
        return ramp.start[:2].copy()
    if floor_index == ramp.to_floor:
        return ramp.end[:2].copy()
    return None


def can_enter_ramp_from_floor(ramp: Ramp, goal_floor: int, floor_index: int, point_xy: Vector) -> bool:
    travel = get_ramp_travel_for_goal(ramp, goal_floor)
    if travel is None:
        return False
    entry_floor, endpoint, _, _, _ = travel
    if floor_index != entry_floor:
        return False

    _, along, lateral, _, _, length = ramp_local_coordinates(ramp, point_xy)
    half_width = 0.5 * ramp.width
    lateral_margin = max(0.15, 0.35 * ramp.width)
    entry_depth = min(length * 0.4, max(1.25, 0.9 * ramp.width))

    if floor_index == ramp.from_floor:
        in_along_band = -0.15 <= along <= entry_depth
    elif floor_index == ramp.to_floor:
        in_along_band = (length - entry_depth) <= along <= (length + 0.15)
    else:
        return False

    in_width_band = abs(lateral) <= half_width + lateral_margin
    return in_along_band and in_width_band and norm(point_xy - endpoint) <= max(0.9, ramp.width)


def ramp_local_coordinates(ramp: Ramp, point_xy: Vector) -> Tuple[Vector, float, float, Vector, Vector, float]:
    start_xy = ramp.start[:2]
    end_xy = ramp.end[:2]
    axis = end_xy - start_xy
    length = max(norm(axis), EPS)
    tangent = axis / length
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    rel = point_xy - start_xy
    along = float(np.dot(rel, tangent))
    lateral = float(np.dot(rel, normal))
    closest_xy = start_xy + clamp(along, 0.0, length) * tangent
    return closest_xy, along, lateral, tangent, normal, length


def point_near_ramp(ramp: Ramp, point_xy: Vector, margin: float = 0.0) -> bool:
    _, along, lateral, _, _, length = ramp_local_coordinates(ramp, point_xy)
    half_width = 0.5 * ramp.width
    return (-margin <= along <= length + margin) and (abs(lateral) <= half_width + margin)


def clamp_point_to_ramp(ramp: Ramp, point_xy: Vector, agent_radius: float) -> Vector:
    _, along, lateral, tangent, normal, length = ramp_local_coordinates(ramp, point_xy)
    half_width = max(0.05, 0.5 * ramp.width - 0.55 * agent_radius)
    along_clamped = clamp(along, 0.0, length)
    lateral_clamped = clamp(lateral, -half_width, half_width)
    return ramp.start[:2] + along_clamped * tangent + lateral_clamped * normal


def ramps_for_floor(floorplan: Floorplan, floor_index: int) -> List[Ramp]:
    return [
        ramp
        for ramp in floorplan.ramps
        if floor_index in {ramp.from_floor, ramp.to_floor}
    ]


def get_ramp_travel_for_goal(
    ramp: Ramp,
    goal_floor: int,
) -> Optional[Tuple[int, Vector, int, Vector, Vector]]:
    (upper_floor, upper_xy), (lower_floor, lower_xy) = get_ramp_entry_exit(ramp)
    upper_delta = abs(upper_floor - goal_floor)
    lower_delta = abs(lower_floor - goal_floor)
    if upper_delta == lower_delta:
        return None
    if lower_delta < upper_delta:
        tangent = unit(lower_xy - upper_xy)
        return upper_floor, upper_xy.copy(), lower_floor, lower_xy.copy(), tangent
    tangent = unit(upper_xy - lower_xy)
    return lower_floor, lower_xy.copy(), upper_floor, upper_xy.copy(), tangent


def ramps_toward_goal_for_floor(floorplan: Floorplan, floor_index: int, goal_floor: int) -> List[Ramp]:
    ramps: List[Ramp] = []
    for ramp in ramps_for_floor(floorplan, floor_index):
        travel = get_ramp_travel_for_goal(ramp, goal_floor)
        if travel is None:
            continue
        entry_floor, _, _, _, _ = travel
        if entry_floor == floor_index:
            ramps.append(ramp)
    return ramps


def local_target_xy_for_floor(
    floorplan: Floorplan,
    floor_index: int,
    goal_floor: int,
    point_xy: Optional[Vector] = None,
) -> Vector:
    if floor_index == goal_floor:
        goals_on_floor = [g for g in floorplan.goals if nearest_floor_index(g.center[2], floorplan.floors) == floor_index]
        if not goals_on_floor:
            goals_on_floor = [floorplan.primary_goal]
        if point_xy is not None:
            nearest = min(goals_on_floor, key=lambda g: norm(point_xy - g.center[:2]))
            return nearest.center[:2].copy()
        return goals_on_floor[0].center[:2].copy()

    ramps = ramps_toward_goal_for_floor(floorplan, floor_index, goal_floor)
    if not ramps:
        return floorplan.primary_goal.center[:2].copy()

    if point_xy is None:
        floor = floorplan.floors[floor_index]
        bx0, bx1, by0, by1 = floor.bbox
        point_xy = np.array([0.5 * (bx0 + bx1), 0.5 * (by0 + by1)], dtype=float)

    best_endpoint = None
    best_distance = INF
    for ramp in ramps:
        travel = get_ramp_travel_for_goal(ramp, goal_floor)
        assert travel is not None
        _, endpoint, _, _, _ = travel
        distance = norm(point_xy - endpoint)
        if distance < best_distance:
            best_distance = distance
            best_endpoint = endpoint

    if best_endpoint is None:
        return floorplan.primary_goal.center[:2].copy()
    return best_endpoint.copy()


def build_navigation_field(floorplan: Floorplan, scenario: Scenario) -> NavigationField:
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise RuntimeError("scipy is required to build the navigation field.") from exc

    all_polygons = [floor.polygon for floor in floorplan.floors] + [ramp.polygon for ramp in floorplan.ramps]
    xmins, xmaxs, ymins, ymaxs = zip(*(polygon_bbox(poly) for poly in all_polygons))
    padding = 1.0
    cell_size = scenario.simulation.cell_size
    x_coords = np.arange(min(xmins) - padding, max(xmaxs) + padding + cell_size, cell_size)
    y_coords = np.arange(min(ymins) - padding, max(ymaxs) + padding + cell_size, cell_size)
    floor_lookup = {floor.index: floor for floor in floorplan.floors}

    walkable_layers = []
    distance_layers = []
    for floor in floorplan.floors:
        mask = floor_walkable_mask(floor, floorplan, x_coords, y_coords)
        walkable_layers.append(mask)
        distance_layers.append(distance_transform_edt(mask) * cell_size)

    walkable = np.stack(walkable_layers, axis=0)
    distance_to_block = np.stack(distance_layers, axis=0)
    goal_floor = nearest_floor_index(floorplan.primary_goal.center[2], floorplan.floors)

    return NavigationField(
        cell_size=cell_size,
        x_coords=x_coords,
        y_coords=y_coords,
        walkable=walkable,
        distance_to_block=distance_to_block,
        goal_floor=goal_floor,
        floor_lookup=floor_lookup,
    )


def relative_sample_on_floor(floor: Floor, placement: Dict) -> Vector:
    bx0, bx1, by0, by1 = floor.bbox
    center = np.array([0.5 * (bx0 + bx1), 0.5 * (by0 + by1)], dtype=float)
    extent = np.array([bx1 - bx0, by1 - by0], dtype=float)

    if "absolute_box" in placement:
        box = placement["absolute_box"]
        min_corner = vec(box["min"])
        max_corner = vec(box["max"])
        return np.array(
            [
                random.uniform(min_corner[0], max_corner[0]),
                random.uniform(min_corner[1], max_corner[1]),
            ],
            dtype=float,
        )

    xr = placement.get("offset_range", {}).get("x")
    yr = placement.get("offset_range", {}).get("y")
    if xr is None or yr is None:
        base = placement.get("offset", [0.0, 0.0])
        jitter = placement.get("jitter", [0.08, 0.08])
        xr = [normalize_percent(base[0]) - abs(normalize_percent(jitter[0])), normalize_percent(base[0]) + abs(normalize_percent(jitter[0]))]
        yr = [normalize_percent(base[1]) - abs(normalize_percent(jitter[1])), normalize_percent(base[1]) + abs(normalize_percent(jitter[1]))]
    xr = [normalize_percent(float(xr[0])), normalize_percent(float(xr[1]))]
    yr = [normalize_percent(float(yr[0])), normalize_percent(float(yr[1]))]

    for _ in range(300):
        rel = np.array([
            random.uniform(xr[0], xr[1]),
            random.uniform(yr[0], yr[1]),
        ], dtype=float)
        point_xy = center + 0.5 * extent * rel
        if point_in_polygon(point_xy, floor.polygon):
            return point_xy
    return center.copy()


def support_surface_z(
    floorplan: Floorplan,
    goal_floor: int,
    floor_index: int,
    point_xy: Vector,
) -> Tuple[float, Optional[Ramp]]:
    candidate_ramp = None
    for ramp in ramps_toward_goal_for_floor(floorplan, floor_index, goal_floor):
        if point_in_polygon(point_xy, ramp.polygon) and can_enter_ramp_from_floor(
            ramp,
            goal_floor,
            floor_index,
            point_xy,
        ):
            candidate_ramp = ramp
            break
    if candidate_ramp is not None:
        return ramp_height_at_xy(candidate_ramp, point_xy), candidate_ramp
    floor = floorplan.floors[floor_index]
    return floor.surface_z, None


def resolve_spawn_position(field: NavigationField, floorplan: Floorplan, floor_index: int, point_xy: Vector, radius: float) -> Tuple[Vector, int, Optional[str]]:
    snapped_xy = nearest_walkable_xy(field, floor_index, point_xy)
    z_value, ramp = support_surface_z(floorplan, field.goal_floor, floor_index, snapped_xy)
    return np.array([snapped_xy[0], snapped_xy[1], z_value + radius], dtype=float), floor_index, None if ramp is None else ramp.name


def initialize_evacuees(scenario: Scenario, floorplan: Floorplan, field: NavigationField) -> List[Evacuee]:
    evacuees: List[Evacuee] = []
    for group in scenario.groups:
        floor = floorplan.floors[group.floor]
        for _ in range(group.count):
            point_xy = relative_sample_on_floor(floor, group.placement)
            position, floor_index, ramp_name = resolve_spawn_position(
                field,
                floorplan,
                group.floor,
                point_xy,
                scenario.agent_defaults.radius,
            )
            evacuee = Evacuee(
                pos=position,
                vel=np.zeros(3, dtype=float),
                floor_index=floor_index,
                radius=scenario.agent_defaults.radius,
                color=group.color,
                repulsion=group.repulsion,
            )
            evacuee.ramp_name = ramp_name
            evacuees.append(evacuee)
    return evacuees


def create_environment(
    scenario: Scenario,
    floorplan: Floorplan,
    field: NavigationField,
) -> Environment:
    env = Environment(floorplan=floorplan, field=field)
    evacuees = initialize_evacuees(scenario, floorplan, field)
    for evacuee in evacuees:
        env.add_actor(evacuee)
    for harasser_group in scenario.harassers:
        floor = floorplan.floors[harasser_group.floor]
        for _ in range(harasser_group.count):
            center_x = floor.bbox[0] + harasser_group.offset[0]
            center_y = floor.bbox[2] + harasser_group.offset[1]
            center_z = floor.surface_z + harasser_group.radius
            pos = np.array([center_x, center_y, center_z], dtype=float)
            harasser = Harasser(
                pos=pos,
                vel=np.zeros(3, dtype=float),
                floor_index=harasser_group.floor,
                radius=harasser_group.radius,
                color=harasser_group.color,
                strength=harasser_group.strength,
                strategy=harasser_group.strategy,
            )
            env.add_actor(harasser)
    for spec in scenario.obstacles:
        raw = spec.raw
        if raw.get("repulsive", False):
            floor_idx = raw["floor"]
            floor = floorplan.floors[floor_idx]
            center_x = floor.bbox[0] + raw["offset"][0] + raw["size"][0] / 2
            center_y = floor.bbox[2] + raw["offset"][1] + raw["size"][1] / 2
            center_z = floor.surface_z + raw["height"] / 2
            radius = max(raw["size"][0], raw["size"][1]) / 2
            body = Body(pos=np.array([center_x, center_y, center_z]), radius=radius, owner_id=0)
            env.add_dynamic_obstacle(body)
    return env


def anisotropic_weight(forward_dir: Vector, direction_to_other: Vector, cfg: RepulsionConfig) -> float:
    if not cfg.anisotropy_enabled:
        return 1.0
    fd = unit(forward_dir)
    od = unit(direction_to_other)
    if norm(fd) < EPS or norm(od) < EPS:
        return 1.0
    angle = math.degrees(math.acos(clamp(float(np.dot(fd, od)), -1.0, 1.0)))
    return cfg.forward_strength if angle <= cfg.forward_angle_deg else cfg.backward_strength


def choose_guiding_ramp(agent, floorplan: Floorplan, goal_floor: int) -> Optional[Ramp]:
    best_ramp = None
    best_value = INF
    agent_xy = agent.pos[:2]
    for ramp in ramps_toward_goal_for_floor(floorplan, agent.floor_index, goal_floor):
        travel = get_ramp_travel_for_goal(ramp, goal_floor)
        assert travel is not None
        _, entry_xy, _, _, _ = travel
        score = norm(agent_xy - entry_xy)
        if score < best_value:
            best_value = score
            best_ramp = ramp
    return best_ramp


def compute_wall_force(agent, field: NavigationField, cfg: AgentConfig) -> Vector:
    dist_grid = field.distance_to_block[agent.floor_index]
    d = sample_scalar(dist_grid, field.x_coords, field.y_coords, agent.pos[:2])
    effective_range = min(cfg.wall_range, max(0.45, 2.4 * agent.radius + 0.12))
    if not math.isfinite(d) or d <= EPS or d >= effective_range:
        return np.zeros(3, dtype=float)
    grad_d = sample_gradient(dist_grid, field.x_coords, field.y_coords, agent.pos[:2], field.cell_size)
    direction = unit(grad_d)
    magnitude = cfg.wall_strength * ((1.0 / max(d, 0.05)) - (1.0 / effective_range)) / max(d * d, 0.05)
    magnitude = min(magnitude, 1.25 * cfg.wall_strength)
    return np.array([direction[0], direction[1], 0.0], dtype=float) * magnitude


def compute_ramp_edge_force(agent, floorplan: Floorplan, cfg: AgentConfig) -> Vector:
    ramp = get_ramp_by_name(floorplan, agent.ramp_name)
    if ramp is None:
        return np.zeros(3, dtype=float)

    _, along, lateral, _, normal, length = ramp_local_coordinates(ramp, agent.pos[:2])
    if along < -agent.radius or along > length + agent.radius:
        return np.zeros(3, dtype=float)

    half_width = 0.5 * ramp.width
    distance_to_edge = half_width - abs(lateral)
    if distance_to_edge >= cfg.wall_range:
        return np.zeros(3, dtype=float)

    sign = -1.0 if lateral >= 0.0 else 1.0
    direction = sign * normal
    clearance = max(distance_to_edge, 0.03)
    magnitude = cfg.wall_strength * ((1.0 / clearance) - (1.0 / cfg.wall_range)) / max(clearance * clearance, 0.03)
    if distance_to_edge < 0.0:
        magnitude *= 2.5
    return np.array([direction[0], direction[1], 0.0], dtype=float) * magnitude


def _compute_navigation_force_for_evacuee(
    evacuee: Evacuee,
    actors: List[Actor],
    dynamic_obstacles: List[Body],
    floorplan: Floorplan,
    field: NavigationField,
    cfg: AgentConfig,
) -> Vector:
    evacuee_xy = evacuee.pos[:2]
    force_xy = np.zeros(2, dtype=float)

    for goal in floorplan.goals:
        goal_xy = goal.center[:2]
        delta = goal_xy - evacuee_xy
        dist = norm(delta)
        if dist > EPS:
            attraction = cfg.goal_gain / max(dist * dist, 0.5)
            force_xy += unit(delta) * attraction

    active_ramp = get_ramp_by_name(floorplan, evacuee.ramp_name)
    if active_ramp is not None:
        travel = get_ramp_travel_for_goal(active_ramp, field.goal_floor)
        if travel is None:
            return np.array([force_xy[0], force_xy[1], 0.0], dtype=float)
        entry_floor, entry_xy, exit_floor, exit_xy, travel_tangent = travel
        centerline_xy, along, lateral, tangent, normal, length = ramp_local_coordinates(
            active_ramp,
            evacuee_xy,
        )

        center_correction = centerline_xy - evacuee_xy
        exit_target = exit_xy - evacuee_xy
        if entry_floor == active_ramp.from_floor:
            progress = clamp(along / max(length, EPS), 0.0, 1.0)
        else:
            progress = clamp((length - along) / max(length, EPS), 0.0, 1.0)
        half_width = max(0.5 * active_ramp.width, EPS)
        lateral_ratio = clamp(abs(lateral) / half_width, 0.0, 1.0)

        force_xy += 1.35 * cfg.ramp_gain * travel_tangent
        force_xy += 0.9 * cfg.ramp_gain * lateral_ratio * unit(center_correction)
        force_xy += (0.55 + 0.35 * progress) * cfg.ramp_gain * unit(exit_target)

        if norm(exit_target) < max(0.8, 0.6 * active_ramp.width):
            next_target = local_target_xy_for_floor(
                floorplan,
                exit_floor,
                field.goal_floor,
                exit_xy,
            )
            force_xy += 0.4 * cfg.nav_gain * unit(next_target - evacuee_xy)

        return np.array([force_xy[0], force_xy[1], 0.0], dtype=float)

    target_xy = local_target_xy_for_floor(floorplan, evacuee.floor_index, field.goal_floor, evacuee_xy)
    force_xy += cfg.nav_gain * unit(target_xy - evacuee_xy)

    ramp = _choose_ramp_by_attractiveness(
        evacuee, actors, dynamic_obstacles, floorplan, field.goal_floor, cfg
    )
    if ramp is not None:
        travel = get_ramp_travel_for_goal(ramp, field.goal_floor)
        assert travel is not None
        entry_floor, entry_xy, exit_floor, exit_xy, travel_tangent = travel
        if evacuee.floor_index == entry_floor:
            dist_to_polygon = 0.0 if point_in_polygon(evacuee_xy, ramp.polygon) else norm(evacuee_xy - entry_xy)
            entry_target = entry_xy - evacuee_xy
            if can_enter_ramp_from_floor(ramp, field.goal_floor, evacuee.floor_index, evacuee_xy):
                centerline_at, _ = closest_point_on_segment_2d(evacuee_xy, ramp.start[:2], ramp.end[:2])
                align = unit(centerline_at - evacuee_xy)
                force_xy += cfg.ramp_gain * travel_tangent + 0.6 * cfg.ramp_gain * align
            elif dist_to_polygon < max(2.0 * ramp.width, 2.5):
                centerline_at, _ = closest_point_on_segment_2d(evacuee_xy, ramp.start[:2], ramp.end[:2])
                align = unit(centerline_at - evacuee_xy)
                force_xy += 1.15 * cfg.ramp_gain * unit(entry_target)
                force_xy += 0.35 * cfg.ramp_gain * align
            elif norm(entry_target) > EPS:
                force_xy += 0.45 * cfg.ramp_gain * unit(entry_target)

    return np.array([force_xy[0], force_xy[1], 0.0], dtype=float)


def _calculate_repulsion_at_point(
    point_xy: np.ndarray,
    actors: List[Actor],
    dynamic_obstacles: List[Body],
    cfg: AgentConfig,
) -> float:
    total_repulsion = 0.0
    for actor in actors:
        if not isinstance(actor, Evacuee):
            continue
        if not actor.active or actor.reached_goal:
            continue
        delta = point_xy - actor.pos[:2]
        dist = norm(delta)
        if dist > EPS and dist < cfg.social_range:
            total_repulsion += cfg.social_strength * (1.0 / max(dist, 0.1))
    for obs in dynamic_obstacles:
        delta = point_xy - obs.pos[:2]
        dist = norm(delta)
        if dist > EPS and dist < cfg.wall_range:
            total_repulsion += cfg.wall_strength * (1.0 / max(dist, 0.1))
    return total_repulsion


def _choose_ramp_by_attractiveness(
    evacuee: Evacuee,
    actors: List[Actor],
    dynamic_obstacles: List[Body],
    floorplan: Floorplan,
    goal_floor: int,
    cfg: AgentConfig,
) -> Optional[Ramp]:
    best_ramp = None
    best_attractiveness = -INF
    evacuee_xy = evacuee.pos[:2]

    for ramp in ramps_toward_goal_for_floor(floorplan, evacuee.floor_index, goal_floor):
        travel = get_ramp_travel_for_goal(ramp, goal_floor)
        if travel is None:
            continue
        _, entry_xy, _, _, _ = travel

        dist_to_ramp = norm(evacuee_xy - entry_xy)
        if dist_to_ramp < EPS:
            dist_to_ramp = EPS

        base_attraction = cfg.ramp_gain / max(dist_to_ramp, 0.5)

        repulsion_at_entry = _calculate_repulsion_at_point(entry_xy, actors, dynamic_obstacles, cfg)

        attractiveness = base_attraction - repulsion_at_entry

        if attractiveness > best_attractiveness:
            best_attractiveness = attractiveness
            best_ramp = ramp

    return best_ramp


def compute_evacuee_force(
    evacuee: Evacuee,
    actors: List[Actor],
    floorplan: Floorplan,
    field: NavigationField,
    dynamic_obstacles: List[Body],
    cfg: AgentConfig,
) -> Vector:
    if not evacuee.active:
        return np.zeros(3, dtype=float)

    force = np.zeros(3, dtype=float)
    navigation_force = _compute_navigation_force_for_evacuee(
        evacuee, actors, dynamic_obstacles, floorplan, field, cfg
    )
    wall_force = compute_wall_force(evacuee, field, cfg)
    ramp_edge_force = compute_ramp_edge_force(evacuee, floorplan, cfg)

    nav_norm = norm(navigation_force[:2])
    wall_follow = np.zeros(3, dtype=float)
    if nav_norm > EPS:
        nav_dir = navigation_force[:2] / nav_norm
        wall_against_nav = float(np.dot(wall_force[:2], nav_dir))
        if wall_against_nav < 0.0:
            wall_dir = unit(wall_force[:2])
            if norm(wall_dir) > EPS:
                tangent_a = np.array([-wall_dir[1], wall_dir[0]], dtype=float)
                tangent_b = -tangent_a
                chosen_tangent = tangent_a if float(np.dot(tangent_a, nav_dir)) >= float(np.dot(tangent_b, nav_dir)) else tangent_b
                wall_follow[:2] = 0.9 * min(abs(wall_against_nav), 2.0 * cfg.wall_strength) * chosen_tangent
            wall_force[:2] -= wall_against_nav * nav_dir

    force += navigation_force
    force += wall_force
    force += wall_follow
    force += ramp_edge_force
    force += _compute_social_force_for_evacuee(evacuee, actors, floorplan, cfg)

    if dynamic_obstacles:
        force += _compute_dynamic_obstacle_force(evacuee, dynamic_obstacles, cfg)

    return force


def _compute_social_force_for_evacuee(
    evacuee: Evacuee,
    actors: List[Actor],
    floorplan: Floorplan,
    cfg: AgentConfig,
) -> Vector:
    total = np.zeros(3, dtype=float)
    for actor in actors:
        if actor is evacuee:
            continue
        if isinstance(actor, Evacuee):
            if not actor.active or actor.reached_goal:
                continue
            delta = evacuee.pos - actor.pos
            distance = norm(delta)
            if distance < EPS or distance > max(cfg.social_range, actor.repulsion.range):
                continue
            weight = anisotropic_weight(actor.vel, evacuee.pos - actor.pos, actor.repulsion)
            effective_range = max(0.25, actor.repulsion.range)
            if distance >= effective_range:
                continue
            direction = unit(delta)
            strength = cfg.social_strength * actor.repulsion.strength * weight
            total += direction * strength * ((1.0 / max(distance, 0.1)) - (1.0 / effective_range)) / max(distance, 0.15)
        elif isinstance(actor, Harasser):
            if not actor.active:
                continue
            delta = evacuee.pos - actor.pos
            distance = norm(delta)
            effective_range = cfg.social_range * 2.0
            if distance < EPS or distance > effective_range:
                continue
            if distance >= effective_range:
                continue
            direction = unit(delta)
            strength = cfg.social_strength * 15.0
            total += direction * strength * ((1.0 / max(distance, 0.1)) - (1.0 / effective_range)) / max(distance, 0.15)

    if any(distance_point_to_aabb(evacuee.pos, g.min_corner, g.max_corner) <= 2.0 * evacuee.radius for g in floorplan.goals):
        total *= 0.1
    return total


def _compute_dynamic_obstacle_force(evacuee: Evacuee, obstacles: List[Body], cfg: AgentConfig) -> Vector:
    total = np.zeros(3, dtype=float)
    for obs in obstacles:
        delta = evacuee.pos - obs.pos
        distance = norm(delta)
        combined_radius = evacuee.radius + obs.radius
        if distance < EPS or distance > max(cfg.social_range, combined_radius * 1.5):
            continue
        if distance >= combined_radius * 1.2:
            continue
        direction = unit(delta)
        strength = cfg.social_strength * 1.5
        clearance = max(distance - combined_radius, 0.1)
        total += direction * strength * (1.0 / clearance)
    return total


RAMP_UNDERSIDE_REPULSION_RADIUS = 2.5
RAMP_UNDERSIDE_REPULSION_STRENGTH = 15.0

CLUSTER_DISTANCE_THRESHOLD = 2.0
CLUSTER_MIN_SIZE = 2

HARASSER_STRATEGIES = ["nearest", "density", "flow", "intercept"]


def _compute_density_grid(
    evacuees: List["Evacuee"],
    floorplan: Floorplan,
    cell_size: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds = floorplan.bounds_min[:2], floorplan.bounds_max[:2]
    x_min, y_min = bounds[0]
    x_max, y_max = bounds[1]

    x_coords = np.arange(x_min, x_max + cell_size, cell_size)
    y_coords = np.arange(y_min, y_max + cell_size, cell_size)
    density = np.zeros((len(y_coords), len(x_coords)))

    for evacuee in evacuees:
        if not evacuee.active or evacuee.reached_goal:
            continue
        ex, ey = evacuee.pos[0], evacuee.pos[1]
        ix = int((ex - x_min) / cell_size)
        iy = int((ey - y_min) / cell_size)
        if 0 <= ix < len(x_coords) and 0 <= iy < len(y_coords):
            density[iy, ix] += 1.0

    return density, x_coords, y_coords


def _find_highest_density_point(
    evacuees: List["Evacuee"],
    floorplan: Floorplan,
    actor_pos: np.ndarray,
) -> np.ndarray:
    density, x_coords, y_coords = _compute_density_grid(evacuees, floorplan)

    if density.max() == 0:
        return actor_pos[:2].copy()

    iy, ix = np.unravel_index(density.argmax(), density.shape)
    center_x = x_coords[ix] + (x_coords[1] - x_coords[0]) / 2 if len(x_coords) > 1 else x_coords[0]
    center_y = y_coords[iy] + (y_coords[1] - y_coords[0]) / 2 if len(y_coords) > 1 else y_coords[0]

    return np.array([center_x, center_y], dtype=float)


def _compute_flow_target(
    evacuees: List["Evacuee"],
    actor_pos: np.ndarray,
    floorplan: Floorplan,
    prediction_time: float = 2.0,
) -> np.ndarray:
    if not evacuees:
        return actor_pos[:2].copy()

    total_pos = np.zeros(2, dtype=float)
    total_vel = np.zeros(2, dtype=float)
    total_weight = 0.0

    for evacuee in evacuees:
        if not evacuee.active or evacuee.reached_goal:
            continue

        pos = evacuee.pos[:2]
        vel = evacuee.vel[:2]

        dist_to_goal = min(norm(pos - g.center[:2]) for g in floorplan.goals)
        weight = 1.0 / max(dist_to_goal, 0.5)

        total_pos += pos * weight
        total_vel += vel * weight
        total_weight += weight

    if total_weight > EPS:
        center = total_pos / total_weight
        avg_vel = total_vel / total_weight
        target = center + avg_vel * prediction_time
        return target

    return actor_pos[:2].copy()


def _find_evacuee_clusters(evacuees: List["Evacuee"]) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    if len(evacuees) < CLUSTER_MIN_SIZE:
        return []

    positions = np.array([e.pos[:2] for e in evacuees])
    visited = set()
    clusters = []

    for i in range(len(evacuees)):
        if i in visited:
            continue

        cluster_indices = [i]
        visited.add(i)
        queue = [i]

        while queue:
            current = queue.pop(0)
            for j in range(len(evacuees)):
                if j in visited:
                    continue
                dist = norm(positions[current] - positions[j])
                if dist < CLUSTER_DISTANCE_THRESHOLD:
                    visited.add(j)
                    queue.append(j)
                    cluster_indices.append(j)

        if len(cluster_indices) >= CLUSTER_MIN_SIZE:
            cluster_positions = positions[cluster_indices]
            centroid = np.mean(cluster_positions, axis=0)
            velocities = np.array([evacuees[idx].vel[:2] for idx in cluster_indices])
            avg_velocity = np.mean(velocities, axis=0) if len(velocities) > 0 else np.zeros(2)
            clusters.append((centroid, avg_velocity, len(cluster_indices)))

    return clusters


def _compute_cluster_attraction(
    actor_pos: np.ndarray,
    clusters: List[Tuple[np.ndarray, np.ndarray, int]],
    floorplan: Floorplan,
    prediction_time: float = 2.0,
) -> np.ndarray:
    if not clusters:
        return np.zeros(2, dtype=float)

    target = np.zeros(2, dtype=float)
    total_weight = 0.0

    for centroid, velocity, size in clusters:
        predicted_pos = centroid + velocity * prediction_time

        min_dist_to_goal = INF
        for goal in floorplan.goals:
            dist = norm(predicted_pos - goal.center[:2])
            min_dist_to_goal = min(min_dist_to_goal, dist)

        urgency = 1.0 / max(min_dist_to_goal, 0.5)

        delta = predicted_pos - actor_pos[:2]
        dist = norm(delta)
        if dist < EPS:
            continue

        weight = float(size) * urgency
        target += unit(delta) * weight
        total_weight += weight

    if total_weight > EPS:
        return target / total_weight
    return np.zeros(2, dtype=float)


def _compute_ramp_underside_repulsion(
    actor_pos: np.ndarray,
    floorplan: Floorplan,
) -> Vector:
    total = np.zeros(3, dtype=float)
    actor_xy = actor_pos[:2]

    for ramp in floorplan.ramps:
        start_xy = ramp.start[:2]
        end_xy = ramp.end[:2]

        closest_point, _ = closest_point_on_segment_2d(actor_xy, start_xy, end_xy)
        delta = actor_xy - closest_point
        dist = norm(delta)

        if dist > RAMP_UNDERSIDE_REPULSION_RADIUS or dist < EPS:
            continue

        tangent = unit(end_xy - start_xy)
        perp = np.array([-tangent[1], tangent[0]], dtype=float)

        to_actor = unit(delta)
        perp_alignment = abs(float(np.dot(to_actor, perp)))

        if perp_alignment < 0.2:
            continue

        perp_sign = 1.0 if float(np.dot(to_actor, perp)) > 0 else -1.0

        strength = RAMP_UNDERSIDE_REPULSION_STRENGTH * (1.0 - dist / RAMP_UNDERSIDE_REPULSION_RADIUS)

        slide_dir = perp * perp_sign

        total[:2] += slide_dir * strength

    return total


def _project_evacuee_to_walkable(evacuee: Evacuee, floorplan: Floorplan, field: NavigationField) -> None:
    current_ramp = get_ramp_by_name(floorplan, evacuee.ramp_name)
    if current_ramp is not None and point_near_ramp(
        current_ramp,
        evacuee.pos[:2],
        margin=max(evacuee.radius * 2.0, 0.2),
    ):
        evacuee.pos[:2] = clamp_point_to_ramp(current_ramp, evacuee.pos[:2], evacuee.radius)
        return

    if is_xy_walkable_on_floor(floorplan, evacuee.floor_index, evacuee.pos[:2]):
        return
    snapped_xy = nearest_walkable_xy(field, evacuee.floor_index, evacuee.pos[:2])
    evacuee.pos[:2] = snapped_xy
    evacuee.vel[:2] *= 0.25


def _update_evacuee_surface(evacuee: Evacuee, floorplan: Floorplan, field: NavigationField) -> None:
    xy = evacuee.pos[:2]
    ramp = get_ramp_by_name(floorplan, evacuee.ramp_name)
    if ramp is not None and not point_near_ramp(ramp, xy, margin=max(evacuee.radius * 2.0, 0.2)):
        ramp = None

    if ramp is None:
        for candidate in ramps_toward_goal_for_floor(floorplan, evacuee.floor_index, field.goal_floor):
            if not can_enter_ramp_from_floor(candidate, field.goal_floor, evacuee.floor_index, xy):
                continue
            if point_in_polygon(xy, candidate.polygon) or point_near_ramp(
                candidate,
                xy,
                margin=max(evacuee.radius * 1.5, 0.15),
            ):
                ramp = candidate
                break

    if ramp is not None:
        travel = get_ramp_travel_for_goal(ramp, field.goal_floor)
        if travel is None:
            evacuee.ramp_name = None
            return
        _, _, exit_floor, exit_xy, _ = travel
        evacuee.pos[:2] = clamp_point_to_ramp(ramp, xy, evacuee.radius)
        xy = evacuee.pos[:2]
        evacuee.ramp_name = ramp.name
        high_floor = ramp.from_floor if ramp.start[2] >= ramp.end[2] else ramp.to_floor
        low_floor = ramp.to_floor if high_floor == ramp.from_floor else ramp.from_floor
        _, t = closest_point_on_segment_2d(xy, ramp.start[:2], ramp.end[:2])
        evacuee.floor_index = high_floor if t < 0.5 else low_floor
        evacuee.pos[2] = ramp_height_at_xy(ramp, xy) + evacuee.radius
        if norm(xy - exit_xy) <= max(ramp.width * 0.4, 0.55):
            evacuee.floor_index = exit_floor
            evacuee.ramp_name = None
        return

    floor = floorplan.floors[evacuee.floor_index]
    if not point_in_polygon(xy, floor.polygon):
        best_floor = evacuee.floor_index
        best_distance = INF
        for candidate in floorplan.floors:
            snapped = nearest_walkable_xy(field, candidate.index, xy)
            dist = norm(snapped - xy)
            if dist < best_distance:
                best_distance = dist
                best_floor = candidate.index
        evacuee.floor_index = best_floor
        floor = floorplan.floors[best_floor]
    evacuee.ramp_name = None
    evacuee.pos[2] = floor.surface_z + evacuee.radius


def create_mesh_actor(vertices: List[List[float]], faces: List[List[int]], color: str, alpha: float):
    from vedo import Mesh

    mesh = Mesh([vertices, faces])
    mesh.c(color).alpha(alpha)
    return mesh


def build_scene(floorplan: Floorplan):
    from vedo import Box, Cylinder

    actors = []

    for floor in floorplan.floors:
        verts, faces = build_surface_mesh(floor.polygon, floor.surface_z)
        actors.append(create_mesh_actor(verts, faces, floor.color, 0.12))

    for fixture in floorplan.fixtures:
        if fixture.kind == "column":
            assert fixture.center is not None and fixture.radius is not None
            height = fixture.top_z - fixture.base_z
            cyl = Cylinder(
                pos=(fixture.center[0], fixture.center[1], fixture.base_z + 0.5 * height),
                r=fixture.radius,
                height=height,
                axis=(0, 0, 1),
                c=fixture.color,
                alpha=fixture.alpha,
            )
            actors.append(cyl)
        elif fixture.polygon is not None:
            verts, faces = build_prism_mesh(fixture.polygon, fixture.base_z, fixture.top_z)
            actors.append(create_mesh_actor(verts, faces, fixture.color, fixture.alpha))

    for floor in floorplan.floors:
        for obstacle in floor.obstacles:
            if obstacle.kind == "column":
                assert obstacle.center is not None and obstacle.radius is not None
                height = obstacle.top_z - obstacle.base_z
                cyl = Cylinder(
                    pos=(obstacle.center[0], obstacle.center[1], obstacle.base_z + 0.5 * height),
                    r=obstacle.radius,
                    height=height,
                    axis=(0, 0, 1),
                    c=obstacle.color,
                    alpha=obstacle.alpha,
                )
                actors.append(cyl)
            elif obstacle.polygon is not None:
                verts, faces = build_prism_mesh(obstacle.polygon, obstacle.base_z, obstacle.top_z)
                actors.append(create_mesh_actor(verts, faces, obstacle.color, obstacle.alpha))

    for ramp in floorplan.ramps:
        verts, faces = build_ramp_mesh(ramp)
        actors.append(create_mesh_actor(verts, faces, ramp.color, 0.6))

    for goal in floorplan.goals:
        goal_size = goal.max_corner - goal.min_corner
        goal_actor = Box(
            pos=goal.center,
            length=goal_size[0],
            width=goal_size[1],
            height=goal_size[2],
        )
        goal_actor.c(goal.color).alpha(0.45)
        actors.append(goal_actor)
    return actors


def run_simulator(sim: Simulator) -> None:
    sim.run()
    print(f"Simulation complete: steps={sim.step_count} active={sim.active_count} reached={sim.reached_count}")


def _get_evacuees(sim: Simulator) -> List:
    if sim._env is not None:
        return sim._env.evacuees
    return sim._agents if sim._agents else []


def render_simulator(
    sim: Simulator,
    output_video: Optional[str] = None,
    offscreen: bool = False,
) -> None:
    if offscreen and output_video is None:
        run_simulator(sim)
        return

    try:
        from vedo import Plotter, Sphere, Text2D, Video
    except ImportError as exc:
        raise RuntimeError("vedo is required for rendering. Install project requirements first.") from exc

    interrupted = False

    def signal_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        sim.interrupt()

    import signal
    old_handler = signal.signal(signal.SIGINT, signal_handler)

    plotter = Plotter(title=sim.floorplan.name, size=(1280, 900))
    plotter.show(*build_scene(sim.floorplan), interactive=False, resetcam=True)

    center = 0.5 * (sim.floorplan.bounds_min + sim.floorplan.bounds_max)
    scene_size = sim.floorplan.bounds_max - sim.floorplan.bounds_min
    max_dim = float(np.max(scene_size))
    cam_pos = center + np.array([1.25, 1.2, 1.0], dtype=float) * max_dim
    plotter.camera.SetPosition(cam_pos.tolist())
    plotter.camera.SetFocalPoint(center.tolist())
    plotter.camera.SetViewUp((0, 0, 1))
    plotter.camera.SetParallelProjection(True)
    plotter.camera.SetParallelScale(0.58 * max(scene_size[0], scene_size[1]))
    plotter.reset_clipping_range()
    plotter.render()

    evacuees = _get_evacuees(sim)
    sphere_actors = []
    for evacuee in evacuees:
        actor = Sphere(pos=evacuee.pos, r=evacuee.radius, c=evacuee.color)
        actor.alpha(0.95)
        sphere_actors.append(actor)
    if sphere_actors:
        plotter.add(*sphere_actors)

    harassers = sim._env.harassers if sim._env else []
    harasser_actors = []
    target_actors = []
    for harasser in harassers:
        actor = Sphere(pos=harasser.pos, r=harasser.radius, c=harasser.color)
        actor.alpha(0.95)
        harasser_actors.append(actor)
        target = Sphere(pos=harasser.pos, r=0.3, c="red")
        target.alpha(0.5)
        target_actors.append(target)
    if harasser_actors:
        plotter.add(*harasser_actors)
    if target_actors:
        plotter.add(*target_actors)

    status = Text2D("", pos="top-left", c="white", font="Courier", s=0.8)
    plotter.add(status)

    writer = None
    if output_video:
        writer = Video(output_video, fps=max(1, int(round(1.0 / sim.dt))))

    def on_timer(event):
        nonlocal interrupted
        if sim.is_finished or interrupted:
            plotter.timer_callback("destroy", event.get("timer_event_id"))
            return
        sim.step()
        for actor, evacuee in zip(sphere_actors, _get_evacuees(sim)):
            actor.pos(evacuee.pos)
            actor.alpha(0.35 if evacuee.reached_goal else 0.95)
        for actor, harasser in zip(harasser_actors, sim._env.harassers if sim._env else []):
            actor.pos(harasser.pos)
        for target_actor, harasser in zip(target_actors, sim._env.harassers if sim._env else []):
            target_pos = harasser._target_pos
            target_actor.pos([target_pos[0], target_pos[1], harasser.pos[2] + 0.5])
        status.text(f"step {sim.step_count}/{sim.total_steps}   active: {sim.active_count}   reached: {sim.reached_count}")
        if writer is not None:
            writer.add_frame()
        plotter.render()

    timer_dt = max(16, int(round(sim.dt * 1000)))
    plotter.timer_callback("create", dt=timer_dt)
    plotter.add_callback("TimerEvent", on_timer)
    plotter.interactive()

    if writer is not None:
        writer.close()

    signal.signal(signal.SIGINT, old_handler)

    if interrupted:
        print("\nSimulation interrupted.")
    plotter.close()


def make_example_data(root: pathlib.Path) -> None:
    scenario_dir = root / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    scenario = {
        "name": "basic_mixed_crowd",
        "grid": {"cell_size": 0.35, "padding": 1.0},
        "floors": [
            {
                "name": "ground",
                "z": 0.0,
                "thickness": 0.18,
                "polygon": [[0, 0], [18, 0], [18, 12], [0, 12]],
                "obstacles": [
                    {"type": "box", "polygon": [[5.2, 4.1], [6.7, 4.1], [6.7, 5.8], [5.2, 5.8]], "height": 1.25, "color": "brown4"},
                    {"type": "column", "center": [12.5, 8.3], "radius": 0.45, "height": 3.0, "color": "tan"},
                ],
            },
            {
                "name": "mid",
                "z": 3.0,
                "thickness": 0.18,
                "polygon": [[0, 0], [18, 0], [18, 12], [0, 12]],
                "obstacles": [
                    {"type": "box", "polygon": [[9.0, 6.1], [10.8, 6.1], [10.8, 7.6], [9.0, 7.6]], "height": 1.2, "color": "brown4"},
                ],
            },
            {
                "name": "top",
                "z": 6.0,
                "thickness": 0.18,
                "polygon": [[0, 0], [18, 0], [18, 12], [0, 12]],
                "obstacles": [
                    {"type": "column", "center": [6.4, 8.4], "radius": 0.5, "height": 2.8, "color": "tan"},
                ],
            },
        ],
        "fixtures": [
            {"type": "wall", "polygon": [[0, 0], [18, 0], [18, 0.22], [0, 0.22]], "z": [0, 9.4], "color": "slategray"},
            {"type": "wall", "polygon": [[0, 11.78], [18, 11.78], [18, 12], [0, 12]], "z": [0, 9.4], "color": "slategray"},
            {"type": "wall", "polygon": [[0, 0], [0.22, 0], [0.22, 12], [0, 12]], "z": [0, 9.4], "color": "slategray"},
            {"type": "wall", "polygon": [[17.78, 0], [18, 0], [18, 12], [17.78, 12]], "z": [0, 9.4], "color": "slategray"},
        ],
        "ramps": [
            {
                "name": "top_to_mid",
                "start": [17.0, 8.1, 6.09],
                "end": [12.5, 8.1, 3.09],
                "width": 2.0,
                "thickness": 0.16,
                "from_floor": 2,
                "to_floor": 1,
            },
            {
                "name": "mid_to_ground",
                "start": [5.0, 3.8, 3.09],
                "end": [9.8, 3.8, 0.09],
                "width": 2.0,
                "thickness": 0.16,
                "from_floor": 1,
                "to_floor": 0,
            },
        ],
        "openings": [
            {"min": [12.2, 7.0, 5.72], "max": [17.4, 9.2, 6.45]},
            {"min": [4.6, 2.8, 2.72], "max": [10.2, 4.8, 3.45]},
            {"min": [-0.5, 5.0, -0.2], "max": [0.6, 7.0, 1.2]},
        ],
        "goals": [
            {"name": "main_exit", "min": [-0.15, 5.2, -0.05], "max": [0.7, 6.8, 1.0], "color": "green4"},
        ],
        "simulation": {"dt": 0.05, "steps": 699, "seed": 7},
        "agent_defaults": {
            "radius": 0.22,
            "max_speed": 1.35,
            "nav_gain": 3.0,
            "wall_strength": 5.4,
            "wall_range": 1.0,
            "social_strength": 1.9,
            "social_range": 1.1,
            "ramp_gain": 3.8,
            "goal_gain": 2.4,
            "damping": 0.72,
        },
        "agents": [
            {
                "count": 24,
                "floor": 2,
                "offset_range": {"x": [-0.75, 0.7], "y": [-0.6, 0.65]},
                "color": "royalblue",
                "repulsion": {
                    "strength": 1.9,
                    "range": 1.05,
                    "anisotropy": {
                        "enabled": True,
                        "forward_strength": 1.7,
                        "backward_strength": 0.55,
                        "forward_angle_deg": 80,
                    },
                },
            },
            {
                "count": 18,
                "floor": 1,
                "offset_range": {"x": [-0.7, 0.75], "y": [-0.65, 0.55]},
                "color": "tomato",
                "repulsion": {
                    "strength": 1.6,
                    "range": 0.95,
                    "anisotropy": {"enabled": False},
                },
            },
            {
                "count": 14,
                "floor": 0,
                "offset_range": {"x": [-0.75, 0.75], "y": [-0.55, 0.55]},
                "color": "goldenrod",
                "repulsion": {
                    "strength": 1.4,
                    "range": 0.9,
                    "anisotropy": {
                        "enabled": True,
                        "forward_strength": 1.45,
                        "backward_strength": 0.7,
                        "forward_angle_deg": 70,
                    },
                },
            },
        ],
        "obstacles": [
            {"type": "box", "floor": 1, "offset": [0.18, -0.24], "size": [1.0, 1.3], "height": 1.15, "color": "sienna"},
            {"type": "column", "floor": 0, "offset": [0.3, 0.15], "radius": 0.38, "height": 2.8, "color": "tan"},
        ],
    }

    with open(scenario_dir / "basic.json", "w", encoding="utf-8") as handle:
        json.dump(scenario, handle, indent=2)
    print(f"Wrote basic scenario to {(scenario_dir / 'basic.json').as_posix()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradient-descent evacuation simulation with vedo visualization.")
    parser.add_argument("--scenario", type=str, default=None, help="Scenario file stem or path (in scenarios/ directory)")
    parser.add_argument("--steps", type=int, default=None, help="Override simulation step count")
    parser.add_argument("--dt", type=float, default=None, help="Override time step")
    parser.add_argument("--video", type=str, default=None, help="Output video path")
    parser.add_argument("--offscreen", action="store_true", help="Render without opening a window")
    parser.add_argument("--make-example-data", action="store_true", help="Write the basic scenario JSON file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parent
    if args.make_example_data:
        make_example_data(root)
        return

    if not args.scenario:
        print("Please provide --scenario, or use --make-example-data.")
        return

    floorplan = load_floorplan(args.scenario)
    scenario = load_scenario(args.scenario, floorplan.floors)
    apply_scenario_obstacles(floorplan, scenario)

    if args.steps is not None:
        scenario.simulation.steps = int(args.steps)
    if args.dt is not None:
        scenario.simulation.dt = float(args.dt)

    headless = not os.environ.get("DISPLAY")
    if args.offscreen:
        scenario.simulation.offscreen = True
    if headless:
        if not scenario.simulation.offscreen:
            print("DISPLAY is not set; using offscreen rendering.")
        scenario.simulation.offscreen = True

    random.seed(scenario.simulation.seed)
    np.random.seed(scenario.simulation.seed)

    field = build_navigation_field(floorplan, scenario)
    env = create_environment(scenario, floorplan, field)

    sim = Simulator(floorplan=floorplan, scenario=scenario, env=env)

    render_simulator(
        sim=sim,
        output_video=args.video,
        offscreen=scenario.simulation.offscreen,
    )


if __name__ == "__main__":
    main()

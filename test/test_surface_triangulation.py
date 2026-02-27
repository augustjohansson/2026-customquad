"""
Tests for the surface triangulation quadrature module.

Tests verify that geometry operations on triangulated surfaces produce
correct results for simple shapes (sphere, cube).
"""

import pytest
import sys
import os
import numpy as np

# Import directly from the module file to avoid dolfinx dependency
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "customquad"))
from surface_triangulation import (
    create_sphere_triangulation,
    create_cube_triangulation,
    _ray_intersects_triangle,
    _ray_intersects_segment_2d,
)


class TestSphereTriangulation:
    """Tests for sphere triangulation generation."""

    def test_sphere_vertex_count(self):
        """Verify vertex count increases with refinement."""
        v1, t1, n1 = create_sphere_triangulation(refinement=1)
        v2, t2, n2 = create_sphere_triangulation(refinement=2)
        assert len(v2) > len(v1)

    def test_sphere_on_surface(self):
        """All vertices should lie on the sphere surface."""
        verts, _, _ = create_sphere_triangulation(radius=1.0, refinement=3)
        radii = np.linalg.norm(verts, axis=1)
        np.testing.assert_allclose(radii, 1.0, atol=1e-10)

    def test_sphere_center(self):
        """Sphere with custom center should have vertices at correct radius."""
        center = np.array([1.0, 2.0, 3.0])
        radius = 2.5
        verts, _, _ = create_sphere_triangulation(
            center=center, radius=radius, refinement=2
        )
        radii = np.linalg.norm(verts - center, axis=1)
        np.testing.assert_allclose(radii, radius, atol=1e-10)

    def test_sphere_normals_unit(self):
        """Surface normals should be unit vectors."""
        _, _, normals = create_sphere_triangulation(refinement=2)
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_sphere_normals_outward(self):
        """Surface normals should point outward from center."""
        verts, tris, normals = create_sphere_triangulation(refinement=2)
        centroids = (
            verts[tris[:, 0]] + verts[tris[:, 1]] + verts[tris[:, 2]]
        ) / 3.0
        dots = np.sum(normals * centroids, axis=1)
        assert np.all(dots > 0), "Not all normals point outward"

    def test_sphere_surface_area(self):
        """Surface area should converge to 4*pi for unit sphere."""
        verts, tris, _ = create_sphere_triangulation(refinement=4)
        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]
        areas = 0.5 * np.linalg.norm(
            np.cross(v1 - v0, v2 - v0), axis=1
        )
        total_area = np.sum(areas)
        exact_area = 4 * np.pi
        rel_err = abs(total_area - exact_area) / exact_area
        assert rel_err < 0.01, f"Surface area error {rel_err} > 1%"

    def test_sphere_area_convergence(self):
        """Surface area should converge with refinement."""
        errors = []
        exact = 4 * np.pi
        for ref in [1, 2, 3, 4]:
            verts, tris, _ = create_sphere_triangulation(refinement=ref)
            v0 = verts[tris[:, 0]]
            v1 = verts[tris[:, 1]]
            v2 = verts[tris[:, 2]]
            areas = 0.5 * np.linalg.norm(
                np.cross(v1 - v0, v2 - v0), axis=1
            )
            errors.append(abs(np.sum(areas) - exact) / exact)

        for i in range(1, len(errors)):
            assert errors[i] < errors[i - 1]


class TestCubeTriangulation:
    """Tests for cube triangulation generation."""

    def test_cube_vertex_count(self):
        """Cube should have 8 vertices."""
        verts, _, _ = create_cube_triangulation()
        assert len(verts) == 8

    def test_cube_triangle_count(self):
        """Cube should have 12 triangles (2 per face)."""
        _, tris, _ = create_cube_triangulation()
        assert len(tris) == 12

    def test_cube_surface_area(self):
        """Surface area should be 6 * (2*half_size)^2."""
        half_size = 1.5
        verts, tris, _ = create_cube_triangulation(half_size=half_size)
        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]
        areas = 0.5 * np.linalg.norm(
            np.cross(v1 - v0, v2 - v0), axis=1
        )
        total_area = np.sum(areas)
        exact_area = 6 * (2 * half_size) ** 2
        np.testing.assert_allclose(total_area, exact_area, rtol=1e-10)

    def test_cube_normals_unit(self):
        """Normals should be unit vectors."""
        _, _, normals = create_cube_triangulation()
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)


class TestRayCasting:
    """Tests for ray intersection algorithms."""

    def test_ray_triangle_hit(self):
        """Ray should hit a triangle directly in front."""
        point = np.array([-1.0, 0.25, 0.25])
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([0.0, 1.0, 0.0])
        v2 = np.array([0.0, 0.0, 1.0])
        assert _ray_intersects_triangle(point, v0, v1, v2)

    def test_ray_triangle_miss(self):
        """Ray should miss a triangle not in its path."""
        point = np.array([-1.0, 2.0, 2.0])
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([0.0, 1.0, 0.0])
        v2 = np.array([0.0, 0.0, 1.0])
        assert not _ray_intersects_triangle(point, v0, v1, v2)

    def test_ray_segment_2d_hit(self):
        """Ray should intersect a segment it crosses."""
        point = np.array([-1.0, 0.5])
        v0 = np.array([0.0, 0.0])
        v1 = np.array([0.0, 1.0])
        assert _ray_intersects_segment_2d(point, v0, v1)

    def test_ray_segment_2d_miss(self):
        """Ray should not intersect a segment below it."""
        point = np.array([-1.0, 1.5])
        v0 = np.array([0.0, 0.0])
        v1 = np.array([0.0, 1.0])
        assert not _ray_intersects_segment_2d(point, v0, v1)

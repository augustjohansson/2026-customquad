"""
Example: Custom quadrature from surface triangulation.

Demonstrates how to generate quadrature rules for customquad from a
triangulated surface (sphere and cube), without requiring algoim.

Usage:
    python surface_triangulation_example.py
"""

import numpy as np
import sys
import os

# Add customquad module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "customquad"))
from surface_triangulation import (
    create_sphere_triangulation,
    create_cube_triangulation,
)


def demo_sphere():
    """Demonstrate sphere triangulation and basic geometry checks."""
    print("=" * 60)
    print("Sphere Triangulation")
    print("=" * 60)

    for ref in range(1, 5):
        verts, tris, normals = create_sphere_triangulation(refinement=ref)
        n_verts = len(verts)
        n_tris = len(tris)

        # Compute surface area
        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        total_area = np.sum(areas)

        # Check vertices are on sphere
        radii = np.linalg.norm(verts, axis=1)
        max_radius_err = np.max(np.abs(radii - 1.0))

        # Area error
        exact_area = 4 * np.pi
        area_err = abs(total_area - exact_area) / exact_area

        print(f"  Refinement {ref}: {n_verts:>6} vertices, {n_tris:>6} triangles, "
              f"area err = {area_err:.4e}, max |r-1| = {max_radius_err:.1e}")

    print()


def demo_cube():
    """Demonstrate cube triangulation and basic geometry checks."""
    print("=" * 60)
    print("Cube Triangulation")
    print("=" * 60)

    for half_size in [0.5, 1.0, 2.0]:
        verts, tris, normals = create_cube_triangulation(half_size=half_size)
        n_verts = len(verts)
        n_tris = len(tris)

        # Compute surface area
        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        total_area = np.sum(areas)

        exact_area = 6 * (2 * half_size) ** 2
        area_err = abs(total_area - exact_area) / exact_area

        print(f"  Half-size {half_size:.1f}: {n_verts} vertices, {n_tris} triangles, "
              f"area = {total_area:.6f} (exact = {exact_area:.6f}), "
              f"err = {area_err:.1e}")

    print()


def demo_custom_center():
    """Demonstrate sphere with custom center and radius."""
    print("=" * 60)
    print("Custom Sphere (center=[1,2,3], radius=2.5)")
    print("=" * 60)

    center = np.array([1.0, 2.0, 3.0])
    radius = 2.5
    verts, tris, normals = create_sphere_triangulation(
        center=center, radius=radius, refinement=3
    )

    # Verify
    dists = np.linalg.norm(verts - center, axis=1)
    print(f"  Vertices: {len(verts)}")
    print(f"  Min distance from center: {np.min(dists):.10f}")
    print(f"  Max distance from center: {np.max(dists):.10f}")
    print(f"  Expected radius: {radius}")

    # Check normals point outward
    centroids = (
        verts[tris[:, 0]] + verts[tris[:, 1]] + verts[tris[:, 2]]
    ) / 3.0
    outward = centroids - center
    dots = np.sum(normals * outward, axis=1)
    print(f"  All normals outward: {np.all(dots > 0)}")
    print()


def demo_quadrature_interface():
    """Show how the triangulation data interfaces with customquad."""
    print("=" * 60)
    print("Quadrature Interface Example")
    print("=" * 60)

    verts, tris, normals = create_sphere_triangulation(refinement=3)
    print(f"  Surface: unit sphere, {len(tris)} triangles")
    print(f"  Vertex array shape: {verts.shape} (np.float64)")
    print(f"  Triangle array shape: {tris.shape} (np.int64)")
    print(f"  Normal array shape: {normals.shape} (np.float64)")
    print()
    print("  These numpy arrays can be passed to:")
    print("    customquad.surface_triangulation.classify_cells(mesh, verts, tris, normals)")
    print("    customquad.surface_triangulation.generate_quadrature_from_triangulation(")
    print("        mesh, verts, tris, normals)")
    print()
    print("  The output quadrature rules are in the same format as algoim_utils:")
    print("    cut_cells, uncut_cells, outside_cells,")
    print("    qr_pts, qr_w, qr_pts_bdry, qr_w_bdry, qr_n")
    print()


if __name__ == "__main__":
    demo_sphere()
    demo_cube()
    demo_custom_center()
    demo_quadrature_interface()

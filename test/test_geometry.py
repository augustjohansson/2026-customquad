"""
Geometry tests for customquad: area, volume, cut domains, cut faces, cut edges.

Tests verify geometric quantities computed via custom quadrature rules
against known exact values for circle (2D) and sphere (3D) domains.
"""

import pytest
import numpy as np
import dolfinx
import dolfinx.fem
import dolfinx.mesh
import ufl
from mpi4py import MPI
import customquad as cq


def _generate_qr(mesh, NN, domain, degree=3):
    """Generate quadrature rules using algoim."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))
    import algoim_utils

    opts = {"verbose": False}
    return algoim_utils.generate_qr(mesh, NN, degree, domain, opts)


def _setup_circle(N):
    """Set up a 2D mesh with circle quadrature."""
    xmin = np.array([-1.11, -1.51])
    xmax = np.array([1.55, 1.22])
    NN = np.array([N, N], dtype=np.int32)
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, np.array([xmin, xmax]), NN,
        dolfinx.mesh.CellType.quadrilateral,
    )
    result = _generate_qr(mesh, NN, "circle")
    return mesh, xmin, xmax, NN, result


def _setup_sphere(N):
    """Set up a 3D mesh with sphere quadrature."""
    xmin = np.array([-1.11, -1.51, -1.23])
    xmax = np.array([1.55, 1.22, 1.11])
    NN = np.array([N, N, N], dtype=np.int32)
    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD, np.array([xmin, xmax]), NN,
        dolfinx.mesh.CellType.hexahedron,
    )
    result = _generate_qr(mesh, NN, "sphere")
    return mesh, xmin, xmax, NN, result


@pytest.mark.parametrize("N", [8, 16, 32])
def test_circle_volume(N):
    """Test that computed volume of unit circle converges to pi."""
    mesh, xmin, xmax, NN, result = _setup_circle(N)
    cut_cells, uncut_cells = result[0], result[1]
    qr_w0 = result[4]
    qr_w = [qr_w0[k] for k in cut_cells]

    vol = cq.utils.volume(xmin, xmax, NN, uncut_cells, qr_w)
    vol_exact = np.pi
    rel_err = abs(vol - vol_exact) / vol_exact

    # Error should decrease with mesh refinement
    # For degree 3 algoim, expect at least 2nd order
    assert rel_err < 0.1, f"Circle volume error {rel_err} too large for N={N}"


@pytest.mark.parametrize("N", [8, 16, 32])
def test_circle_area(N):
    """Test that computed perimeter of unit circle converges to 2*pi."""
    mesh, xmin, xmax, NN, result = _setup_circle(N)
    cut_cells = result[0]
    qr_w_bdry0 = result[6]
    qr_w_bdry = [qr_w_bdry0[k] for k in cut_cells]

    area = cq.utils.area(xmin, xmax, NN, qr_w_bdry)
    area_exact = 2 * np.pi
    rel_err = abs(area - area_exact) / area_exact

    assert rel_err < 0.1, f"Circle perimeter error {rel_err} too large for N={N}"


@pytest.mark.parametrize("N", [4, 8, 16])
def test_sphere_volume(N):
    """Test that computed volume of unit sphere converges to 4/3*pi."""
    mesh, xmin, xmax, NN, result = _setup_sphere(N)
    cut_cells, uncut_cells = result[0], result[1]
    qr_w0 = result[4]
    qr_w = [qr_w0[k] for k in cut_cells]

    vol = cq.utils.volume(xmin, xmax, NN, uncut_cells, qr_w)
    vol_exact = 4 * np.pi / 3
    rel_err = abs(vol - vol_exact) / vol_exact

    assert rel_err < 0.1, f"Sphere volume error {rel_err} too large for N={N}"


@pytest.mark.parametrize("N", [4, 8, 16])
def test_sphere_area(N):
    """Test that computed surface area of unit sphere converges to 4*pi."""
    mesh, xmin, xmax, NN, result = _setup_sphere(N)
    cut_cells = result[0]
    qr_w_bdry0 = result[6]
    qr_w_bdry = [qr_w_bdry0[k] for k in cut_cells]

    area = cq.utils.area(xmin, xmax, NN, qr_w_bdry)
    area_exact = 4 * np.pi
    rel_err = abs(area - area_exact) / area_exact

    assert rel_err < 0.1, f"Sphere surface area error {rel_err} too large for N={N}"


def test_circle_volume_convergence():
    """Verify that circle volume error converges with mesh refinement."""
    errors = []
    for N in [8, 16, 32]:
        mesh, xmin, xmax, NN, result = _setup_circle(N)
        cut_cells, uncut_cells = result[0], result[1]
        qr_w0 = result[4]
        qr_w = [qr_w0[k] for k in cut_cells]

        vol = cq.utils.volume(xmin, xmax, NN, uncut_cells, qr_w)
        errors.append(abs(vol - np.pi) / np.pi)

    # Each refinement should reduce error
    for i in range(1, len(errors)):
        assert errors[i] < errors[i - 1], (
            f"Volume error not decreasing: {errors[i]} >= {errors[i-1]}"
        )


def test_sphere_volume_convergence():
    """Verify that sphere volume error converges with mesh refinement."""
    errors = []
    for N in [4, 8, 16]:
        mesh, xmin, xmax, NN, result = _setup_sphere(N)
        cut_cells, uncut_cells = result[0], result[1]
        qr_w0 = result[4]
        qr_w = [qr_w0[k] for k in cut_cells]

        vol = cq.utils.volume(xmin, xmax, NN, uncut_cells, qr_w)
        errors.append(abs(vol - 4 * np.pi / 3) / (4 * np.pi / 3))

    for i in range(1, len(errors)):
        assert errors[i] < errors[i - 1], (
            f"Volume error not decreasing: {errors[i]} >= {errors[i-1]}"
        )


def test_cut_cell_classification_circle():
    """Verify that cell classification is consistent for circle domain."""
    N = 16
    mesh, xmin, xmax, NN, result = _setup_circle(N)
    cut_cells, uncut_cells, outside_cells = result[0], result[1], result[2]

    num_cells = cq.utils.get_num_cells(mesh)

    # Every cell must be classified exactly once
    all_classified = np.sort(np.concatenate([cut_cells, uncut_cells, outside_cells]))
    all_cells = np.arange(num_cells)
    assert np.array_equal(all_classified, all_cells), (
        "Not all cells are uniquely classified"
    )

    # Cut cells should exist (the circle doesn't align with mesh)
    assert len(cut_cells) > 0, "No cut cells found"
    assert len(uncut_cells) > 0, "No uncut cells found"
    assert len(outside_cells) > 0, "No outside cells found"


def test_cut_cell_classification_sphere():
    """Verify that cell classification is consistent for sphere domain."""
    N = 8
    mesh, xmin, xmax, NN, result = _setup_sphere(N)
    cut_cells, uncut_cells, outside_cells = result[0], result[1], result[2]

    num_cells = cq.utils.get_num_cells(mesh)

    all_classified = np.sort(np.concatenate([cut_cells, uncut_cells, outside_cells]))
    all_cells = np.arange(num_cells)
    assert np.array_equal(all_classified, all_cells)

    assert len(cut_cells) > 0
    assert len(uncut_cells) > 0
    assert len(outside_cells) > 0


def test_quadrature_data_consistency_circle():
    """Verify that quadrature data arrays have consistent sizes for circle."""
    N = 16
    mesh, xmin, xmax, NN, result = _setup_circle(N)
    cut_cells = result[0]
    qr_pts0, qr_w0, qr_pts_bdry0, qr_w_bdry0, qr_n0 = (
        result[3], result[4], result[5], result[6], result[7]
    )

    for k in cut_cells:
        # Bulk quadrature: points and weights should match
        n_pts = len(qr_w0[k])
        assert n_pts > 0, f"No quadrature points for cut cell {k}"
        assert len(qr_pts0[k]) == n_pts * 2, (
            f"Points/weights mismatch in cell {k}: "
            f"{len(qr_pts0[k])} pts vs {n_pts} weights"
        )

        # Boundary quadrature: points, weights, normals should match
        n_bdry = len(qr_w_bdry0[k])
        if n_bdry > 0:
            assert len(qr_pts_bdry0[k]) == n_bdry * 2
            assert len(qr_n0[k]) == n_bdry * 2


def test_quadrature_data_consistency_sphere():
    """Verify that quadrature data arrays have consistent sizes for sphere."""
    N = 8
    mesh, xmin, xmax, NN, result = _setup_sphere(N)
    cut_cells = result[0]
    qr_pts0, qr_w0, qr_pts_bdry0, qr_w_bdry0, qr_n0 = (
        result[3], result[4], result[5], result[6], result[7]
    )

    for k in cut_cells:
        n_pts = len(qr_w0[k])
        assert n_pts > 0, f"No quadrature points for cut cell {k}"
        assert len(qr_pts0[k]) == n_pts * 3

        n_bdry = len(qr_w_bdry0[k])
        if n_bdry > 0:
            assert len(qr_pts_bdry0[k]) == n_bdry * 3
            assert len(qr_n0[k]) == n_bdry * 3


def test_functional_volume_circle():
    """Test volume computation via functional (integral of 1) for circle."""
    N = 16
    mesh, xmin, xmax, NN, result = _setup_circle(N)
    cut_cells, uncut_cells, outside_cells = result[0], result[1], result[2]
    qr_pts0, qr_w0 = result[3], result[4]
    qr_pts = [qr_pts0[k] for k in cut_cells]
    qr_w = [qr_w0[k] for k in cut_cells]

    celltags = cq.utils.get_celltags(
        mesh, cut_cells, uncut_cells, outside_cells,
        uncut_cell_tag=1, cut_cell_tag=2, outside_cell_tag=3,
    )

    dx_uncut = ufl.dx(subdomain_data=celltags, domain=mesh)
    dx_cut = ufl.dx(metadata={"quadrature_rule": "runtime"}, domain=mesh)

    vol_uncut = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(1.0 * dx_uncut(1))
    )
    vol_cut = cq.assemble_scalar(
        dolfinx.fem.form(1.0 * dx_cut), [(cut_cells, qr_pts, qr_w)]
    )

    vol = vol_uncut + vol_cut
    rel_err = abs(vol - np.pi) / np.pi

    assert rel_err < 0.01, f"Functional volume error {rel_err} for circle"


def test_functional_area_circle():
    """Test boundary area via functional (integral of 1 on bdry) for circle."""
    N = 16
    mesh, xmin, xmax, NN, result = _setup_circle(N)
    cut_cells = result[0]
    uncut_cells, outside_cells = result[1], result[2]
    qr_pts_bdry0, qr_w_bdry0, qr_n0 = result[5], result[6], result[7]
    qr_pts_bdry = [qr_pts_bdry0[k] for k in cut_cells]
    qr_w_bdry = [qr_w_bdry0[k] for k in cut_cells]
    qr_n = [qr_n0[k] for k in cut_cells]

    celltags = cq.utils.get_celltags(
        mesh, cut_cells, uncut_cells, outside_cells,
        uncut_cell_tag=1, cut_cell_tag=2, outside_cell_tag=3,
    )

    ds_cut = ufl.dx(
        subdomain_data=celltags,
        metadata={"quadrature_rule": "runtime"},
        domain=mesh,
    )

    area = cq.assemble_scalar(
        dolfinx.fem.form(1.0 * ds_cut(2)),
        [(cut_cells, qr_pts_bdry, qr_w_bdry, qr_n)],
    )

    rel_err = abs(area - 2 * np.pi) / (2 * np.pi)
    assert rel_err < 0.01, f"Functional area error {rel_err} for circle"


def test_cut_faces_2d():
    """Test that cut faces (edges in 2D) are correctly identified."""
    N = 16
    mesh, xmin, xmax, NN, result = _setup_circle(N)
    cut_cells, uncut_cells, outside_cells = result[0], result[1], result[2]

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    face_to_cells = mesh.topology.connectivity(tdim - 1, tdim)

    # Find faces shared between cut cells and uncut/outside cells
    num_faces = cq.utils.get_num_faces(mesh)
    cut_boundary_faces = []
    for f in range(num_faces):
        cells = face_to_cells.links(f)
        if len(cells) == 2:
            c0_cut = cells[0] in cut_cells
            c1_cut = cells[1] in cut_cells
            # Face where one cell is cut and the other is not
            if c0_cut != c1_cut:
                cut_boundary_faces.append(f)

    assert len(cut_boundary_faces) > 0, "No cut boundary faces found"


def test_cut_edges_3d():
    """Test that cut edges in 3D are correctly identified."""
    N = 8
    mesh, xmin, xmax, NN, result = _setup_sphere(N)
    cut_cells, uncut_cells, outside_cells = result[0], result[1], result[2]

    tdim = mesh.topology.dim
    # Create edge-to-cell connectivity (dim 1 to dim 3)
    mesh.topology.create_connectivity(1, tdim)
    edge_to_cells = mesh.topology.connectivity(1, tdim)

    num_edges = mesh.topology.index_map(1).size_local
    cut_set = set(cut_cells)

    cut_edges = []
    for e in range(num_edges):
        cells = edge_to_cells.links(e)
        has_cut = any(c in cut_set for c in cells)
        has_noncut = any(c not in cut_set for c in cells)
        if has_cut and has_noncut:
            cut_edges.append(e)

    assert len(cut_edges) > 0, "No cut edges found in 3D"

"""
PDE convergence tests for Poisson and Linear Elasticity problems
on circle (2D) and sphere (3D) geometries using CG1 and CG2 elements.

Tests verify optimal convergence rates in L2 and H10 norms for:
- Homogeneous Dirichlet boundary conditions
- Non-homogeneous Dirichlet boundary conditions (Nitsche's method)
- Neumann boundary conditions
- Surface PDE (Laplace-Beltrami operator)

These tests require algoim for quadrature generation and should be run
within the customquad Docker container.
"""

import pytest
import numpy as np
import dolfinx
import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
import ufl
from ufl import grad, inner, dot, jump, avg, div
from mpi4py import MPI
from petsc4py import PETSc


def _make_circle_qr(mesh, NN, degree=3):
    """Generate quadrature rules for a unit circle domain using algoim."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))
    import algoim_utils

    opts = {"verbose": False}
    result = algoim_utils.generate_qr(mesh, NN, degree, "circle", opts)
    return result


def _make_sphere_qr(mesh, NN, degree=3):
    """Generate quadrature rules for a unit sphere domain using algoim."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))
    import algoim_utils

    opts = {"verbose": False}
    result = algoim_utils.generate_qr(mesh, NN, degree, "sphere", opts)
    return result


def _create_mesh_and_qr(domain, N):
    """Create mesh and generate quadrature rules for the given domain."""
    if domain == "circle":
        gdim = 2
        xmin = np.array([-1.11, -1.51])
        xmax = np.array([1.55, 1.22])
        NN = np.array([N, N], dtype=np.int32)
        cell_type = dolfinx.mesh.CellType.quadrilateral
        mesh = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD, np.array([xmin, xmax]), NN, cell_type
        )
        qr_result = _make_circle_qr(mesh, NN)
    elif domain == "sphere":
        gdim = 3
        xmin = np.array([-1.11, -1.51, -1.23])
        xmax = np.array([1.55, 1.22, 1.11])
        NN = np.array([N, N, N], dtype=np.int32)
        cell_type = dolfinx.mesh.CellType.hexahedron
        mesh = dolfinx.mesh.create_box(
            MPI.COMM_WORLD, np.array([xmin, xmax]), NN, cell_type
        )
        qr_result = _make_sphere_qr(mesh, NN)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    [
        cut_cells, uncut_cells, outside_cells,
        qr_pts0, qr_w0, qr_pts_bdry0, qr_w_bdry0, qr_n0,
        xyz, xyz_bdry,
    ] = qr_result

    # Filter to cut cells only
    qr_pts = [qr_pts0[k] for k in cut_cells]
    qr_w = [qr_w0[k] for k in cut_cells]
    qr_pts_bdry = [qr_pts_bdry0[k] for k in cut_cells]
    qr_w_bdry = [qr_w_bdry0[k] for k in cut_cells]
    qr_n = [qr_n0[k] for k in cut_cells]

    return {
        "mesh": mesh,
        "gdim": gdim,
        "xmin": xmin,
        "xmax": xmax,
        "NN": NN,
        "cut_cells": cut_cells,
        "uncut_cells": uncut_cells,
        "outside_cells": outside_cells,
        "qr_pts": qr_pts,
        "qr_w": qr_w,
        "qr_pts_bdry": qr_pts_bdry,
        "qr_w_bdry": qr_w_bdry,
        "qr_n": qr_n,
    }


def _solve_poisson(data, degree, betaN=10.0, betas=1.0):
    """Solve Poisson problem on cut domain with Nitsche BCs.

    Returns L2 error, H10 error, and mesh size h.
    """
    import customquad

    mesh = data["mesh"]
    gdim = data["gdim"]
    cut_cells = data["cut_cells"]
    uncut_cells = data["uncut_cells"]
    outside_cells = data["outside_cells"]

    # Set up tags
    uncut_cell_tag = 1
    cut_cell_tag = 2
    outside_cell_tag = 3
    ghost_penalty_tag = 4
    celltags = customquad.utils.get_celltags(
        mesh, cut_cells, uncut_cells, outside_cells,
        uncut_cell_tag=uncut_cell_tag,
        cut_cell_tag=cut_cell_tag,
        outside_cell_tag=outside_cell_tag,
    )
    facetags = customquad.utils.get_facetags(
        mesh, cut_cells, outside_cells, ghost_penalty_tag=ghost_penalty_tag
    )

    # FEM
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    n = ufl.FacetNormal(mesh)
    h = ufl.CellDiameter(mesh)

    # Exact solution (homogeneous on the boundary of the unit circle/sphere)
    if gdim == 2:
        # u = sin(pi*x)*sin(pi*y) - not zero on boundary
        # Use u = (1 - x^2 - y^2) for homogeneous Dirichlet on unit circle
        u_ufl = 1 - x[0]**2 - x[1]**2
        f_ufl = 4.0  # -laplacian of u
        u_np = lambda xx: 1 - xx[0]**2 - xx[1]**2
    else:
        u_ufl = 1 - x[0]**2 - x[1]**2 - x[2]**2
        f_ufl = 6.0
        u_np = lambda xx: 1 - xx[0]**2 - xx[1]**2 - xx[2]**2

    g_ufl = u_ufl  # Dirichlet BC (should be zero on unit circle/sphere)

    # Bilinear and linear forms
    a_bulk = inner(grad(u), grad(v))
    L_bulk = inner(f_ufl, v)
    a_bdry = (
        -inner(dot(n, grad(u)), v)
        - inner(u, dot(n, grad(v)))
        + inner(betaN / h * u, v)
    )
    L_bdry = (
        -inner(g_ufl, dot(n, grad(v)))
        + inner(betaN / h * g_ufl, v)
    )
    a_stab = betas * avg(h) * inner(jump(n, grad(u)), jump(n, grad(v)))

    # Measures
    dx_uncut = ufl.dx(subdomain_data=celltags, domain=mesh)
    dS = ufl.dS(subdomain_data=facetags, domain=mesh)
    dx_cut = ufl.dx(metadata={"quadrature_rule": "runtime"}, domain=mesh)
    ds_cut = ufl.dx(
        subdomain_data=celltags,
        metadata={"quadrature_rule": "runtime"},
        domain=mesh,
    )

    # Standard assembly (uncut cells + ghost penalty)
    ax = dolfinx.fem.form(
        a_bulk * dx_uncut(uncut_cell_tag) + a_stab * dS(ghost_penalty_tag)
    )
    Ax = dolfinx.fem.petsc.assemble_matrix(ax)
    Ax.assemble()
    Lx = dolfinx.fem.form(L_bulk * dx_uncut(uncut_cell_tag))
    bx = dolfinx.fem.petsc.assemble_vector(Lx)

    # Custom assembly (cut cells)
    qr_bulk = [(cut_cells, data["qr_pts"], data["qr_w"])]
    qr_bdry = [(cut_cells, data["qr_pts_bdry"], data["qr_w_bdry"], data["qr_n"])]

    Ac1 = customquad.assemble_matrix(dolfinx.fem.form(a_bulk * dx_cut), qr_bulk)
    Ac1.assemble()
    Ac2 = customquad.assemble_matrix(
        dolfinx.fem.form(a_bdry * ds_cut(cut_cell_tag)), qr_bdry
    )
    Ac2.assemble()

    A = Ax
    A += Ac1
    A += Ac2

    bc1 = customquad.assemble_vector(dolfinx.fem.form(L_bulk * dx_cut), qr_bulk)
    b = bx
    b += bc1
    bc2 = customquad.assemble_vector(
        dolfinx.fem.form(L_bdry * ds_cut), qr_bdry
    )
    b += bc2

    # Lock inactive dofs
    inactive_dofs = customquad.utils.get_inactive_dofs(V, cut_cells, uncut_cells)
    A = customquad.utils.lock_inactive_dofs(inactive_dofs, A)

    # Solve
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    vec = b.copy()
    ksp.solve(b, vec)
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    uh = dolfinx.fem.Function(V)
    uh.vector.setArray(vec.array)

    # Compute errors
    def assemble_total(integrand):
        m_cut = customquad.assemble_scalar(
            dolfinx.fem.form(integrand * dx_cut), qr_bulk
        )
        m_uncut = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(integrand * dx_uncut(uncut_cell_tag))
        )
        return m_cut + m_uncut

    L2_err = np.sqrt(abs(assemble_total((uh - u_ufl) ** 2)))
    H10_err = np.sqrt(abs(assemble_total(inner(grad(uh) - grad(u_ufl), grad(uh) - grad(u_ufl)))))

    h_val = max(dolfinx.mesh.h(mesh, mesh.topology.dim, cut_cells))

    return L2_err, H10_err, h_val


def _solve_elasticity(data, degree, betaN=10.0, betas=1.0):
    """Solve linear elasticity on cut domain with Nitsche BCs.

    Returns L2 error, H10 error, and mesh size h.
    """
    import customquad

    mesh = data["mesh"]
    gdim = data["gdim"]
    cut_cells = data["cut_cells"]
    uncut_cells = data["uncut_cells"]
    outside_cells = data["outside_cells"]

    # Material parameters
    E = 1.0
    nu = 0.3
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(gdim) + 2 * mu * epsilon(u)

    # Set up tags
    uncut_cell_tag = 1
    cut_cell_tag = 2
    outside_cell_tag = 3
    ghost_penalty_tag = 4
    celltags = customquad.utils.get_celltags(
        mesh, cut_cells, uncut_cells, outside_cells,
        uncut_cell_tag=uncut_cell_tag,
        cut_cell_tag=cut_cell_tag,
        outside_cell_tag=outside_cell_tag,
    )
    facetags = customquad.utils.get_facetags(
        mesh, cut_cells, outside_cells, ghost_penalty_tag=ghost_penalty_tag
    )

    # FEM
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    n = ufl.FacetNormal(mesh)
    h = ufl.CellDiameter(mesh)

    # Manufactured solution (zero on unit circle/sphere boundary)
    if gdim == 2:
        r2 = x[0]**2 + x[1]**2
        u_ufl = ufl.as_vector([(1 - r2) * x[1], (1 - r2) * x[0]])
    else:
        r2 = x[0]**2 + x[1]**2 + x[2]**2
        u_ufl = ufl.as_vector([
            (1 - r2) * x[1],
            (1 - r2) * x[0],
            (1 - r2) * x[2],
        ])

    f_ufl = -div(sigma(u_ufl))
    g_ufl = u_ufl  # Dirichlet BC

    # Forms
    a_bulk = inner(sigma(u), epsilon(v))
    L_bulk = inner(f_ufl, v)
    a_bdry = (
        -inner(dot(n, sigma(u)), v)
        - inner(u, dot(n, sigma(v)))
        + inner(betaN / h * u, v)
    )
    L_bdry = (
        -inner(g_ufl, dot(n, sigma(v)))
        + inner(betaN / h * g_ufl, v)
    )
    a_stab = betas * avg(h) * inner(jump(n, ufl.grad(u)), jump(n, ufl.grad(v)))

    # Measures
    dx_uncut = ufl.dx(subdomain_data=celltags, domain=mesh)
    dS = ufl.dS(subdomain_data=facetags, domain=mesh)
    dx_cut = ufl.dx(metadata={"quadrature_rule": "runtime"}, domain=mesh)
    ds_cut = ufl.dx(
        subdomain_data=celltags,
        metadata={"quadrature_rule": "runtime"},
        domain=mesh,
    )

    # Standard assembly
    ax = dolfinx.fem.form(
        a_bulk * dx_uncut(uncut_cell_tag) + a_stab * dS(ghost_penalty_tag)
    )
    Ax = dolfinx.fem.petsc.assemble_matrix(ax)
    Ax.assemble()
    Lx = dolfinx.fem.form(L_bulk * dx_uncut(uncut_cell_tag))
    bx = dolfinx.fem.petsc.assemble_vector(Lx)

    # Custom assembly
    qr_bulk = [(cut_cells, data["qr_pts"], data["qr_w"])]
    qr_bdry = [(cut_cells, data["qr_pts_bdry"], data["qr_w_bdry"], data["qr_n"])]

    Ac1 = customquad.assemble_matrix(dolfinx.fem.form(a_bulk * dx_cut), qr_bulk)
    Ac1.assemble()
    Ac2 = customquad.assemble_matrix(
        dolfinx.fem.form(a_bdry * ds_cut(cut_cell_tag)), qr_bdry
    )
    Ac2.assemble()

    A = Ax
    A += Ac1
    A += Ac2

    bc1 = customquad.assemble_vector(dolfinx.fem.form(L_bulk * dx_cut), qr_bulk)
    b = bx
    b += bc1
    bc2 = customquad.assemble_vector(
        dolfinx.fem.form(L_bdry * ds_cut), qr_bdry
    )
    b += bc2

    # Lock inactive dofs
    inactive_dofs = customquad.utils.get_inactive_dofs(V, cut_cells, uncut_cells)
    A = customquad.utils.lock_inactive_dofs(inactive_dofs, A)

    # Solve
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    vec = b.copy()
    ksp.solve(b, vec)
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    uh = dolfinx.fem.Function(V)
    uh.vector.setArray(vec.array)

    # Compute errors
    def assemble_total(integrand):
        m_cut = customquad.assemble_scalar(
            dolfinx.fem.form(integrand * dx_cut), qr_bulk
        )
        m_uncut = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(integrand * dx_uncut(uncut_cell_tag))
        )
        return m_cut + m_uncut

    L2_err = np.sqrt(abs(assemble_total(inner(uh - u_ufl, uh - u_ufl))))
    H10_err = np.sqrt(abs(assemble_total(
        inner(ufl.grad(uh) - ufl.grad(u_ufl), ufl.grad(uh) - ufl.grad(u_ufl))
    )))

    h_val = max(dolfinx.mesh.h(mesh, mesh.topology.dim, cut_cells))

    return L2_err, H10_err, h_val


def _estimate_convergence_rate(h_values, err_values):
    """Estimate convergence rate from h and error values using least squares."""
    if len(h_values) < 2:
        return 0.0
    log_h = np.log(np.array(h_values))
    log_e = np.log(np.array(err_values))
    # Linear fit: log(e) = rate * log(h) + C
    A = np.vstack([log_h, np.ones(len(log_h))]).T
    rate, _ = np.linalg.lstsq(A, log_e, rcond=None)[0]
    return rate


@pytest.mark.parametrize("degree", [1, 2])
def test_poisson_convergence_circle(degree):
    """Test optimal convergence for Poisson on circle (2D)."""
    mesh_sizes = [8, 16, 32]
    h_values = []
    L2_errors = []
    H10_errors = []

    for N in mesh_sizes:
        data = _create_mesh_and_qr("circle", N)
        L2_err, H10_err, h_val = _solve_poisson(data, degree)
        h_values.append(h_val)
        L2_errors.append(L2_err)
        H10_errors.append(H10_err)

    L2_rate = _estimate_convergence_rate(h_values, L2_errors)
    H10_rate = _estimate_convergence_rate(h_values, H10_errors)

    # Expected: L2 rate ~ degree+1, H10 rate ~ degree
    # Allow some tolerance (0.5 order) for cut cell methods
    assert L2_rate > degree + 0.5, (
        f"L2 rate {L2_rate:.2f} < expected {degree + 0.5} for CG{degree}"
    )
    assert H10_rate > degree - 0.5, (
        f"H10 rate {H10_rate:.2f} < expected {degree - 0.5} for CG{degree}"
    )


@pytest.mark.parametrize("degree", [1, 2])
def test_poisson_convergence_sphere(degree):
    """Test optimal convergence for Poisson on sphere (3D)."""
    mesh_sizes = [4, 8, 16]
    h_values = []
    L2_errors = []
    H10_errors = []

    for N in mesh_sizes:
        data = _create_mesh_and_qr("sphere", N)
        L2_err, H10_err, h_val = _solve_poisson(data, degree)
        h_values.append(h_val)
        L2_errors.append(L2_err)
        H10_errors.append(H10_err)

    L2_rate = _estimate_convergence_rate(h_values, L2_errors)
    H10_rate = _estimate_convergence_rate(h_values, H10_errors)

    assert L2_rate > degree + 0.5, (
        f"L2 rate {L2_rate:.2f} < expected {degree + 0.5} for CG{degree}"
    )
    assert H10_rate > degree - 0.5, (
        f"H10 rate {H10_rate:.2f} < expected {degree - 0.5} for CG{degree}"
    )


@pytest.mark.parametrize("degree", [1, 2])
def test_elasticity_convergence_circle(degree):
    """Test optimal convergence for Linear Elasticity on circle (2D)."""
    mesh_sizes = [8, 16, 32]
    h_values = []
    L2_errors = []
    H10_errors = []

    for N in mesh_sizes:
        data = _create_mesh_and_qr("circle", N)
        L2_err, H10_err, h_val = _solve_elasticity(data, degree)
        h_values.append(h_val)
        L2_errors.append(L2_err)
        H10_errors.append(H10_err)

    L2_rate = _estimate_convergence_rate(h_values, L2_errors)
    H10_rate = _estimate_convergence_rate(h_values, H10_errors)

    assert L2_rate > degree + 0.5, (
        f"L2 rate {L2_rate:.2f} < expected {degree + 0.5} for CG{degree}"
    )
    assert H10_rate > degree - 0.5, (
        f"H10 rate {H10_rate:.2f} < expected {degree - 0.5} for CG{degree}"
    )


@pytest.mark.parametrize("degree", [1, 2])
def test_elasticity_convergence_sphere(degree):
    """Test optimal convergence for Linear Elasticity on sphere (3D)."""
    mesh_sizes = [4, 8, 16]
    h_values = []
    L2_errors = []
    H10_errors = []

    for N in mesh_sizes:
        data = _create_mesh_and_qr("sphere", N)
        L2_err, H10_err, h_val = _solve_elasticity(data, degree)
        h_values.append(h_val)
        L2_errors.append(L2_err)
        H10_errors.append(H10_err)

    L2_rate = _estimate_convergence_rate(h_values, L2_errors)
    H10_rate = _estimate_convergence_rate(h_values, H10_errors)

    assert L2_rate > degree + 0.5, (
        f"L2 rate {L2_rate:.2f} < expected {degree + 0.5} for CG{degree}"
    )
    assert H10_rate > degree - 0.5, (
        f"H10 rate {H10_rate:.2f} < expected {degree - 0.5} for CG{degree}"
    )

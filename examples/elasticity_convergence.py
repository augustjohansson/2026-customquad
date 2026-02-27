"""
Example: Linear elasticity convergence test on circle/sphere domain.

Solves the linear elasticity equations on a circle (2D) or sphere (3D)
domain using CutFEM with Nitsche's method for boundary conditions.
Shows optimal convergence in L2 and H10 norms.

Usage:
    python elasticity_convergence.py --domain circle --degree 1
    python elasticity_convergence.py --domain sphere --degree 2
"""

import argparse
import sys
import os
import numpy as np
import dolfinx
import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
import ufl
from ufl import inner, grad, dot, div, jump, avg, sym, tr
from mpi4py import MPI
from petsc4py import PETSc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))
import algoim_utils
import customquad


def solve_elasticity(domain, N, degree, betaN=10.0, betas=1.0, verbose=False):
    """Solve linear elasticity on a cut domain."""

    E = 1.0
    nu = 0.3
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

    def epsilon(u):
        return sym(grad(u))

    def sigma(u):
        return lmbda * tr(epsilon(u)) * ufl.Identity(gdim) + 2 * mu * epsilon(u)

    if domain == "circle":
        gdim = 2
        xmin = np.array([-1.11, -1.51])
        xmax = np.array([1.55, 1.22])
        NN = np.array([N, N], dtype=np.int32)
        cell_type = dolfinx.mesh.CellType.quadrilateral
        mesh = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD, np.array([xmin, xmax]), NN, cell_type
        )
    elif domain == "sphere":
        gdim = 3
        xmin = np.array([-1.11, -1.51, -1.23])
        xmax = np.array([1.55, 1.22, 1.11])
        NN = np.array([N, N, N], dtype=np.int32)
        cell_type = dolfinx.mesh.CellType.hexahedron
        mesh = dolfinx.mesh.create_box(
            MPI.COMM_WORLD, np.array([xmin, xmax]), NN, cell_type
        )
    else:
        raise ValueError(f"Unknown domain: {domain}")

    # Generate quadrature rules
    opts = {"verbose": verbose}
    result = algoim_utils.generate_qr(mesh, NN, degree, domain, opts)
    [
        cut_cells, uncut_cells, outside_cells,
        qr_pts0, qr_w0, qr_pts_bdry0, qr_w_bdry0, qr_n0,
        xyz, xyz_bdry,
    ] = result

    qr_pts = [qr_pts0[k] for k in cut_cells]
    qr_w = [qr_w0[k] for k in cut_cells]
    qr_pts_bdry = [qr_pts_bdry0[k] for k in cut_cells]
    qr_w_bdry = [qr_w_bdry0[k] for k in cut_cells]
    qr_n = [qr_n0[k] for k in cut_cells]

    # Tags
    celltags = customquad.utils.get_celltags(
        mesh, cut_cells, uncut_cells, outside_cells,
        uncut_cell_tag=1, cut_cell_tag=2, outside_cell_tag=3,
    )
    facetags = customquad.utils.get_facetags(
        mesh, cut_cells, outside_cells, ghost_penalty_tag=4
    )

    # FEM
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    n = ufl.FacetNormal(mesh)
    h = ufl.CellDiameter(mesh)

    # Manufactured solution
    if gdim == 2:
        r2 = x[0]**2 + x[1]**2
        u_ufl = ufl.as_vector([(1 - r2) * x[1], (1 - r2) * x[0]])
    else:
        r2 = x[0]**2 + x[1]**2 + x[2]**2
        u_ufl = ufl.as_vector([
            (1 - r2) * x[1], (1 - r2) * x[0], (1 - r2) * x[2]
        ])

    f_ufl = -div(sigma(u_ufl))
    g_ufl = u_ufl

    # Forms
    a_bulk = inner(sigma(u), epsilon(v))
    L_bulk = inner(f_ufl, v)
    a_bdry = (
        -inner(dot(n, sigma(u)), v)
        - inner(u, dot(n, sigma(v)))
        + inner(betaN / h * u, v)
    )
    L_bdry = -inner(g_ufl, dot(n, sigma(v))) + inner(betaN / h * g_ufl, v)
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

    # Assemble
    ax = dolfinx.fem.form(a_bulk * dx_uncut(1) + a_stab * dS(4))
    Ax = dolfinx.fem.petsc.assemble_matrix(ax)
    Ax.assemble()
    Lx = dolfinx.fem.form(L_bulk * dx_uncut(1))
    bx = dolfinx.fem.petsc.assemble_vector(Lx)

    qr_bulk = [(cut_cells, qr_pts, qr_w)]
    qr_bdry_data = [(cut_cells, qr_pts_bdry, qr_w_bdry, qr_n)]

    Ac1 = customquad.assemble_matrix(dolfinx.fem.form(a_bulk * dx_cut), qr_bulk)
    Ac1.assemble()
    Ac2 = customquad.assemble_matrix(
        dolfinx.fem.form(a_bdry * ds_cut(2)), qr_bdry_data
    )
    Ac2.assemble()

    A = Ax
    A += Ac1
    A += Ac2

    bc1 = customquad.assemble_vector(dolfinx.fem.form(L_bulk * dx_cut), qr_bulk)
    b_vec = bx
    b_vec += bc1
    bc2 = customquad.assemble_vector(
        dolfinx.fem.form(L_bdry * ds_cut), qr_bdry_data
    )
    b_vec += bc2

    inactive_dofs = customquad.utils.get_inactive_dofs(V, cut_cells, uncut_cells)
    A = customquad.utils.lock_inactive_dofs(inactive_dofs, A)

    # Solve
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    vec = b_vec.copy()
    ksp.solve(b_vec, vec)
    vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    uh = dolfinx.fem.Function(V)
    uh.vector.setArray(vec.array)

    # Errors
    def assemble_total(integrand):
        m_cut = customquad.assemble_scalar(
            dolfinx.fem.form(integrand * dx_cut), qr_bulk
        )
        m_uncut = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(integrand * dx_uncut(1))
        )
        return m_cut + m_uncut

    L2_err = np.sqrt(abs(assemble_total(inner(uh - u_ufl, uh - u_ufl))))
    H10_err = np.sqrt(abs(assemble_total(
        inner(grad(uh) - grad(u_ufl), grad(uh) - grad(u_ufl))
    )))

    h_val = max(dolfinx.mesh.h(mesh, mesh.topology.dim, cut_cells))

    return h_val, L2_err, H10_err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elasticity convergence test")
    parser.add_argument("--domain", type=str, default="circle",
                        choices=["circle", "sphere"])
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.domain == "circle":
        mesh_sizes = [8, 16, 32, 64]
    else:
        mesh_sizes = [4, 8, 16]

    print(f"Elasticity convergence on {args.domain} with CG{args.degree}")
    print(f"{'N':>5} {'h':>10} {'L2_err':>12} {'H10_err':>12}")
    print("-" * 45)

    results = []
    for N in mesh_sizes:
        h, L2, H10 = solve_elasticity(
            args.domain, N, args.degree, verbose=args.verbose
        )
        results.append((h, L2, H10))
        print(f"{N:>5} {h:>10.4f} {L2:>12.4e} {H10:>12.4e}")

    if len(results) >= 2:
        h_vals = [r[0] for r in results]
        L2_vals = [r[1] for r in results]
        H10_vals = [r[2] for r in results]

        print("\nConvergence rates:")
        for i in range(1, len(results)):
            L2_rate = np.log(L2_vals[i] / L2_vals[i-1]) / np.log(h_vals[i] / h_vals[i-1])
            H10_rate = np.log(H10_vals[i] / H10_vals[i-1]) / np.log(h_vals[i] / h_vals[i-1])
            print(f"  N={mesh_sizes[i]:>3}: L2 rate = {L2_rate:.2f}, "
                  f"H10 rate = {H10_rate:.2f}")

        print(f"\nExpected: L2 rate ~ {args.degree + 1}, "
              f"H10 rate ~ {args.degree}")

"""
Example: Laplace-Beltrami equation on a surface.

Solves the surface PDE -Delta_S u + u = f on the boundary of a
circle (2D) or sphere (3D) using CutFEM with surface quadrature.

For the unit circle (2D curve), u = x is an eigenfunction of the
Laplace-Beltrami operator with eigenvalue 1.

For the unit sphere (3D surface), u = x is an eigenfunction with
eigenvalue 2.

Usage:
    python laplace_beltrami.py --domain circle --degree 1
    python laplace_beltrami.py --domain sphere --degree 2
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
from ufl import inner, grad, dot
from mpi4py import MPI
from petsc4py import PETSc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))
import algoim_utils
import customquad


def solve_laplace_beltrami(domain, N, degree, verbose=False):
    """Solve surface Poisson (-Delta_S u + u = f) on a surface."""

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

    opts = {"verbose": verbose}
    result = algoim_utils.generate_qr(mesh, NN, 3, domain, opts)
    [
        cut_cells, uncut_cells, outside_cells,
        qr_pts0, qr_w0, qr_pts_bdry0, qr_w_bdry0, qr_n0,
        xyz, xyz_bdry,
    ] = result

    qr_pts_bdry = [qr_pts_bdry0[k] for k in cut_cells]
    qr_w_bdry = [qr_w_bdry0[k] for k in cut_cells]
    qr_n = [qr_n0[k] for k in cut_cells]

    celltags = customquad.utils.get_celltags(
        mesh, cut_cells, uncut_cells, outside_cells,
        uncut_cell_tag=1, cut_cell_tag=2, outside_cell_tag=3,
    )

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    n = ufl.FacetNormal(mesh)

    def grad_t(u, n):
        return grad(u) - dot(n, grad(u)) * n

    # Exact solution and RHS
    if gdim == 2:
        u_exact = x[0]
        f_rhs = x[0] + x[0]  # -Delta_S x + x = x + x = 2x
    else:
        u_exact = x[0]
        f_rhs = 2 * x[0] + x[0]  # -Delta_S x + x = 2x + x = 3x

    a_surf = inner(grad_t(u, n), grad_t(v, n)) + inner(u, v)
    L_surf = inner(f_rhs, v)

    ds_cut = ufl.dx(
        subdomain_data=celltags,
        metadata={"quadrature_rule": "runtime"},
        domain=mesh,
    )

    qr_bdry = [(cut_cells, qr_pts_bdry, qr_w_bdry, qr_n)]

    A = customquad.assemble_matrix(
        dolfinx.fem.form(a_surf * ds_cut(2)), qr_bdry
    )
    A.assemble()

    b = customquad.assemble_vector(
        dolfinx.fem.form(L_surf * ds_cut(2)), qr_bdry
    )

    inactive_dofs = customquad.utils.get_inactive_dofs(V, cut_cells, [])
    A = customquad.utils.lock_inactive_dofs(inactive_dofs, A)

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

    L2_err_sq = customquad.assemble_scalar(
        dolfinx.fem.form((uh - u_exact) ** 2 * ds_cut(2)), qr_bdry
    )
    L2_err = np.sqrt(abs(L2_err_sq))

    h_val = max(dolfinx.mesh.h(mesh, mesh.topology.dim, cut_cells))

    return h_val, L2_err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laplace-Beltrami example")
    parser.add_argument("--domain", type=str, default="circle",
                        choices=["circle", "sphere"])
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.domain == "circle":
        mesh_sizes = [8, 16, 32, 64]
    else:
        mesh_sizes = [4, 8, 16]

    print(f"Laplace-Beltrami on {args.domain} with CG{args.degree}")
    print(f"{'N':>5} {'h':>10} {'L2_err':>12}")
    print("-" * 30)

    results = []
    for N in mesh_sizes:
        h, L2 = solve_laplace_beltrami(
            args.domain, N, args.degree, args.verbose
        )
        results.append((h, L2))
        print(f"{N:>5} {h:>10.4f} {L2:>12.4e}")

    if len(results) >= 2:
        print("\nConvergence rates:")
        for i in range(1, len(results)):
            rate = np.log(results[i][1] / results[i-1][1]) / np.log(
                results[i][0] / results[i-1][0]
            )
            print(f"  N={mesh_sizes[i]:>3}: L2 rate = {rate:.2f}")

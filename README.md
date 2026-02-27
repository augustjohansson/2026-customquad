# Customquad

The Customquad library allows for custom quadrature rules to be used
in FEniCSx (https://fenicsproject.org). By custom quadrature we mean
user-specified quadrature rules in different elements specified at
runtime. These can be used for performing surface and volume integrals
over cut elements in methods such as CutFEM, TraceFEM and
φ-FEM. The user can also provide normals in the quadrature
points.

See the demo `poisson.py`, the examples in `examples/`, the tests,
and read the description below to see how to use the library.

## Dependencies

This library targets **DOLFINx v0.9.0** and its ecosystem:

- **DOLFINx** v0.9.0 (https://github.com/FEniCS/dolfinx/)
- **Basix** v0.9.0 (https://github.com/FEniCS/basix/)
- **FFCx** (custom fork): https://github.com/augustjohansson/2026-ffcx-priv
- **UFL** (https://github.com/FEniCS/ufl/)

Optional dependencies for quadrature generation:
- **Algoim** (https://algoim.github.io) — quadrature on implicitly defined domains
- **cppyy** — C++/Python bindings for Algoim integration

## Installation

### Docker (recommended)

Use the provided Docker file based on the DOLFINx v0.9.0 image:
```bash
docker build -f docker/Dockerfile -t customquad .
docker run -it -v `pwd`:/root customquad bash -i
```

Then install the customquad module:
```bash
pip3 install . -U
```

Compiling the ffcx forms with runtime quadrature requires a C++
compiler. Override the C compiler with a C++ compiler:
```bash
export CC="/usr/lib/ccache/g++ -fpermissive"
```

A bashrc file with useful aliases is provided in the `utils/` directory.

### Development

```bash
git clone git@github.com:augustjohansson/2026-customquad-priv.git
cd 2026-customquad-priv
git clone git@github.com:augustjohansson/2026-ffcx-priv.git
```

Then start the Docker container and use `install-all` from
`utils/bashrc.sh`.

## Quick Start

```python
import dolfinx
import ufl
import customquad
import numpy as np
from mpi4py import MPI

# Create mesh
mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, 10, 10,
    dolfinx.mesh.CellType.quadrilateral
)

# Define form with runtime quadrature
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
dx_cut = ufl.dx(metadata={"quadrature_rule": "runtime"})
form = dolfinx.fem.form(ufl.inner(u, v) * dx_cut)

# Provide custom quadrature data
cells = np.array([0, 1, 2])  # cells to integrate over
qr_pts = [np.array([0.5, 0.5])] * 3  # quadrature points per cell
qr_w = [np.array([1.0])] * 3  # quadrature weights per cell

# Assemble
A = customquad.assemble_matrix(form, [(cells, qr_pts, qr_w)])
A.assemble()
```

## Quadrature Sources

### 1. Algoim (Level Set)

The Algoim library generates high-order quadrature rules for domains
defined by implicit functions (level sets). See `demo/algoim_utils.py`
for the interface.

```python
import algoim_utils
result = algoim_utils.generate_qr(mesh, NN, degree, "circle", opts)
```

### 2. Surface Triangulation

The `customquad.surface_triangulation` module generates quadrature rules
from triangulated surfaces (e.g., STL files). This is useful when the
domain boundary is given as a mesh rather than an implicit function.

```python
from customquad.surface_triangulation import (
    create_sphere_triangulation,
    generate_quadrature_from_triangulation,
    load_stl,
)

# From built-in shapes
verts, tris, normals = create_sphere_triangulation(radius=1.0, refinement=3)

# Or from an STL file
verts, tris, normals = load_stl("model.stl")

# Generate quadrature
result = generate_quadrature_from_triangulation(mesh, verts, tris, normals)
```

### 3. CAD (Future Work)

For CAD-based quadrature generation, a separate repository/tool is
recommended. The input should be in an open, freely available CAD format
(e.g., STEP, IGES). The interface to customquad should use numpy arrays,
following the same pattern as the surface triangulation module.

## Tests

Run the full test suite:
```bash
cd test && python -m pytest -v
```

### PDE Tests

- **Poisson convergence** (`test_pde_convergence.py`): Verifies optimal
  convergence of the Poisson equation on circle and sphere domains for
  CG1 and CG2 elements in L2 and H10 norms.

- **Linear elasticity convergence** (`test_pde_convergence.py`): Verifies
  optimal convergence of the linear elasticity equations with the same
  domain/element combinations.

- **Surface PDE** (`test_surface_pde.py`): Tests the Laplace-Beltrami
  operator on circle and sphere surfaces.

### Geometry Tests

- **Volume and area** (`test_geometry.py`): Verifies that computed volumes
  and surface areas converge to exact values.

- **Cell classification** (`test_geometry.py`): Verifies that cells are
  consistently classified as cut, uncut, or outside.

- **Cut domains** (`test_geometry.py`): Tests cut faces (2D) and cut
  edges (3D).

- **Quadrature consistency** (`test_geometry.py`): Verifies that
  quadrature data arrays have consistent sizes.

### Surface Triangulation Tests

- **Geometry checks** (`test_surface_triangulation.py`): Tests sphere and
  cube triangulation, ray casting, normals, and surface area convergence.

## Examples

- `examples/poisson_convergence.py` — Poisson convergence study
- `examples/elasticity_convergence.py` — Linear elasticity convergence study
- `examples/laplace_beltrami.py` — Surface PDE example
- `examples/geometry_check.py` — Geometry verification
- `examples/surface_triangulation_example.py` — Surface triangulation demo
- `demo/poisson.py` — Original Poisson demo with algoim
- `demo/vector_poisson.py` — Vector Poisson demo

## Description

The idea behind the library is to modify ffcx to change the generated
forms such that they evaluate basis functions in provided quadrature
points at runtime.

For example, to evaluate the bilinear form
```
a = ∫_Ω ∇u · ∇v
```
with custom quadrature points in each cell:

```python
# Custom measure
dx_cut = ufl.dx(metadata={"quadrature_rule": "runtime"})

# Custom assembly
form = dolfinx.fem.form(a_bulk * dx_cut)
A = customquad.assemble_matrix(form, [(cut_cells, qr_pts, qr_w)])
```

The generated C++ code calls `basix` at runtime to evaluate basis
functions at the provided quadrature points, rather than using
precomputed values at fixed quadrature points.

## Relevant PDEs and Applications

### PDEs of Interest

The following PDEs are particularly relevant for cut element methods:

1. **Poisson equation** — The standard benchmark for elliptic PDEs.
   Tests the basic discretization accuracy.

2. **Linear elasticity** — Tests vector-valued problems and the
   interaction between different stress components.

3. **Stokes equations** — Tests saddle-point problems and inf-sup
   stability on cut meshes.

4. **Navier-Stokes equations** — Fluid flow around immersed bodies
   is a primary application of CutFEM.

5. **Convection-diffusion** — Tests stabilization techniques on cut
   elements with convection-dominated problems.

6. **Surface PDEs (Laplace-Beltrami)** — Important for surface
   diffusion, phase-field models, and membrane mechanics.

7. **Coupled bulk-surface problems** — E.g., diffusion in a domain
   coupled to surface reactions.

8. **Heat equation** — Time-dependent problems on evolving domains.

9. **Wave equation** — Tests stability and accuracy for hyperbolic
   problems on cut meshes.

10. **Interface problems** — Two-material problems with discontinuous
    coefficients across the interface.

### Geometry Representations

Different geometry representations may be used to define the cut domain:

1. **Level set functions** — The domain is defined as {x : φ(x) < 0}
   where φ is a smooth function. Used by Algoim. Best for smooth,
   implicitly defined surfaces.

2. **CAD models (B-rep)** — Boundary representation using NURBS patches.
   Formats: STEP, IGES, BREP. Exact geometry, but complex quadrature
   generation.

3. **Surface triangulations** — Piecewise linear approximation of the
   surface. Formats: STL, OBJ, PLY. Easy to generate and widely
   available, but only piecewise linear accuracy.

4. **Constructive Solid Geometry (CSG)** — Boolean operations on
   primitive shapes (spheres, cubes, cylinders). Good for simple
   geometries.

5. **Signed distance functions** — Special case of level sets where
   |∇φ| = 1. Useful for narrow-band methods and mesh adaptation.

6. **Phase-field functions** — Smooth approximation of the interface
   with a diffuse transition region. Used in phase-field models.

7. **Point clouds** — Unstructured set of points on the surface.
   Requires reconstruction (e.g., Poisson surface reconstruction)
   before quadrature generation.

8. **Octree/voxel representations** — Hierarchical spatial
   decomposition. Used in computational geometry and medical imaging.

9. **Implicit neural representations** — Neural networks that
   represent the signed distance function. Emerging approach for
   complex geometries.

10. **Parametric surfaces** — Surfaces defined by parametric mappings.
    Used in isogeometric analysis.

## CI/CD

Automated testing runs when pull requests are merged to `main`.
See `.github/workflows/ci.yml`.

## License

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

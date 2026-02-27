"""
Surface triangulation quadrature module for customquad.

This module generates approximate quadrature rules for cut cells from
a surface triangulation (e.g., STL files). The surface triangulation
defines the boundary of a domain, and this module computes:

1. Cell classification (cut, uncut, outside) based on the triangulated
   surface
2. Volume quadrature rules for the portion of cut cells inside the domain
3. Surface quadrature rules on the intersection of the surface with each
   cut cell
4. Surface normals at quadrature points

The interface uses numpy arrays, compatible with the customquad library.

Input formats supported:
- STL files (binary and ASCII)
- Raw numpy arrays of vertices and triangles
"""

import numpy as np


def load_stl(filename):
    """Load a surface triangulation from an STL file.

    Parameters
    ----------
    filename : str
        Path to the STL file.

    Returns
    -------
    vertices : np.ndarray, shape (n_vertices, 3)
        Vertex coordinates.
    triangles : np.ndarray, shape (n_triangles, 3)
        Triangle connectivity (indices into vertices array).
    normals : np.ndarray, shape (n_triangles, 3)
        Outward unit normals for each triangle.
    """
    try:
        return _load_stl_binary(filename)
    except Exception:
        return _load_stl_ascii(filename)


def _load_stl_ascii(filename):
    """Load ASCII STL file."""
    vertices = []
    triangles = []
    normals = []

    vertex_map = {}
    current_normal = None
    current_tri = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("facet normal"):
                parts = line.split()
                current_normal = [float(parts[2]), float(parts[3]), float(parts[4])]
            elif line.startswith("vertex"):
                parts = line.split()
                v = (float(parts[1]), float(parts[2]), float(parts[3]))
                if v not in vertex_map:
                    vertex_map[v] = len(vertices)
                    vertices.append(list(v))
                current_tri.append(vertex_map[v])
            elif line.startswith("endfacet"):
                if len(current_tri) == 3 and current_normal is not None:
                    triangles.append(current_tri)
                    normals.append(current_normal)
                current_tri = []
                current_normal = None

    return (
        np.array(vertices, dtype=np.float64),
        np.array(triangles, dtype=np.int64),
        np.array(normals, dtype=np.float64),
    )


def _load_stl_binary(filename):
    """Load binary STL file."""
    with open(filename, "rb") as f:
        f.read(80)  # header
        n_triangles = np.frombuffer(f.read(4), dtype=np.uint32)[0]

        normals = np.zeros((n_triangles, 3), dtype=np.float64)
        all_verts = np.zeros((n_triangles, 3, 3), dtype=np.float64)

        for i in range(n_triangles):
            data = np.frombuffer(f.read(48), dtype=np.float32)
            normals[i] = data[0:3]
            all_verts[i, 0] = data[3:6]
            all_verts[i, 1] = data[6:9]
            all_verts[i, 2] = data[9:12]
            f.read(2)  # attribute byte count

    # Merge duplicate vertices
    flat_verts = all_verts.reshape(-1, 3)
    unique_verts, inverse = np.unique(
        flat_verts, axis=0, return_inverse=True
    )
    triangles = inverse.reshape(-1, 3)

    return unique_verts, triangles, normals


def create_sphere_triangulation(center=None, radius=1.0, refinement=3):
    """Create a triangulated sphere by recursive subdivision of an icosahedron.

    Parameters
    ----------
    center : array-like or None
        Center of the sphere. Default is origin.
    radius : float
        Radius of the sphere.
    refinement : int
        Number of subdivision levels. Level 0 = icosahedron (20 triangles).

    Returns
    -------
    vertices : np.ndarray, shape (n_vertices, 3)
    triangles : np.ndarray, shape (n_triangles, 3)
    normals : np.ndarray, shape (n_triangles, 3)
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = np.asarray(center, dtype=np.float64)

    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)

    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms

    # Icosahedron faces
    tris = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)

    # Subdivide
    for _ in range(refinement):
        verts, tris = _subdivide(verts, tris)

    # Scale and translate
    verts = verts * radius + center

    # Compute normals (outward from center)
    normals = _compute_triangle_normals(verts, tris, center)

    return verts, tris, normals


def create_cube_triangulation(center=None, half_size=1.0):
    """Create a triangulated cube (axis-aligned box).

    Parameters
    ----------
    center : array-like or None
        Center of the cube. Default is origin.
    half_size : float
        Half the side length.

    Returns
    -------
    vertices : np.ndarray, shape (8, 3)
    triangles : np.ndarray, shape (12, 3)
    normals : np.ndarray, shape (12, 3)
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = np.asarray(center, dtype=np.float64)

    s = half_size
    verts = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
    ], dtype=np.float64) + center

    # Two triangles per face, 6 faces
    tris = np.array([
        # -z face
        [0, 2, 1], [0, 3, 2],
        # +z face
        [4, 5, 6], [4, 6, 7],
        # -y face
        [0, 1, 5], [0, 5, 4],
        # +y face
        [2, 3, 7], [2, 7, 6],
        # -x face
        [0, 4, 7], [0, 7, 3],
        # +x face
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)

    normals = _compute_triangle_normals(verts, tris, center)

    return verts, tris, normals


def _subdivide(vertices, triangles):
    """Subdivide each triangle into 4 by inserting midpoint vertices."""
    edge_midpoints = {}
    new_verts = list(vertices)
    new_tris = []

    def get_midpoint(i, j):
        key = (min(i, j), max(i, j))
        if key in edge_midpoints:
            return edge_midpoints[key]
        mid = (vertices[i] + vertices[j]) / 2.0
        mid = mid / np.linalg.norm(mid)  # project to unit sphere
        idx = len(new_verts)
        new_verts.append(mid)
        edge_midpoints[key] = idx
        return idx

    for tri in triangles:
        a, b, c = tri
        ab = get_midpoint(a, b)
        bc = get_midpoint(b, c)
        ca = get_midpoint(c, a)
        new_tris.extend([
            [a, ab, ca],
            [b, bc, ab],
            [c, ca, bc],
            [ab, bc, ca],
        ])

    return np.array(new_verts, dtype=np.float64), np.array(new_tris, dtype=np.int64)


def _compute_triangle_normals(vertices, triangles, center):
    """Compute outward unit normals for each triangle."""
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0
    normals = np.cross(edge1, edge2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-16)
    normals = normals / norms

    # Ensure outward orientation
    centroids = (v0 + v1 + v2) / 3.0
    outward = centroids - center
    dots = np.sum(normals * outward, axis=1)
    flip = dots < 0
    normals[flip] *= -1

    return normals


def classify_cells(mesh, vertices, triangles, normals):
    """Classify mesh cells as cut, uncut (inside), or outside.

    Uses ray casting to determine if cell centroids are inside/outside
    the surface, and AABB intersection to find cut cells.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The background mesh.
    vertices : np.ndarray, shape (n_vertices, 3)
        Surface triangulation vertices.
    triangles : np.ndarray, shape (n_triangles, 3)
        Surface triangulation connectivity.
    normals : np.ndarray, shape (n_triangles, 3)
        Surface triangle normals.

    Returns
    -------
    cut_cells : np.ndarray
        Indices of cells intersected by the surface.
    uncut_cells : np.ndarray
        Indices of cells fully inside the domain.
    outside_cells : np.ndarray
        Indices of cells fully outside the domain.
    """
    import dolfinx.mesh

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    gdim = mesh.geometry.dim

    # Get cell bounding boxes
    geom_dofmap = mesh.geometry.dofmap.reshape(num_cells, -1)
    coords = mesh.geometry.x

    cell_min = np.zeros((num_cells, gdim))
    cell_max = np.zeros((num_cells, gdim))
    cell_centroids = np.zeros((num_cells, gdim))

    for c in range(num_cells):
        cell_coords = coords[geom_dofmap[c]][:, :gdim]
        cell_min[c] = cell_coords.min(axis=0)
        cell_max[c] = cell_coords.max(axis=0)
        cell_centroids[c] = cell_coords.mean(axis=0)

    # Triangle bounding boxes
    tri_verts = vertices[triangles]
    tri_min = tri_verts.min(axis=1)[:, :gdim]
    tri_max = tri_verts.max(axis=1)[:, :gdim]

    # Find cut cells (cells whose AABB intersects any triangle AABB)
    is_cut = np.zeros(num_cells, dtype=bool)
    for t in range(len(triangles)):
        overlaps = np.all(
            (cell_min <= tri_max[t]) & (cell_max >= tri_min[t]),
            axis=1,
        )
        is_cut |= overlaps

    # For non-cut cells, use ray casting to classify inside/outside
    is_inside = _ray_cast_inside(cell_centroids, vertices, triangles, normals, gdim)

    cut_cells = np.where(is_cut)[0]
    uncut_cells = np.where(~is_cut & is_inside)[0]
    outside_cells = np.where(~is_cut & ~is_inside)[0]

    return cut_cells, uncut_cells, outside_cells


def _ray_cast_inside(points, vertices, triangles, normals, gdim):
    """Determine if points are inside a closed surface using ray casting."""
    n_points = len(points)
    inside = np.zeros(n_points, dtype=bool)

    # Cast ray along x-axis
    for i in range(n_points):
        crossings = 0
        p = points[i]

        for t in range(len(triangles)):
            v0 = vertices[triangles[t, 0]][:gdim]
            v1 = vertices[triangles[t, 1]][:gdim]
            v2 = vertices[triangles[t, 2]][:gdim]

            if gdim == 2:
                # 2D: count edge crossings
                if _ray_intersects_segment_2d(p, v0, v1):
                    crossings += 1
            else:
                # 3D: count triangle crossings
                if _ray_intersects_triangle(p, v0, v1, v2):
                    crossings += 1

        inside[i] = (crossings % 2) == 1

    return inside


def _ray_intersects_segment_2d(point, v0, v1):
    """Test if a horizontal ray from point intersects segment v0-v1 (2D)."""
    py = point[1]
    if (v0[1] > py) == (v1[1] > py):
        return False
    t = (py - v0[1]) / (v1[1] - v0[1])
    x_intersect = v0[0] + t * (v1[0] - v0[0])
    return x_intersect > point[0]


def _ray_intersects_triangle(point, v0, v1, v2):
    """Test if a ray along +x from point intersects triangle (3D)."""
    edge1 = v1 - v0
    edge2 = v2 - v0
    ray_dir = np.array([1.0, 0.0, 0.0])

    h = np.cross(ray_dir, edge2)
    a = np.dot(edge1, h)
    if abs(a) < 1e-16:
        return False

    f = 1.0 / a
    s = point - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False

    q = np.cross(s, edge1)
    v = f * np.dot(ray_dir, q)
    if v < 0.0 or u + v > 1.0:
        return False

    t = f * np.dot(edge2, q)
    return t > 1e-16


def generate_quadrature_from_triangulation(
    mesh, vertices, triangles, normals, quadrature_degree=2
):
    """Generate custom quadrature rules from a surface triangulation.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The background mesh.
    vertices : np.ndarray, shape (n_vertices, 3)
        Surface triangulation vertices.
    triangles : np.ndarray, shape (n_triangles, 3)
        Surface triangulation connectivity.
    normals : np.ndarray, shape (n_triangles, 3)
        Surface triangle normals.
    quadrature_degree : int
        Degree of quadrature rule on each triangle.

    Returns
    -------
    cut_cells : np.ndarray
        Indices of cut cells.
    uncut_cells : np.ndarray
        Indices of uncut (inside) cells.
    outside_cells : np.ndarray
        Indices of outside cells.
    qr_pts : list of np.ndarray
        Quadrature points per cut cell (in reference coordinates).
    qr_w : list of np.ndarray
        Quadrature weights per cut cell.
    qr_pts_bdry : list of np.ndarray
        Surface quadrature points per cut cell.
    qr_w_bdry : list of np.ndarray
        Surface quadrature weights per cut cell.
    qr_n : list of np.ndarray
        Surface normals at quadrature points per cut cell.
    """
    # Classify cells
    cut_cells, uncut_cells, outside_cells = classify_cells(
        mesh, vertices, triangles, normals
    )

    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    num_cells = mesh.topology.index_map(tdim).size_local

    geom_dofmap = mesh.geometry.dofmap.reshape(num_cells, -1)
    coords = mesh.geometry.x

    # Triangle quadrature points on reference triangle
    if quadrature_degree <= 1:
        # Centroid rule
        tri_qr_pts = np.array([[1.0/3, 1.0/3]])
        tri_qr_w = np.array([0.5])
    elif quadrature_degree <= 2:
        # 3-point rule
        tri_qr_pts = np.array([
            [1.0/6, 1.0/6],
            [2.0/3, 1.0/6],
            [1.0/6, 2.0/3],
        ])
        tri_qr_w = np.array([1.0/6, 1.0/6, 1.0/6])
    else:
        # 7-point rule (degree 5)
        tri_qr_pts = np.array([
            [1.0/3, 1.0/3],
            [0.059715871789770, 0.470142064105115],
            [0.470142064105115, 0.059715871789770],
            [0.470142064105115, 0.470142064105115],
            [0.797426985353087, 0.101286507323456],
            [0.101286507323456, 0.797426985353087],
            [0.101286507323456, 0.101286507323456],
        ])
        tri_qr_w = np.array([
            0.225 / 2,
            0.132394152788506 / 2,
            0.132394152788506 / 2,
            0.132394152788506 / 2,
            0.125939180544827 / 2,
            0.125939180544827 / 2,
            0.125939180544827 / 2,
        ])

    # For each cut cell, find intersecting triangles and compute quadrature
    qr_pts_list = []
    qr_w_list = []
    qr_pts_bdry_list = []
    qr_w_bdry_list = []
    qr_n_list = []

    for cell in cut_cells:
        cell_coords = coords[geom_dofmap[cell]][:, :gdim]
        cell_min = cell_coords.min(axis=0)
        cell_max = cell_coords.max(axis=0)
        cell_size = cell_max - cell_min
        cell_vol = np.prod(cell_size)

        # Find triangles intersecting this cell
        tri_verts_all = vertices[triangles]
        tri_mins = tri_verts_all.min(axis=1)[:, :gdim]
        tri_maxs = tri_verts_all.max(axis=1)[:, :gdim]

        overlaps = np.all(
            (cell_min <= tri_maxs) & (cell_max >= tri_mins), axis=1
        )
        intersecting = np.where(overlaps)[0]

        # Surface quadrature: integrate over triangles clipped to cell
        bdry_pts = []
        bdry_w = []
        bdry_n = []

        for t_idx in intersecting:
            v0 = vertices[triangles[t_idx, 0]][:gdim]
            v1 = vertices[triangles[t_idx, 1]][:gdim]
            v2 = vertices[triangles[t_idx, 2]][:gdim]
            n_tri = normals[t_idx][:gdim]

            # Triangle area
            if gdim == 2:
                edge = v1 - v0
                tri_area = np.linalg.norm(edge)
            else:
                edge1 = v1 - v0
                edge2 = v2 - v0
                tri_area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))

            if tri_area < 1e-16:
                continue

            # Map quadrature points to physical triangle
            for q in range(len(tri_qr_w)):
                if gdim == 2:
                    # 1D quadrature on edge
                    s = tri_qr_pts[q, 0] if len(tri_qr_pts[q]) > 0 else 0.5
                    phys_pt = v0 + s * (v1 - v0)
                else:
                    s, t = tri_qr_pts[q]
                    phys_pt = v0 + s * (v1 - v0) + t * (v2 - v0)

                # Map to reference cell coordinates [0,1]^d
                ref_pt = (phys_pt - cell_min) / cell_size

                # Check if point is inside cell
                if np.all(ref_pt >= -0.01) and np.all(ref_pt <= 1.01):
                    ref_pt = np.clip(ref_pt, 0, 1)
                    bdry_pts.extend(ref_pt)
                    bdry_w.append(tri_qr_w[q] * tri_area / cell_vol)
                    bdry_n.extend(n_tri / np.linalg.norm(n_tri))

        # Bulk quadrature: use centroid with approximate inside fraction
        # This is a simple approximation; for better accuracy, use
        # sub-triangulation or recursive subdivision
        centroid_ref = np.full(gdim, 0.5)
        centroid_phys = cell_min + centroid_ref * cell_size

        # Approximate volume fraction by checking multiple sample points
        n_samples = 5
        sample_pts = np.random.RandomState(cell).uniform(
            cell_min, cell_max, size=(n_samples**gdim, gdim)
        )
        inside_count = 0
        for sp in sample_pts:
            if _point_inside_surface(sp, vertices, triangles, normals, gdim):
                inside_count += 1
        vol_fraction = inside_count / len(sample_pts) if len(sample_pts) > 0 else 0.5

        qr_pts_list.append(np.array(centroid_ref))
        qr_w_list.append(np.array([vol_fraction]))
        qr_pts_bdry_list.append(np.array(bdry_pts))
        qr_w_bdry_list.append(np.array(bdry_w))
        qr_n_list.append(np.array(bdry_n))

    return (
        cut_cells, uncut_cells, outside_cells,
        qr_pts_list, qr_w_list,
        qr_pts_bdry_list, qr_w_bdry_list, qr_n_list,
    )


def _point_inside_surface(point, vertices, triangles, normals, gdim):
    """Check if a point is inside the surface using ray casting."""
    crossings = 0
    for t in range(len(triangles)):
        v0 = vertices[triangles[t, 0]][:gdim]
        v1 = vertices[triangles[t, 1]][:gdim]
        v2 = vertices[triangles[t, 2]][:gdim]

        if gdim == 2:
            if _ray_intersects_segment_2d(point, v0, v1):
                crossings += 1
        else:
            if _ray_intersects_triangle(point, v0, v1, v2):
                crossings += 1

    return (crossings % 2) == 1

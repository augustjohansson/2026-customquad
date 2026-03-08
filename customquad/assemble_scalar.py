import dolfinx
import numba
import numpy as np
from .setup_types import ffi, PETSc
from . import utils


def _get_cpp_form(form):
    """Get the C++ form object, handling both v0.6.x and v0.7.x dolfinx APIs."""
    return getattr(form, '_cpp_object', form)


def _get_cell_integrals(form):
    """Return (integral_ids, integral_structs) for cell integrals from a ufcx form."""
    cell_type_idx = 0  # cell = 0 in ufcx_integral_type enum
    start = form.ufcx_form.form_integral_offsets[cell_type_idx]
    end = form.ufcx_form.form_integral_offsets[cell_type_idx + 1]
    ids = [form.ufcx_form.form_integral_ids[start + i] for i in range(end - start)]
    itgs = [form.ufcx_form.form_integrals[start + i] for i in range(end - start)]
    return ids, itgs


def assemble_scalar(form, qr_data, debug=False, debug_message="Mapping FE coeffs..."):
    vertices, coords, _ = utils.get_vertices(form.mesh)
    cpp_form = _get_cpp_form(form)
    integral_ids, integral_structs = _get_cell_integrals(form)
    fem_coeffs = dolfinx.cpp.fem.pack_coefficients(cpp_form)
    consts = dolfinx.cpp.fem.pack_constants(cpp_form)

    # Map coeffs if coeffs are restricted to subdomain (eg if using
    # form(v*dx(subdomain_id))
    if form.ufcx_form.num_coefficients > 0:
        for i, id in enumerate(integral_ids):
            coeffs = fem_coeffs[(dolfinx.cpp.fem.IntegralType.cell, id)]
            cmax = max(qr_data[i][0]) + 1
            if coeffs.shape[0] < cmax:
                if debug:
                    print(debug_message)
                coeffs_exp = np.zeros((cmax, coeffs.shape[1]))
                coeffs_exp[qr_data[i][0], :] = coeffs
                fem_coeffs[(dolfinx.cpp.fem.IntegralType.cell, id)] = coeffs_exp

    m = np.zeros(1, dtype=PETSc.ScalarType)

    for i, id in enumerate(integral_ids):
        kernel = integral_structs[i].tabulate_tensor_runtime_float64
        coeffs = fem_coeffs[(dolfinx.cpp.fem.IntegralType.cell, id)]

        assemble_cells(
            m,
            kernel,
            vertices,
            coords,
            coeffs,
            consts,
            qr_data[i],
        )

    return m[0]


@numba.njit  # (fastmath=True)
def assemble_cells(m, kernel, vertices, coords, coeffs, consts, qr):
    # Unpack qr
    if len(qr) == 3:
        cells, qr_pts, qr_w = qr
        qr_n = qr_pts  # dummy
    else:
        cells, qr_pts, qr_w, qr_n = qr
        assert len(cells) == len(qr_n)

    assert len(cells) == len(qr_pts)
    assert len(cells) == len(qr_w)

    # Initialize
    num_loc_vertices = vertices.shape[1]
    cell_coords = np.zeros((num_loc_vertices, 3))
    m_local = np.zeros(1, dtype=PETSc.ScalarType)
    entity_local_index = np.array([0], dtype=np.intc)

    # Don't permute
    perm = np.array([0], dtype=np.uint8)

    for k, cell in enumerate(cells):
        cell_coords[:, :] = coords[vertices[cell, :]]
        num_quadrature_points = len(qr_w[k])
        m_local.fill(0.0)

        kernel(
            ffi.from_buffer(m_local),
            ffi.from_buffer(coeffs[cell]),
            ffi.from_buffer(consts),
            ffi.from_buffer(cell_coords),
            ffi.from_buffer(entity_local_index),
            ffi.from_buffer(perm),
            num_quadrature_points,
            ffi.from_buffer(qr_pts[k]),
            ffi.from_buffer(qr_w[k]),
            ffi.from_buffer(qr_n[k]),
        )

        m[0] += m_local[0]


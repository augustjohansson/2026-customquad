# Code generation format strings for UFC (Unified Form-assembly Code)
# This code is released into the public domain.
#
# The FEniCS Project (http://www.fenicsproject.org/) 2018

declaration = """
extern ufcx_integral {factory_name};
"""

factory = """
// Code for integral {factory_name}

void tabulate_tensor_{factory_name}({scalar_type}*  A,
                                    const {scalar_type}*  w,
                                    const {scalar_type}*  c,
                                    const {geom_type}*  coordinate_dofs,
                                    const int*  entity_local_index,
                                    const uint8_t*  quadrature_permutation)
{{
{tabulate_tensor}
}}

{enabled_coefficients_init}

ufcx_integral {factory_name} =
{{
  .enabled_coefficients = {enabled_coefficients},
  .tabulate_tensor_{np_scalar_type} = tabulate_tensor_{factory_name},
  .needs_facet_permutations = {needs_facet_permutations},
  .coordinate_element = {coordinate_element},
}};

// End of code for integral {factory_name}
"""

factory_runtime = """
// Code for runtime integral {factory_name}

void tabulate_tensor_{factory_name}({scalar_type}* A,
                                    const {scalar_type}* w,
                                    const {scalar_type}* c,
                                    const {geom_type}* coordinate_dofs,
                                    const int* entity_local_index,
                                    const uint8_t* quadrature_permutation,
                                    int num_quadrature_points,
                                    const {scalar_type}* quadrature_points,
                                    const {scalar_type}* quadrature_weights,
                                    const {scalar_type}* quadrature_normals)
{{
{tabulate_tensor}
}}

// Dummy standard kernel so dolfinx form creation can store a non-NULL kernel.
// This function should never be called directly; use tabulate_tensor_runtime_{np_scalar_type}.
static void tabulate_tensor_{factory_name}_standard({scalar_type}* A,
                                    const {scalar_type}* w,
                                    const {scalar_type}* c,
                                    const {geom_type}* coordinate_dofs,
                                    const int* entity_local_index,
                                    const uint8_t* quadrature_permutation) {{}}

{enabled_coefficients_init}

ufcx_integral {factory_name} =
{{
  .enabled_coefficients = {enabled_coefficients},
  .tabulate_tensor_{np_scalar_type} = tabulate_tensor_{factory_name}_standard,
  .needs_facet_permutations = {needs_facet_permutations},
  .coordinate_element = {coordinate_element},
  .tabulate_tensor_runtime_{np_scalar_type} = tabulate_tensor_{factory_name},
}};

// End of code for runtime integral {factory_name}
"""

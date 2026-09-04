// Verified on 2026-06-27 with diameter 2.0 mm:
//   gmsh vertical_pipe_500.geo -3 -format msh2
// Result: about 506 tetrahedra.

lc_min_mm = 0.635;
lc_max_mm = 0.875;

Include "vertical_pipe_parametric.geo";

SetFactory("OpenCASCADE");
Geometry.AutoCoherence = 0;

// Define millimeters in SI so the exported mesh coordinates are in meters.
// This keeps the geometry comparable to t_junction while matching solver units.
mm = 1e-3;

// ---------------------------------------------------------------------------
// Geometry parameters (mm)
// ---------------------------------------------------------------------------
D_pipe = 2.000 * mm;
L_pipe = 18.000 * mm;
r_pipe = D_pipe / 2;

z0 = 0.0 * mm;
z1 = z0 + L_pipe;

// ---------------------------------------------------------------------------
// Mesh parameters
// You can provide either:
//   1) lc_min / lc_max directly in SI meters
//   2) lc_min_mm / lc_max_mm in millimeters, matching the style of older .geo files
// Override from CLI with for example:
//   gmsh vertical_pipe_parametric.geo -3 -setnumber lc_min_mm 0.90 -setnumber lc_max_mm 1.20
// ---------------------------------------------------------------------------
If(Exists(lc_min_mm))
  lc_min = lc_min_mm * mm;
ElseIf(!Exists(lc_min))
  lc_min = 0.90 * mm;
EndIf

If(Exists(lc_max_mm))
  lc_max = lc_max_mm * mm;
ElseIf(!Exists(lc_max))
  lc_max = 1.20 * mm;
EndIf

// ---------------------------------------------------------------------------
// Fluid volume
// ---------------------------------------------------------------------------
Cylinder(1) = {0, 0, z0, 0, 0, L_pipe, r_pipe};

// ---------------------------------------------------------------------------
// Physical groups
// ---------------------------------------------------------------------------
bnd() = CombinedBoundary{ Volume{1}; };
eps = 0.05 * mm;

sIn() = Surface In BoundingBox{
  -r_pipe - eps, -r_pipe - eps, z0 - eps,
   r_pipe + eps,  r_pipe + eps, z0 + eps
};

sOut() = Surface In BoundingBox{
  -r_pipe - eps, -r_pipe - eps, z1 - eps,
   r_pipe + eps,  r_pipe + eps, z1 + eps
};

sWalls() = bnd();
sWalls() -= {sIn(), sOut()};

Physical Volume("fluid", 1) = {1};
Physical Surface("inlet", 2) = {sIn()};
Physical Surface("outlet", 3) = {sOut()};
Physical Surface("walls", 4) = {sWalls()};

// ---------------------------------------------------------------------------
// 3D tetra mesh settings
// ---------------------------------------------------------------------------
Mesh.CharacteristicLengthMin = lc_min;
Mesh.CharacteristicLengthMax = lc_max;
Mesh.CharacteristicLengthFromCurvature = 1;
Mesh.Algorithm3D = 10;

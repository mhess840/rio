# AG95 gripper mesh for the XR ghost

Turns the DH Robotics AG95 STEP assembly into `experiments/xr/gripper_mesh.npz`,
which `rio.xr.ghost.GripperModel` draws on the ghost arm's flange. Run once from the
repo root; the CAD tools come in throwaway environments, not as project dependencies.

```sh
STEP=~/Downloads/AG-160-95-W-S_DH-Robotics-.step
OUT=/tmp/ag95 && mkdir -p $OUT
# Solid parts with their names (gmsh)
uv run --no-project --with gmsh --with numpy python scripts/ag95_mesh/tessellate_solids.py $STEP $OUT/parts.npz
# The housing, stored as a surface model that gmsh skips (OpenCascade)
uv run --no-project --with cadquery-ocp --with numpy python scripts/ag95_mesh/tessellate_housing.py $STEP $OUT/housing.npy
# Group the parts, map to the flange frame, thin out (3 mm cells)
uv run python scripts/ag95_mesh/build.py $OUT 3 experiments/xr/gripper_mesh.npz
```

Frame: the STEP's mount face centre is the flange origin, its y axis (approach) the
flange z, its x axis (opening) the flange x. The proximal phalanx pivot and its joints
in `build.py` were read off the CAD by hand; the inner drive links are approximate.

# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""AG95 STEP housing -> triangle soup (mm), with OpenCascade. See README.md.

The housing is stored as a surface model, which gmsh does not import: every face
that belongs to no solid is meshed here.

Usage: tessellate_housing.py <assembly.step> <housing.npy>
"""

import sys

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS


def faces(shape):
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        yield explorer.Current()
        explorer.Next()


def main(source: str, out: str) -> None:
    reader = STEPControl_Reader()
    reader.ReadFile(source)
    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, 0.3, False, 0.3, True)
    in_solid = set()
    solids = TopExp_Explorer(shape, TopAbs_SOLID)
    while solids.More():
        in_solid.update(face.__hash__() for face in faces(solids.Current()))
        solids.Next()
    soups = []
    for face in faces(shape):
        if face.__hash__() in in_solid:
            continue
        location = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(TopoDS.Face(face), location)
        if mesh is None:
            continue
        move = location.Transformation()
        nodes = [mesh.Node(i).Transformed(move) for i in range(1, mesh.NbNodes() + 1)]
        points = np.array([[p.X(), p.Y(), p.Z()] for p in nodes])
        corners = np.array([mesh.Triangle(i).Get() for i in range(1, mesh.NbTriangles() + 1)]) - 1
        soups.append(points[corners.reshape(-1)])
    soup = np.concatenate(soups)
    print("housing triangles", len(soup) // 3)
    np.save(out, soup)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

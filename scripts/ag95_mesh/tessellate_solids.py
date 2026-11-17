# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""AG95 STEP solids -> one named triangle soup (mm) per part, with gmsh. See README.md.

Usage: tessellate_solids.py <assembly.step> <parts.npz>
"""

import sys

import gmsh
import numpy as np


def main(source: str, out: str) -> None:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Geometry.OCCImportLabels", 1)
    gmsh.model.occ.importShapes(source)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", 4.0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 1.0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
    gmsh.model.mesh.generate(2)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    index = {tag: i for i, tag in enumerate(tags.tolist())}
    points = coords.reshape(-1, 3)
    parts = {}
    for dim, tag in gmsh.model.getEntities(3):
        triangles = []
        for _, surface in gmsh.model.getBoundary([(dim, tag)], oriented=False):
            types, _, nodes = gmsh.model.mesh.getElements(2, abs(surface))
            triangles += [np.asarray(n).reshape(-1, 3) for t, n in zip(types, nodes, strict=True) if t == 2]
        if not triangles:
            continue
        corners = np.concatenate(triangles).reshape(-1)
        soup = points[[index[node] for node in corners]]
        name = gmsh.model.getEntityName(dim, tag)
        parts[f"{tag}:{name}".replace("/", "|")] = soup
        print(tag, name.split("/")[-1], len(soup) // 3)
    np.savez_compressed(out, **parts)
    gmsh.finalize()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

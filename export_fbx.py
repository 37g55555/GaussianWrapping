#!/usr/bin/env python3

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def find_colmap():
    candidates = [Path(__file__).resolve().parent/'tools/colmap/bin/colmap', shutil.which('colmap')]
    for path in dict.fromkeys(str(p) for p in candidates if p):
        if Path(path).is_file() and subprocess.run([path, 'mesh_texturer', '-h'], capture_output=True).returncode == 0:
            return path
    raise RuntimeError('No installed COLMAP supports mesh_texturer (COLMAP 4.2.1 required).')


def mesh_spacing(vertices, faces):
    import numpy as np
    ids = np.linspace(0, len(faces)-1, min(50000, len(faces)), dtype=int)
    xyz = vertices[faces[ids]]
    lengths = np.linalg.norm(xyz-np.roll(xyz, 1, axis=1), axis=2).min(1)
    return float(np.median(lengths[lengths > 0]))


def boundary_groups(faces):
    import numpy as np
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    adjacency = {}
    for a, b in edges[counts == 1]:
        adjacency.setdefault(int(a), []).append(int(b)); adjacency.setdefault(int(b), []).append(int(a))
    remaining = set(adjacency); groups = []
    while remaining:
        start = next(iter(remaining)); group = {start}; stack = [start]; remaining.remove(start)
        while stack:
            for nxt in adjacency[stack.pop()]:
                if nxt in remaining:
                    remaining.remove(nxt); group.add(nxt); stack.append(nxt)
        groups.append(np.array(sorted(group)))
    return groups, adjacency


def _prepare_contact_boundaries(vertices, faces):
    """Join only roundoff duplicates without moving retained vertices."""
    import numpy as np
    from scipy.spatial import cKDTree
    epsilon = max(float(np.ptp(vertices, axis=0).max())*1e-8, 1e-14)
    groups, _ = boundary_groups(faces)
    boundary = np.concatenate(groups) if groups else np.empty(0, dtype=np.int64)
    pairs = boundary[cKDTree(vertices[boundary]).query_pairs(epsilon, output_type='ndarray')]
    mapping = np.arange(len(vertices))
    def find(i):
        while mapping[i] != i:
            mapping[i] = mapping[mapping[i]]; i = mapping[i]
        return i
    for a, b in pairs:
        a, b = find(a), find(b); mapping[max(a, b)] = min(a, b)
    joined = 0
    if len(pairs):
        mapping = np.array([find(i) for i in range(len(vertices))])
        joined = len(vertices)-len(np.unique(mapping))
        faces = mapping[faces]
        faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])]
    else:
        faces = faces.copy()
    if joined:
        print('Contact boundaries: joined roundoff duplicates:', joined, flush=True)
    return vertices, faces


def _boundary_loops(edges, directed=False):
    """Use every boundary edge once, separating cycles at shared vertices."""
    import numpy as np
    remaining = set(map(tuple, (edges if directed else np.sort(edges, axis=1)).tolist())); adjacency = {}; balance = {}
    def key(a, b): return (a, b) if directed else tuple(sorted((a, b)))
    for a, b in remaining:
        adjacency.setdefault(a, []).append(b)
        if not directed: adjacency.setdefault(b, []).append(a)
        balance[a] = balance.get(a, 0)+1; balance[b] = balance.get(b, 0)-1
    if (directed and any(balance.values())) or (not directed and any(len(n) % 2 for n in adjacency.values())):
        raise ValueError('Contact boundary contains an open chain.')
    loops = []
    while remaining:
        a, b = min(remaining); ring = [a]; remaining.remove((a, b))
        while b != ring[0]:
            ring.append(b)
            nxt = min(c for c in adjacency[b] if key(b, c) in remaining)
            remaining.remove(key(b, nxt)); a, b = b, nxt
        walk = []; seen = {}
        for vertex in ring+[ring[0]]:
            if vertex in seen:
                k = seen[vertex]; loops.append(np.asarray(walk[k:], dtype=np.int64))
                walk = walk[:k]; seen = {p: i for i, p in enumerate(walk)}
            seen[vertex] = len(walk); walk.append(vertex)
    return loops


def _contact_basis(vertices, ids, normal):
    import numpy as np
    delta = vertices[ids]-vertices[ids[0]]
    delta -= (delta@normal)[:, None]*normal
    length = np.linalg.norm(delta, axis=1)
    if length.max() <= 1e-15: raise ValueError('Contact boundary has no stable planar projection.')
    axis = delta[np.flatnonzero(length >= length.max()*(1-1e-10))[0]]
    axis /= np.linalg.norm(axis)
    return np.column_stack([axis, np.cross(normal, axis)])


def _face_edge_codes(faces):
    import numpy as np
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).astype(np.uint64), axis=1)
    return np.unique((edges[:, 0] << np.uint64(32)) | edges[:, 1])


def _contact_patch(v, loops, triangles, normal, offset, spacing, noise, occupied_codes):
    """Keep the observed boundary; interpolate only uneven contact interiors."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    boundary = np.concatenate(loops)
    basis = _contact_basis(v, boundary, normal)
    height = v@normal+offset
    base_count = len(v); points = v.tolist(); triangles = np.asarray(triangles, dtype=np.int64)
    # Small, near-planar contacts need no invented interior detail.
    area = np.linalg.norm(np.cross(v[triangles[:, 1]]-v[triangles[:, 0]], v[triangles[:, 2]]-v[triangles[:, 0]]), axis=1).sum()*.5
    target_edge = max(spacing*3, np.sqrt(area/400))
    flat = np.percentile(abs(height[boundary]), 90) <= max(noise*2, spacing*2)
    for _ in range(4 if not flat else 1):
        p = np.asarray(points); edges = np.sort(np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        # A cap diagonal must not reuse an existing interior body edge: that
        # would attach four faces to one edge even when all hole edges close.
        codes = (unique[:, 0].astype(np.uint64) << np.uint64(32)) | unique[:, 1].astype(np.uint64)
        clash = unique[:, 1] < base_count
        occupied_edge = np.zeros(len(codes), dtype=bool)
        for occupied in occupied_codes:
            pos = np.searchsorted(occupied, codes)
            occupied_edge |= (pos < len(occupied)) & (occupied[np.minimum(pos, len(occupied)-1)] == codes)
        clash &= occupied_edge
        long = np.linalg.norm((p[unique[:, 0]]-p[unique[:, 1]])@basis, axis=1) > target_edge
        edges = unique[(counts == 2) & (clash | ((not flat) & long))]
        if not len(edges): break
        mids = {tuple(edge): len(points)+i for i, edge in enumerate(edges)}
        points.extend(((p[edges[:, 0]]+p[edges[:, 1]])*.5).tolist()); refined = []
        for triangle in triangles:
            a, b, c = map(int, triangle)
            m = [mids.get(tuple(sorted(pair))) for pair in ((a, b), (b, c), (c, a))]
            count = sum(x is not None for x in m)
            if count == 0: refined.append([a, b, c]); continue
            if count == 3:
                ab, bc, ca = m; refined.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]); continue
            shift = next(k for k in range(3) if m[k] is not None and (count == 1 or m[(k+1)%3] is not None))
            a, b, c = np.roll(triangle, -shift); ab, bc, ca = np.roll(np.array(m, dtype=object), -shift)
            if count == 1: refined.extend([[a, ab, c], [ab, b, c]])
            else: refined.extend([[b, bc, ab], [a, ab, c], [ab, bc, c]])
        triangles = np.asarray(refined, dtype=np.int64)
    p = np.asarray(points)
    if len(p) == base_count:
        return p, triangles
    used = np.unique(triangles); local = np.full(len(p), -1, dtype=int); local[used] = np.arange(len(used))
    edges = np.unique(np.sort(np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1), axis=0)
    a, b = local[edges[:, 0]], local[edges[:, 1]]
    weight = 1/np.maximum(np.linalg.norm((p[edges[:, 0]]-p[edges[:, 1]])@basis, axis=1), spacing*.01)
    laplace = coo_matrix((np.r_[weight, weight, -weight, -weight], (np.r_[a, b, a, b], np.r_[a, b, b, a])), shape=(len(used), len(used))).tocsr()
    fixed = used < base_count; free = ~fixed
    if free.any():
        z = spsolve(laplace[free][:, free], -laplace[free][:, fixed]@height[used[fixed]])
        if not np.isfinite(z).all():
            raise ValueError('Contact interpolation is unstable; review the boundary.')
        z = np.clip(z, 0, max(float(height[boundary].max()), 0))
        p[used[free]] += (z-(p[used[free]]@normal+offset))[:, None]*normal
    return p, triangles


def _patch_intersection_mask(v, faces, new_faces, repaired_loops, spacing, epsilon):
    """Return intersecting NEW faces; retained geometry is never edited."""
    import numpy as np
    import open3d as o3d
    mask = np.zeros(len(new_faces), bool)
    if not len(faces) or not len(new_faces): return mask
    xyz = v[new_faces]
    low, high = xyz.min((0, 1))-epsilon*10, xyz.max((0, 1))+epsilon*10
    body_xyz = v[faces]
    nearby = (body_xyz.max(1) >= low).all(1) & (body_xyz.min(1) <= high).all(1)
    if not nearby.any(): return mask
    boundary_edges = np.vstack([np.column_stack([ring, np.roll(ring, -1)]) for ring in repaired_loops])
    unit = max(float(np.ptp(v, axis=0).max()), spacing)
    local_v = (v-v[0])/unit
    seam = local_v[boundary_edges]; delta = seam[:, 1]-seam[:, 0]
    squared = np.maximum(np.sum(delta*delta, axis=1), 1e-30)
    seam_epsilon = max(epsilon*50/unit, np.finfo(np.float32).eps*16)
    def off_seam(points):
        keep = np.zeros(len(points), bool)
        for start in range(0, len(points), 512):
            distance = points[start:start+512, None, :]-seam[None, :, 0, :]
            t = np.clip(np.einsum('nkj,kj->nk', distance, delta)/squared, 0, 1)
            distance -= t[:, :, None]*delta
            keep[start:start+512] = np.sum(distance*distance, axis=2).min(1) > seam_epsilon**2
        return keep
    def crossing(source, target):
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(local_v.astype(np.float32)), o3d.core.Tensor(target.astype(np.uint32)))
        raw_edges = np.sort(np.vstack([source[:, [0, 1]], source[:, [1, 2]], source[:, [2, 0]]]), axis=1)
        edges, inverse = np.unique(raw_edges, axis=0, return_inverse=True)
        owners = np.tile(np.arange(len(source)), 3)
        target_edges = np.sort(np.vstack([target[:, [0, 1]], target[:, [1, 2]], target[:, [2, 0]]]), axis=1)
        common = set(map(tuple, target_edges))
        selected = np.array([i for i, edge in enumerate(edges) if tuple(edge) not in common], dtype=int)
        a, b = local_v[edges[selected, 0]], local_v[edges[selected, 1]]; direction = b-a
        length = np.linalg.norm(direction, axis=1); direction /= np.maximum(length[:, None], epsilon/unit)
        inset = max(epsilon*10/unit, spacing*1e-3/unit, np.finfo(np.float32).eps*8)
        rays = np.column_stack([a+direction*inset, direction]).astype(np.float32)
        if len(rays):
            result = scene.cast_rays(o3d.core.Tensor(rays)); hit = result['t_hit'].numpy()
            bad = np.flatnonzero((hit > inset) & (hit < length-inset*2))
            keep = off_seam(a[bad]+direction[bad]*(hit[bad]+inset)[:, None])
            source_ids = owners[np.isin(inverse, selected[bad[keep]])]
            target_ids = result['primitive_ids'].numpy()[bad[keep]]
        else:
            source_ids = target_ids = np.array([], dtype=int)
        # Edge rays cannot detect coplanar overlap; probe face interiors as well.
        tri = local_v[source]; centers = tri.mean(1)
        norms = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        norms /= np.maximum(np.linalg.norm(norms, axis=1)[:, None], 1e-30)
        probe = max(epsilon*10/unit, np.finfo(np.float32).eps*8)
        rays = np.column_stack([centers+norms*probe, -norms]).astype(np.float32)
        result = scene.cast_rays(o3d.core.Tensor(rays)); hit = result['t_hit'].numpy()
        ids = np.flatnonzero((hit >= 0) & (hit <= probe*2)); ids = ids[off_seam(centers[ids])]
        return np.unique(np.r_[source_ids, ids]).astype(int), np.unique(np.r_[target_ids, result['primitive_ids'].numpy()[ids]]).astype(int)
    forward, _ = crossing(new_faces, faces[nearby])
    _, reverse = crossing(faces[nearby], new_faces)
    mask[forward] = True; mask[reverse] = True
    return mask


def _flatten_intersecting_patch(v, patch, bad, normal, offset, fixed_count):
    """Replace only bad faces: planar interiors plus fixed-boundary transitions."""
    import numpy as np
    p = v.copy(); safe_ids = np.unique(patch[~bad])
    movable = np.setdiff1d(np.unique(patch[bad]), safe_ids)
    movable = movable[movable >= fixed_count]
    p[movable] -= (p[movable]@normal+offset)[:, None]*normal
    added, result = [], []
    for triangle, replace in zip(patch, bad):
        if not replace:
            result.append(triangle.tolist()); continue
        # An inset planar triangle keeps shared boundary vertices untouched.
        # Its transition triangles occupy only the original bad triangle.
        xyz = p[triangle]; inner = xyz*.8+xyz.mean(0)*.2
        inner -= (inner@normal+offset)[:, None]*normal
        x, y, z = np.arange(len(p)+len(added), len(p)+len(added)+3)
        added.extend(inner.tolist()); a, b, c = map(int, triangle)
        result.extend([[a, b, y], [a, y, x], [b, c, z], [b, z, y], [c, a, x], [c, x, z], [x, y, z]])
    assert np.array_equal(p[:fixed_count], v[:fixed_count])
    assert np.array_equal(p[safe_ids], v[safe_ids])
    return np.vstack([p, np.asarray(added).reshape(-1, 3)]), np.asarray(result, dtype=np.int64)


def _triangulate_contact(v, rings, xy, epsilon):
    """Preserve all boundary edges, including small holes lost in projection."""
    import numpy as np
    import mapbox_earcut
    def triangulate(rings, xy):
        ids = np.concatenate(rings); ends = np.cumsum([len(g) for g in rings]).astype(np.uint32)
        triangles = ids[mapbox_earcut.triangulate_float64(np.ascontiguousarray(xy[ids]), ends).reshape(-1, 3)]
        # A collinear Earcut face is a T-junction: insert its middle vertex
        # into the neighboring triangle instead of retaining a zero-area face.
        xyz = v[triangles]
        flat = np.linalg.norm(np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]), axis=1) <= epsilon**2
        pending = triangles[flat]; triangles = triangles[~flat]
        for triangle in sorted(pending, key=lambda t: -np.ptp(v[t], axis=0).max()):
            pairs = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
            a, b, c = triangle[list(max(pairs, key=lambda p: np.linalg.norm(v[triangle[p[0]]]-v[triangle[p[1]]])))]
            neighbors = np.flatnonzero(np.isin(triangles, [a, b]).sum(1) == 2)
            if not len(neighbors): raise ValueError('Degenerate contact triangle cannot be connected reliably.')
            replacement = []
            for index in neighbors:
                t = triangles[index]; k = next(k for k in range(3) if {t[k], t[(k+1)%3]} == {a, b})
                x, y, z = np.roll(t, -k); replacement.extend([[x, c, z], [c, y, z]])
            triangles = np.vstack([np.delete(triangles, neighbors, axis=0), np.asarray(replacement)])
        # Restore boundary vertices omitted by Earcut on collinear polygon edges.
        edges = np.sort(np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
        edges, counts = np.unique(edges, axis=0, return_counts=True)
        lookup = {int(p): (ring, k) for ring in rings for k, p in enumerate(ring)}; chains = {}
        for a, b in edges[counts == 1]:
            ring, k = lookup[int(a)]; other, j = lookup[int(b)]
            if ring is not other or (k-j)%len(ring) in (1, len(ring)-1): continue
            for step in (1, -1):
                indices = [k]; position = (k+step)%len(ring)
                while position != j: indices.append(position); position = (position+step)%len(ring)
                chain = ring[indices+[j]]; delta = xy[b]-xy[a]; length = float(delta@delta)
                t = (xy[chain]-xy[a])@delta/max(length, 1e-30)
                error = np.linalg.norm(xy[chain]-xy[a]-t[:, None]*delta, axis=1)
                if len(chain)>2 and error.max()<epsilon and t.min()>-1e-8 and t.max()<1+1e-8:
                    chains[(int(a), int(b))] = chain; break
        pending = triangles.tolist(); local_caps = []
        while pending:
            triangle = pending.pop()
            for k in range(3):
                a, b, c = triangle[k], triangle[(k+1)%3], triangle[(k+2)%3]
                chain = chains.get(tuple(sorted((a, b))))
                if chain is not None:
                    chain = chain if chain[0] == a else chain[::-1]
                    pending.extend([[int(x), int(y), c] for x, y in zip(chain, chain[1:])]); break
            else: local_caps.append(triangle)
        return np.asarray(local_caps, dtype=np.int64).reshape(-1, 3)
    triangles = triangulate(rings, xy)
    expected = np.sort(np.vstack([np.column_stack([ring, np.roll(ring, -1)]) for ring in rings]), axis=1)
    previous = float('inf')
    while True:
        edges = np.sort(np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
        edges, count = np.unique(edges, axis=0, return_counts=True)
        if (count > 2).any(): raise ValueError('Contact triangulation has overlapping connections.')
        residual, count = np.unique(np.vstack([expected, edges[count == 1]]), axis=0, return_counts=True)
        residual = residual[count == 1]
        if not len(residual): return triangles
        if len(residual) >= previous: raise ValueError('Contact boundary cannot be triangulated reliably.')
        previous = len(residual); added = []
        for ring in _boundary_loops(residual):
            points = v[ring]; _, _, basis = np.linalg.svd(points-points.mean(0), full_matrices=False)
            added.append(triangulate([ring], v@basis[:2].T))
        triangles = np.vstack([triangles, *added])


def _contact_connection_errors(retained, patch, rings):
    """Check new edges and seams against retained faces without editing them."""
    import numpy as np
    edges = np.sort(np.vstack([patch[:, [0, 1]], patch[:, [1, 2]], patch[:, [2, 0]]]), axis=1)
    boundary = np.sort(np.vstack([np.column_stack([r, np.roll(r, -1)]) for r in rings]), axis=1)
    monitored = np.unique(np.vstack([edges, boundary]), axis=0)
    vertices = np.zeros(max(int(retained.max()), int(monitored.max()))+1, dtype=bool)
    vertices[monitored] = True
    nearby = retained[vertices[retained].sum(1) >= 2]
    faces = np.vstack([nearby, patch])
    edges, count = np.unique(np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0, return_counts=True)
    stride = int(max(edges.max(), monitored.max()))+1
    selected = np.isin(edges[:, 0]*stride+edges[:, 1], monitored[:, 0]*stride+monitored[:, 1])
    return edges[selected & (count != 2)]


def _repair_contact(v, retained, rings, xy, normal, offset, spacing, noise, epsilon, occupied_codes, retry=False):
    """Retry only this contact using local projection and boundary interpolation."""
    import numpy as np
    for attempt in ((1,) if retry else (0, 1)):
        try:
            projection = xy
            if attempt:
                points = v[np.concatenate(rings)]
                _, _, basis = np.linalg.svd(points-points.mean(0), full_matrices=False)
                projection = v@basis[:2].T
            triangles = _triangulate_contact(v, rings, projection, epsilon)
            candidate, patch = _contact_patch(v, rings, triangles, normal, offset, spacing, noise, occupied_codes)
            bad = _patch_intersection_mask(candidate, retained, patch, rings, spacing, epsilon)
            if bad.any():
                print('Contact: planar replacement of', int(bad.sum()), 'intersecting new faces; safe faces unchanged.', flush=True)
                candidate, patch = _flatten_intersecting_patch(candidate, patch, bad, normal, offset, len(v))
            xyz = candidate[patch]
            if (np.linalg.norm(np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]), axis=1) <= epsilon**2).any():
                raise ValueError('Degenerate contact patch.')
            if len(_contact_connection_errors(retained, patch, rings)):
                raise ValueError('Contact connections need retriangulation.')
            assert np.array_equal(candidate[:len(v)], v)
            return candidate, patch
        except ValueError as error:
            if attempt:
                raise ValueError(f'Contact repair failed after local retriangulation: {error}') from error
            print('Contact: retrying failed boundary with local projection:', error, flush=True)


def close_flat_base(vertices, faces, support):
    """Close supported contact holes without recutting or moving observed surfaces."""
    import numpy as np
    import trimesh
    if support is None:
        raise ValueError('Matching .support.json is required; rerun object isolation.')
    groups, _ = boundary_groups(faces)
    if not groups:
        print('Already closed; no open base to repair.', flush=True)
        return vertices, faces, np.zeros(len(faces), bool)
    vertices, faces = _prepare_contact_boundaries(vertices, faces)
    groups, adjacency = boundary_groups(faces)
    if not groups:
        return vertices, faces, np.zeros(len(faces), bool)
    spacing = mesh_spacing(vertices, faces)
    normal = np.asarray(support['plane'][:3], dtype=float); offset = float(support['plane'][3])
    length = np.linalg.norm(normal); normal /= length; offset /= length
    size = float(np.ptp(vertices@normal)); epsilon = max(size*1e-8, 1e-12)
    noise = max(float(support['noise']), epsilon)
    height = vertices@normal+offset; band = min(max(noise*3, spacing*2), size*.1)
    bottom = [g for g in groups if height[g].min() >= -band and np.median(height[g]) <= band and height[g].max() <= band*4]
    if not bottom: raise ValueError('No small contact boundary near the support plane; review the mesh.')
    edges = np.asarray([(int(a), b) for group in bottom for a in group for b in adjacency[int(a)] if a < b])
    raw = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    ordered = np.sort(raw, axis=1); count = len(vertices)
    selected = np.isin(ordered[:, 0]*count+ordered[:, 1], edges[:, 0]*count+edges[:, 1])
    loops = _boundary_loops(raw[selected, ::-1], directed=True)
    v = vertices.copy(); patches = []; body_codes = _face_edge_codes(faces); patch_codes = []
    xy = v@_contact_basis(v, np.concatenate(loops), normal)
    def contains(polygon, point):
        a = polygon; b = np.roll(a, -1, axis=0)
        active = np.flatnonzero((a[:, 1] > point[1]) != (b[:, 1] > point[1]))
        x = a[active, 0]+(point[1]-a[active, 1])*(b[active, 0]-a[active, 0])/(b[active, 1]-a[active, 1])
        return bool(np.count_nonzero(x > point[0]) % 2)
    signed_areas = np.asarray([np.sum(xy[g, 0]*np.roll(xy[g, 1], -1)-xy[g, 1]*np.roll(xy[g, 0], -1))*.5 for g in loops])
    areas = abs(signed_areas)
    parents = []
    for i, ring in enumerate(loops):
        if areas[i] < epsilon**2: raise ValueError('Contact boundary cannot be projected onto the support plane.')
        enclosing = [j for j, outer in enumerate(loops) if areas[j] > areas[i] and signed_areas[j]*signed_areas[i] < 0 and contains(xy[outer], xy[ring[0]])]
        parents.append(min(enclosing, key=lambda j: areas[j]) if enclosing else -1)
    depth = []
    for i in range(len(loops)):
        n = 0; p = parents[i]
        while p >= 0: n += 1; p = parents[p]
        depth.append(n)
    repaired_loops = []
    for i, outer in enumerate(loops):
        if depth[i] % 2: continue
        rings = [outer]+[loops[j] for j in range(len(loops)) if parents[j] == i]
        retained = np.vstack([faces, *[p for _, p in patches]])
        v, patch = _repair_contact(v, retained, rings, xy, normal, offset, spacing, noise, epsilon, [body_codes, *patch_codes])
        patches.append((rings, patch)); patch_codes.append(_face_edge_codes(patch)); repaired_loops.extend(rings)
    new_faces = np.vstack([p for _, p in patches])
    if len(_contact_connection_errors(faces, new_faces, repaired_loops)):
        for i, (rings, patch) in enumerate(patches):
            retained = np.vstack([faces, *[p for j, (_, p) in enumerate(patches) if j != i]])
            if len(_contact_connection_errors(retained, patch, rings)):
                print('Contact: reprocessing boundary after final connection check.', flush=True)
                occupied = [body_codes, *[codes for j, codes in enumerate(patch_codes) if j != i]]
                v, patch = _repair_contact(v, retained, rings, xy, normal, offset, spacing, noise, epsilon, occupied, retry=True)
                patches[i] = (rings, patch); patch_codes[i] = _face_edge_codes(patch)
        new_faces = np.vstack([p for _, p in patches])
        if len(_contact_connection_errors(faces, new_faces, repaired_loops)):
            raise ValueError('Contact connections remain invalid after local repair.')
    final = np.vstack([faces, new_faces]); mesh = trimesh.Trimesh(v, final, process=False)
    trimesh.repair.fix_normals(mesh, multibody=True); final = np.asarray(mesh.faces)
    edges = np.sort(np.vstack([final[:, [0, 1]], final[:, [1, 2]], final[:, [2, 0]]]), axis=1)
    _, count = np.unique(edges, axis=0, return_counts=True)
    assert np.array_equal(v[:len(vertices)], vertices)
    invalid = int(np.count_nonzero(count != 2))
    if invalid:
        raise ValueError(f'Mesh has {invalid} open or non-manifold edges; FBX export stopped for review.')
    print('Base repaired without recutting; original vertices unchanged; new faces:', len(new_faces), flush=True)
    return v, final, np.r_[np.zeros(len(faces), bool), np.ones(len(new_faces), bool)]


def texture_obj(v, f, synthetic, dense, folder, colmap):
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree
    from plyfile import PlyData
    from PIL import Image
    body = folder/'body.ply'; textured = folder/'textured'; textured.mkdir()
    trimesh.Trimesh(v, f[~synthetic], process=False).export(body)
    subprocess.run([str(colmap), 'mesh_texturer', '--workspace_path', str(dense), '--input_path', str(body), '--output_path', str(textured)], check=True)
    ply = PlyData.read(str(textured/'mesh.ply')); pv = np.column_stack([ply['vertex'].data[k] for k in ('x', 'y', 'z')])
    pf = np.stack(ply['face'].data['vertex_indices']); uv = np.stack(ply['face'].data['texcoord']).reshape(-1, 3, 2)
    photo_faces = f[~synthetic]
    # COLMAP preserves input indices. Nearest-position remapping would merge
    # deliberately split coincident vertices and reopen the repaired contact.
    if len(pv) != len(v) or not np.array_equal(pf, photo_faces):
        raise ValueError('Texture mapping changed mesh topology.')
    if np.linalg.norm(pv[np.unique(pf)]-v[np.unique(pf)], axis=1).max() > 1e-6:
        raise ValueError('Texture mapping changed mesh coordinates.')
    image = Image.open(textured/'texture.png').convert('RGB')
    # Reserve a narrow atlas strip for the new base, using nearby body color.
    base = f[synthetic]; width = 4096; strip = 128 if len(base) else 0; pixels = np.asarray(image)
    if len(base):
        # Query from the contact into the body, avoiding expensive searches
        # from every distant body face into a nearly planar contact point set.
        distance, ids = cKDTree(v[photo_faces].mean(1)).query(v[np.unique(base)])
        near = np.full(len(photo_faces), np.inf); np.minimum.at(near, ids, distance)
        chosen = np.argsort(near)[:min(512, int(np.isfinite(near).sum()))]; centers = uv[chosen].mean(1)
        x = np.clip((centers[:, 0]*image.width).astype(int), 0, image.width-1)
        y = np.clip(((1-centers[:, 1])*image.height).astype(int), 0, image.height-1)
        color = tuple(np.median(pixels[y, x], axis=0).astype('uint8'))
    else:
        color = (127, 127, 127)
    atlas = Image.new('RGB', (width, width), color); atlas.paste(image.resize((width, width-strip), Image.Resampling.LANCZOS), (0, 0))
    texture_name = 'albedo.png'
    atlas.save(folder/texture_name); uv[:, :, 1] = strip/width+uv[:, :, 1]*(1-strip/width)
    if len(base):
        points = v[np.unique(base)]; _, vectors = np.linalg.eigh(np.cov(points.T)); xy = v@vectors[:, -2:]
        low, high = xy[np.unique(base)].min(0), xy[np.unique(base)].max(0)
        buv = (xy[base]-low)/np.maximum(high-low, 1e-12)
        buv[:, :, 0] = .01+.98*buv[:, :, 0]; buv[:, :, 1] = strip/width*(.05+.90*buv[:, :, 1])
        uv = np.concatenate([uv, buv])
    triangles = np.vstack([photo_faces, base]); (folder/'mesh.mtl').write_text(f'newmtl Surface\nKd 1 1 1\nmap_Kd {texture_name}\n\nnewmtl FlatBase\nKd 1 1 1\nmap_Kd {texture_name}\n')
    with (folder/'mesh.obj').open('w') as handle:
        handle.write('mtllib mesh.mtl\nusemtl Surface\n')
        for p in v: handle.write('v %.9g %.9g %.9g\n'%tuple(p))
        for p in uv.reshape(-1, 2): handle.write('vt %.9g %.9g\n'%tuple(p))
        for i, triangle in enumerate(triangles):
            if i == len(photo_faces): handle.write('usemtl FlatBase\n')
            handle.write('f '+' '.join('%d/%d'%(int(j)+1, i*3+k+1) for k,j in enumerate(triangle))+'\n')


def blender_export(obj_path, output, support_path):
    import bpy
    from mathutils import Matrix, Vector
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.obj_import(filepath=obj_path, forward_axis='Y', up_axis='Z')
    obj = bpy.context.view_layer.objects.active
    plane = json.loads(Path(support_path).read_text())['plane']
    normal = Vector(plane[:3]); offset = float(plane[3])/normal.length
    normal.normalize()
    rotation = normal.rotation_difference(Vector((0, 0, 1))).to_matrix().to_4x4()
    obj.data.transform(rotation @ obj.matrix_world); obj.matrix_world = Matrix.Identity(4)
    low = [min(v.co[i] for v in obj.data.vertices) for i in (0, 1)]
    high = [max(v.co[i] for v in obj.data.vertices) for i in (0, 1)]
    obj.data.transform(Matrix.Translation((-(low[0]+high[0])*.5, -(low[1]+high[1])*.5, offset)))
    obj.data.update()
    for p in obj.data.polygons: p.use_smooth = True
    obj.data.set_sharp_from_angle(angle=.7853981633974483)
    for material in obj.data.materials:
        for node in material.node_tree.nodes:
            image = getattr(node, 'image', None)
            if image: image.pack()
    bpy.ops.export_scene.fbx(filepath=output, use_selection=True, object_types={'MESH'}, path_mode='COPY', embed_textures=True, add_leaf_bones=False, axis_forward='Y', axis_up='Z')


def main():
    import numpy as np
    import open3d as o3d
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-r', type=Path, required=True)
    parser.add_argument('-s', type=Path, required=True)
    parser.add_argument('-m', type=Path)
    args = parser.parse_args(); output_dir = (args.m or args.r.parent).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir/(args.r.stem+'_repaired.fbx')
    colmap = find_colmap()
    blender = next((str(p) for p in (Path(__file__).resolve().parent/'tools/blender/blender', Path(__file__).resolve().parent/'tools/blender/blender.exe') if p.is_file()), None) or shutil.which('blender')
    if not blender:
        raise RuntimeError('Blender is required.')
    mesh = o3d.io.read_triangle_mesh(str(args.r)); v = np.asarray(mesh.vertices); f = np.asarray(mesh.triangles)
    if not len(f): raise ValueError('Input must be an isolated triangle mesh.')
    metadata = args.r.with_suffix('.support.json')
    support = json.loads(metadata.read_text()) if metadata.is_file() and metadata.stat().st_mtime_ns >= args.r.stat().st_mtime_ns else None
    v, f, synthetic = close_flat_base(v, f, support)
    with tempfile.TemporaryDirectory(prefix='.base_export_', dir=output_dir) as name:
        folder = Path(name); texture_obj(v, f, synthetic, args.s.resolve(), folder, colmap)
        def native(path):
            return subprocess.check_output(['wslpath', '-w', str(path.resolve())], text=True).strip() if blender.lower().endswith('.exe') else str(path.resolve())
        subprocess.run([blender, '-b', '-t', '8', '--python', native(Path(__file__)), '--', '--blender-export', native(folder/'mesh.obj'), native(output), native(metadata)], check=True)
    if not output.is_file(): raise RuntimeError('FBX export failed.')
    print('Saved:', output, flush=True)


if __name__ == '__main__':
    if '--blender-export' in sys.argv:
        i = sys.argv.index('--blender-export'); blender_export(sys.argv[i+1], sys.argv[i+2], sys.argv[i+3])
    else:
        main()

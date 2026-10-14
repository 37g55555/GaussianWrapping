#!/usr/bin/env python3

import argparse
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


def close_flat_base(vertices, faces, up):
    import numpy as np
    import open3d as o3d
    import trimesh
    import mapbox_earcut
    from trimesh.intersections import slice_faces_plane
    groups, _ = boundary_groups(faces)
    if not groups:
        print('Already closed; no open base to repair.', flush=True)
        return vertices, faces, np.zeros(len(faces), bool)
    height = vertices@up; size = float(np.ptp(height)); epsilon = max(size*1e-6, 1e-10)
    ids = np.unique(np.concatenate(groups)); ids = ids[height[ids] < height.min()+size*.08]
    if len(ids) < 3:
        raise ValueError('No open boundary near the bottom.')
    points = vertices[ids]; plane = None
    o3d.utility.random.seed(44)
    for _ in range(4):
        if len(points) < 3:
            break
        eq, inliers = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).segment_plane(epsilon, 3, 1200)
        normal = np.asarray(eq[:3]); offset = float(eq[3])
        if normal@up < 0:
            normal, offset = -normal, -offset
        if normal@up > .90:
            plane = normal, offset; break
        points = np.delete(points, inliers, axis=0)
    if plane is None:
        raise ValueError('Cannot identify the open base plane safely.')
    normal, offset = plane; height = vertices@normal+offset
    bottom = [g for g in groups if height[g].max() < size*.04 and height[g].min() > -size*.01]
    if not bottom:
        raise ValueError('Bottom boundary is too irregular to repair safely.')
    bottom_ids = np.concatenate(bottom)
    adjacent = np.isin(faces, bottom_ids).any(1)
    upper = height[faces[adjacent]].max(1)
    low_triangles = upper[upper <= size*.04+epsilon]
    trim = max(max(float(height[g].max()) for g in bottom), float(low_triangles.max()) if len(low_triangles) else 0)+epsilon*5
    v, f, _ = slice_faces_plane(vertices, faces, normal, (trim-offset)*normal)
    _, first, inverse = np.unique(np.round(v, 10), axis=0, return_index=True, return_inverse=True)
    v, f = v[first], inverse[f]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])]
    used = np.unique(f); index = np.full(len(v), -1); index[used] = np.arange(len(used)); v, f = v[used], index[f]
    groups, adjacency = boundary_groups(f)
    groups = [g for g in groups if np.max(np.abs(v[g]@normal+offset-trim)) < epsilon*10]
    if not groups:
        raise ValueError('No clean cut boundary found.')
    original = v.copy(); loops = []; walls = []; caps = []; added = []; total = len(v)
    # Project the actual cut contour onto the base plane; XY outline is kept.
    for group in groups:
        if any(len(adjacency[int(i)]) != 2 for i in group):
            raise ValueError('Cut boundary is not a simple loop.')
        order = [int(group[0])]; visited = set(order); previous = -1; current = order[0]
        while True:
            neighbors = adjacency[current]; nxt = neighbors[0] if neighbors[0] != previous else neighbors[1]
            if nxt == order[0]:
                break
            if nxt in visited:
                raise ValueError('Self-touching cut boundary.')
            order.append(nxt); visited.add(nxt); previous, current = current, nxt
        order = np.asarray(order); ring = v[order]-(v[order]@normal+offset)[:, None]*normal
        lower = np.arange(total, total+len(ring)); total += len(ring); added.append(ring); loops.append(lower)
        for a, b, c, d in zip(order, np.roll(order, -1), lower, np.roll(lower, -1)):
            walls.extend([[int(a), int(c), int(d)], [int(a), int(d), int(b)]])
    v = np.vstack([v, *added])
    axis = np.cross(normal, [1, 0, 0] if abs(normal[0]) < .8 else [0, 1, 0]); axis /= np.linalg.norm(axis)
    xy = v@np.column_stack([axis, np.cross(normal, axis)])
    def contains(polygon, point):
        a = polygon; b = np.roll(a, -1, axis=0)
        crosses = (a[:, 1] > point[1]) != (b[:, 1] > point[1])
        active = np.flatnonzero(crosses)
        x = a[active, 0]+(point[1]-a[active, 1])*(b[active, 0]-a[active, 0])/(b[active, 1]-a[active, 1])
        return bool(np.count_nonzero(x > point[0]) % 2)
    areas = [abs(np.sum(xy[g, 0]*np.roll(xy[g, 1], -1)-xy[g, 1]*np.roll(xy[g, 0], -1))) for g in loops]
    parents = []
    for i, ring in enumerate(loops):
        enclosing = [j for j, outer in enumerate(loops) if areas[j] > areas[i] and contains(xy[outer], xy[ring[0]])]
        parents.append(min(enclosing, key=lambda j:areas[j]) if enclosing else -1)
    depth = []
    for i in range(len(loops)):
        n = 0; p = parents[i]
        while p >= 0:
            n += 1; p = parents[p]
        depth.append(n)
    # Nested rings become holes in the polygon, rather than overlapping caps.
    for i, outer in enumerate(loops):
        if depth[i] % 2:
            continue
        rings = [outer]+[loops[j] for j in range(len(loops)) if parents[j] == i]
        ids = np.concatenate(rings); ends = np.cumsum([len(g) for g in rings]).astype(np.uint32)
        triangles = mapbox_earcut.triangulate_float64(np.ascontiguousarray(xy[ids]), ends).reshape(-1, 3)
        triangles = ids[triangles]
        # Earcut can omit collinear boundary vertices, especially on holes.
        # Split those cap edges so every original boundary edge is paired.
        cap_edges = np.sort(np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
        cap_edges, counts = np.unique(cap_edges, axis=0, return_counts=True)
        lookup = {int(p):(ring, k) for ring in rings for k,p in enumerate(ring)}; chains = {}
        for a,b in cap_edges[counts == 1]:
            ring,k = lookup[int(a)]; other,j = lookup[int(b)]
            if ring is not other or (k-j)%len(ring) in (1, len(ring)-1): continue
            for step in (1,-1):
                indices = [k]; position = (k+step)%len(ring)
                while position != j: indices.append(position); position = (position+step)%len(ring)
                chain = ring[indices+[j]]; delta = xy[b]-xy[a]; length = float(delta@delta)
                t = (xy[chain]-xy[a])@delta/max(length,1e-30)
                error = np.linalg.norm(xy[chain]-xy[a]-t[:,None]*delta,axis=1)
                if len(chain)>2 and error.max()<size*1e-8 and t.min()>-1e-8 and t.max()<1+1e-8:
                    chains[(int(a),int(b))] = chain; break
        pending = triangles.tolist()
        while pending:
            triangle = pending.pop()
            for k in range(3):
                a,b,c = triangle[k],triangle[(k+1)%3],triangle[(k+2)%3]
                chain = chains.get(tuple(sorted((a,b))))
                if chain is not None:
                    chain = chain if chain[0] == a else chain[::-1]
                    pending.extend([[int(x),int(y),c] for x,y in zip(chain,chain[1:])]); break
            else: caps.append(triangle)
    new_faces = np.asarray(walls+caps, dtype=np.int64); final_faces = np.vstack([f, new_faces])
    mesh = trimesh.Trimesh(v, final_faces, process=False); trimesh.repair.fix_normals(mesh, multibody=True)
    faces = np.asarray(mesh.faces); synthetic = np.r_[np.zeros(len(f), bool), np.ones(len(new_faces), bool)]
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    if ((count != 2) & np.isin(unique, np.concatenate(loops)).any(1)).any():
        raise ValueError('Base triangulation did not close cleanly.')
    assert np.array_equal(v[:len(original)], original)
    print('Base repaired; trim height:', trim, '; new faces:', len(new_faces), flush=True)
    return v, faces, synthetic


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
    dist, mapping = cKDTree(v).query(pv)
    if dist[np.unique(pf)].max() > 1e-6:
        raise ValueError('Texture mapping changed mesh coordinates.')
    photo_faces = mapping[pf]; image = Image.open(textured/'texture.png').convert('RGB')
    # Reserve a narrow atlas strip for the new base, using nearby body color.
    base = f[synthetic]; width = 4096; strip = 128 if len(base) else 0; pixels = np.asarray(image)
    if len(base):
        from scipy.spatial import cKDTree
        near = cKDTree(v[np.unique(base)]).query(v[photo_faces].mean(1))[0]
        chosen = np.argsort(near)[:min(512, len(near))]; centers = uv[chosen].mean(1)
        x = np.clip((centers[:, 0]*image.width).astype(int), 0, image.width-1)
        y = np.clip(((1-centers[:, 1])*image.height).astype(int), 0, image.height-1)
        color = tuple(np.median(pixels[y, x], axis=0).astype('uint8'))
    else:
        color = (127, 127, 127)
    atlas = Image.new('RGB', (width, width), color); atlas.paste(image.resize((width, width-strip), Image.Resampling.LANCZOS), (0, 0))
    atlas.save(folder/'atlas.png'); uv[:, :, 1] = strip/width+uv[:, :, 1]*(1-strip/width)
    if len(base):
        points = v[np.unique(base)]; _, vectors = np.linalg.eigh(np.cov(points.T)); xy = v@vectors[:, -2:]
        low, high = xy[np.unique(base)].min(0), xy[np.unique(base)].max(0)
        buv = (xy[base]-low)/np.maximum(high-low, 1e-12)
        buv[:, :, 0] = .01+.98*buv[:, :, 0]; buv[:, :, 1] = strip/width*(.05+.90*buv[:, :, 1])
        uv = np.concatenate([uv, buv])
    triangles = np.vstack([photo_faces, base]); (folder/'mesh.mtl').write_text('newmtl Surface\nKd 1 1 1\nmap_Kd atlas.png\n\nnewmtl FlatBase\nKd 1 1 1\nmap_Kd atlas.png\n')
    with (folder/'mesh.obj').open('w') as handle:
        handle.write('mtllib mesh.mtl\nusemtl Surface\n')
        for p in v: handle.write('v %.9g %.9g %.9g\n'%tuple(p))
        for p in uv.reshape(-1, 2): handle.write('vt %.9g %.9g\n'%tuple(p))
        for i, triangle in enumerate(triangles):
            if i == len(photo_faces): handle.write('usemtl FlatBase\n')
            handle.write('f '+' '.join('%d/%d'%(int(j)+1, i*3+k+1) for k,j in enumerate(triangle))+'\n')


def blender_export(obj_path, output):
    import bpy
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.obj_import(filepath=obj_path, forward_axis='Y', up_axis='Z')
    obj = bpy.context.view_layer.objects.active
    for p in obj.data.polygons: p.use_smooth = obj.data.materials[p.material_index].name != 'FlatBase'
    for material in obj.data.materials:
        for node in material.node_tree.nodes:
            image = getattr(node, 'image', None)
            if image: image.pack()
    bpy.ops.export_scene.fbx(filepath=output, use_selection=True, object_types={'MESH'}, path_mode='COPY', embed_textures=True, add_leaf_bones=False)


def main():
    import numpy as np
    import open3d as o3d
    from isolate_supported_object import camera_focus
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-r', type=Path, required=True)
    parser.add_argument('-s', type=Path, required=True)
    parser.add_argument('-m', type=Path)
    args = parser.parse_args(); output_dir = (args.m or args.r.parent).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir/(args.r.stem+'_repaired.fbx')
    colmap = find_colmap()
    blender = next((str(p) for p in (Path(__file__).resolve().parent/'tools/blender/blender', Path(__file__).resolve().parent/'tools/blender/blender.exe') if p.is_file()), None) or shutil.which('blender')
    if not colmap or not blender:
        raise RuntimeError('COLMAP mesh_texturer and Blender are required.')
    _, up, _ = camera_focus(args.s)
    mesh = o3d.io.read_triangle_mesh(str(args.r)); v = np.asarray(mesh.vertices); f = np.asarray(mesh.triangles)
    if not len(f): raise ValueError('Input must be an isolated triangle mesh.')
    v, f, synthetic = close_flat_base(v, f, up)
    with tempfile.TemporaryDirectory(prefix='.base_export_', dir=output_dir) as name:
        folder = Path(name); texture_obj(v, f, synthetic, args.s.resolve(), folder, colmap)
        def native(path):
            return subprocess.check_output(['wslpath', '-w', str(path.resolve())], text=True).strip() if blender.lower().endswith('.exe') else str(path.resolve())
        subprocess.run([blender, '-b', '-t', '8', '--python', native(Path(__file__)), '--', '--blender-export', native(folder/'mesh.obj'), native(output)], check=True)
    if not output.is_file(): raise RuntimeError('FBX export failed.')
    print('Saved:', output, flush=True)


if __name__ == '__main__':
    if '--blender-export' in sys.argv:
        i = sys.argv.index('--blender-export'); blender_export(sys.argv[i+1], sys.argv[i+2])
    else:
        main()

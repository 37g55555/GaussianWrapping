#!/usr/bin/env python3

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import open3d as o3d
from trimesh.intersections import slice_faces_plane


def unit(x):
    n = np.linalg.norm(x)
    if n < 1e-10:
        raise ValueError('Cannot determine camera up direction.')
    return x / n


def rotation(q):
    w, x, y, z = unit(np.asarray(q, dtype=float))
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                     [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])


def camera_focus(colmap):
    paths = [colmap, colmap/'0', colmap/'sparse', colmap/'sparse/0']
    path = next((p/n for p in paths for n in ('images.bin', 'images.txt')
                 if (p/n).is_file()), None)
    if path is None:
        raise ValueError('COLMAP images.bin or images.txt not found.')
    poses = []
    if path.suffix == '.bin':
        with path.open('rb') as handle:
            def read(fmt):
                return struct.unpack('<'+fmt, handle.read(struct.calcsize('<'+fmt)))
            for _ in range(read('Q')[0]):
                row = read('idddddddi')
                poses.append(row[1:8])
                while True:
                    char = handle.read(1)
                    if char == b'\0':
                        break
                    if not char:
                        raise EOFError('Truncated COLMAP image record.')
                handle.seek(read('Q')[0]*24, 1)
    else:
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip() or line.lstrip().startswith('#'):
                    continue
                row = line.split()
                poses.append([float(x) for x in row[1:8]])
                # The observations line may be empty. It is still one line.
                next(handle, '')
    if len(poses) < 3:
        raise ValueError('At least three registered cameras are required.')
    centers, axes, ups = [], [], []
    for pose in poses:
        r = rotation(pose[:4]); t = np.asarray(pose[4:])
        centers.append(-r.T@t); axes.append(r.T@[0, 0, 1]); ups.append(r.T@[0, -1, 0])
    centers, axes, ups = map(np.asarray, (centers, axes, ups))
    keep = np.ones(len(centers), dtype=bool)
    for _ in range(6):
        p = np.eye(3)-axes[keep, :, None]*axes[keep, None, :]
        a = p.sum(0)
        if np.linalg.cond(a) > 1e8:
            raise ValueError('Camera axes do not constrain a 3D target center.')
        focus = np.linalg.solve(a, np.einsum('nij,nj->i', p, centers[keep]))
        error = np.linalg.norm(np.cross(focus-centers, axes), axis=1)
        median = np.median(error[keep]); mad = np.median(abs(error[keep]-median))
        good = error <= median+3.5*1.4826*max(mad, 1e-10)
        if good.sum() < max(3, len(centers)//3) or np.array_equal(good, keep):
            break
        keep = good
    up = unit(np.median(ups[keep], axis=0))
    scale = float(np.median(np.linalg.norm(centers[keep]-focus, axis=1)))
    if scale < 1e-8:
        raise ValueError('Degenerate camera arrangement.')
    return focus, up, scale


def support_plane(vertices, focus, up, scale):
    delta = vertices-focus; h = delta@up
    radial = np.linalg.norm(delta-np.outer(h, up), axis=1)
    rng = np.random.default_rng(44)
    tolerance = scale*.0015
    candidates = []
    for attempt, (radius, depth) in enumerate(((.45, .5), (.65, 1.0)), 1):
        points = vertices[(h < -scale*.005) & (h > -scale*depth) & (radial < scale*radius)]
        if len(points) < 200:
            continue
        points = points[rng.choice(len(points), min(60000, len(points)), replace=False)]
        print(f'[Support] search {attempt}/2: {len(points)} samples', flush=True)
        o3d.utility.random.seed(44)
        for _ in range(8):
            if len(points) < 200:
                break
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            eq, ids = cloud.segment_plane(tolerance, 3, 1600)
            normal = unit(np.asarray(eq[:3])); offset = float(eq[3])
            if normal@up < 0:
                normal, offset = -normal, -offset
            height = float(normal@focus+offset)
            if normal@up > .90 and tolerance < height < scale*depth and len(ids) >= 200:
                patch = points[ids]; projected = patch-np.outer(patch@normal, normal)
                span = np.linalg.norm(np.ptp(projected, axis=0))
                if span > scale*.12:
                    candidates.append((normal, offset, height, len(ids)))
            mask = np.ones(len(points), bool); mask[ids] = False; points = points[mask]
        if candidates:
            break
        if attempt == 1:
            print('[Support] no plane nearby; expanding search below the focus.', flush=True)
    if not candidates:
        raise ValueError('No substantial horizontal support plane found after expanding the search.')
    largest = max(c[3] for c in candidates)
    normal, offset, _, _ = min((c for c in candidates if c[3] >= largest*.2), key=lambda c:c[2])
    return normal, offset, tolerance


def isolate(vertices, faces, focus, normal, offset, scale, tolerance):
    v, f, _ = slice_faces_plane(vertices, faces, normal, -offset*normal)
    # Exact coordinate welding preserves positions and joins duplicate indices.
    v, mapping = np.unique(v, axis=0, return_inverse=True); f = mapping[f]
    distinct = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])
    f = f[distinct]
    height = v@normal+offset
    near = np.flatnonzero(height[f].max(1) < tolerance*1.5)
    xyz = v[f[near]]
    normals = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
    planar = abs(normals@normal) > .92*np.linalg.norm(normals, axis=1)
    keep = np.ones(len(f), bool); keep[near[planar]] = False; f = f[keep]
    if not len(f):
        raise ValueError('No geometry remains above the support plane.')
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f.astype(np.int32)))
    labels, counts, _ = mesh.cluster_connected_triangles(); labels = np.asarray(labels)
    # Local surface area votes, rather than component size or the nearest
    # isolated speck, identify the object around the camera focus.
    scores = np.zeros(len(counts))
    for start in range(0, len(f), 200000):
        ids = f[start:start+200000]; xyz = v[ids]; centers = xyz.mean(1)
        area = np.linalg.norm(np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]), axis=1)*.5
        weights = area*np.exp(-np.sum((centers-focus)**2, axis=1)/(scale*.15)**2)
        weights[height[ids].max(1) <= tolerance*2] = 0
        scores += np.bincount(labels[start:start+len(ids)], weights=weights, minlength=len(scores))
    if not np.isfinite(scores).all() or scores.max() <= 1e-12*scale**2:
        raise ValueError('No target component near the camera focus.')
    chosen = int(scores.argmax()); f = f[labels == chosen]
    used = np.unique(f); mapping = np.full(len(v), -1, dtype=np.int64); mapping[used] = np.arange(len(used))
    print('Support plane:', [*normal.tolist(), offset], flush=True)
    print('Components:', len(counts), '-> 1; retained faces:', len(f), flush=True)
    return v[used], mapping[f]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-r', required=True, type=Path, help='Input triangle mesh')
    parser.add_argument('-s', required=True, type=Path, help='dense or sparse COLMAP directory')
    parser.add_argument('-m', type=Path, help='Output directory (default: input mesh directory)')
    args = parser.parse_args()
    output_dir = args.m if args.m is not None else args.r.parent
    output = output_dir/(args.r.stem+'_post.ply')
    if output_dir.exists() and not output_dir.is_dir():
        parser.error('-m must be a directory')
    if args.r.resolve() == output.resolve():
        parser.error('Output must differ from the original input mesh.')
    focus, up, scale = camera_focus(args.s)
    print('Camera focus:', focus.tolist(), flush=True)
    source = o3d.io.read_triangle_mesh(str(args.r))
    if not len(source.triangles):
        raise ValueError('Input is not a triangle mesh.')
    v = np.asarray(source.vertices); f = np.asarray(source.triangles)
    normal, offset, tolerance = support_plane(v, focus, up, scale)
    residual = abs(v@normal+offset)
    residual = residual[residual < tolerance]
    noise = max(float(np.percentile(residual, 90)) if len(residual) else tolerance, scale*1e-7)
    support = {'plane': [*normal.tolist(), float(offset)], 'noise': noise}
    v, f = isolate(v, f, focus, normal, offset, scale, tolerance)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f.astype(np.int32)))
    if not o3d.io.write_triangle_mesh(str(output), result, write_vertex_normals=False):
        raise OSError('Failed to write output PLY.')
    output.with_suffix('.support.json').write_text(json.dumps(support, indent=2))
    print('Saved:', output, flush=True)


if __name__ == '__main__':
    main()

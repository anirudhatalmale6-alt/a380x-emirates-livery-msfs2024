"""Parse the paintkit OBJ into per-material numpy arrays and cache them as .npz.

Produces, per material:
    tri_v  (N,3,3) float32  triangle vertex positions in aircraft space
    tri_uv (N,3,2) float32  matching UVs
"""
import sys, os, numpy as np

OBJ = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)

verts = []
uvs = []
mats = {}          # material -> list of (vi,vti) triples
cur = None

with open(OBJ, 'r', errors='replace') as f:
    for line in f:
        if line.startswith('v '):
            _, a, b, c = line.split()
            verts.append((float(a), float(b), float(c)))
        elif line.startswith('vt '):
            p = line.split()
            uvs.append((float(p[1]), float(p[2])))
        elif line.startswith('usemtl'):
            cur = line.split(None, 1)[1].strip()
            mats.setdefault(cur, [])
        elif line.startswith('f '):
            p = line.split()[1:]
            idx = []
            for tok in p:
                bits = tok.split('/')
                vi = int(bits[0]) - 1
                vti = int(bits[1]) - 1 if len(bits) > 1 and bits[1] else -1
                idx.append((vi, vti))
            # fan-triangulate
            for k in range(1, len(idx) - 1):
                mats[cur].append((idx[0], idx[k], idx[k + 1]))

V = np.array(verts, dtype=np.float32)
T = np.array(uvs, dtype=np.float32)
print('verts', V.shape, 'uvs', T.shape)
print('bbox min', V.min(0), 'max', V.max(0))

for m, tris in mats.items():
    if not tris:
        continue
    a = np.array(tris, dtype=np.int64)      # (N,3,2)
    tri_v = V[a[:, :, 0]]                   # (N,3,3)
    tri_uv = T[a[:, :, 1]]                  # (N,3,2)
    np.savez_compressed(os.path.join(OUT, m + '.npz'), tri_v=tri_v, tri_uv=tri_uv)
    print('%-42s tris=%7d  vmin=%s vmax=%s' % (m, len(tris),
          np.round(tri_v.reshape(-1, 3).min(0), 2), np.round(tri_v.reshape(-1, 3).max(0), 2)))

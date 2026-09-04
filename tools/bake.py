"""Bake UV -> 3D position for a material sheet.

Rasterises every triangle in UV space and writes, per texel, the interpolated
aircraft-space position (x lateral, y up, z nose-positive) plus a coverage mask.
Also writes a false-colour debug image so the layout can be eyeballed:
    R = z (nose bright)   G = y (top bright)   B = |x| (outboard bright)
"""
import sys, os, numpy as np
from PIL import Image

MAT = sys.argv[1]
SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
MESH = 'mesh'
OUT = 'bake'
os.makedirs(OUT, exist_ok=True)

d = np.load(os.path.join(MESH, MAT + '.npz'))
tri_v = d['tri_v'].astype(np.float64)      # (N,3,3)
tri_uv = d['tri_uv'].astype(np.float64)    # (N,3,2)

# UV -> pixel. OBJ v is bottom-up, image is top-down.
px = tri_uv[:, :, 0] * SIZE
py = (1.0 - tri_uv[:, :, 1]) * SIZE

pos = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
mask = np.zeros((SIZE, SIZE), dtype=bool)

x0 = np.floor(px.min(1)).astype(int)
x1 = np.ceil(px.max(1)).astype(int)
y0 = np.floor(py.min(1)).astype(int)
y1 = np.ceil(py.max(1)).astype(int)
np.clip(x0, 0, SIZE - 1, out=x0); np.clip(x1, 0, SIZE, out=x1)
np.clip(y0, 0, SIZE - 1, out=y0); np.clip(y1, 0, SIZE, out=y1)

N = len(tri_v)
for i in range(N):
    ax, ay = px[i, 0], py[i, 0]
    bx, by = px[i, 1], py[i, 1]
    cx, cy = px[i, 2], py[i, 2]
    den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(den) < 1e-12:
        continue
    xs = np.arange(x0[i], max(x1[i], x0[i] + 1)) + 0.5
    ys = np.arange(y0[i], max(y1[i], y0[i] + 1)) + 0.5
    if xs.size == 0 or ys.size == 0:
        continue
    X, Y = np.meshgrid(xs, ys)
    l1 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / den
    l2 = ((cy - ay) * (X - cx) + (ax - cx) * (Y - cy)) / den
    l3 = 1.0 - l1 - l2
    inside = (l1 >= -0.002) & (l2 >= -0.002) & (l3 >= -0.002)
    if not inside.any():
        continue
    p = (l1[..., None] * tri_v[i, 0] + l2[..., None] * tri_v[i, 1] + l3[..., None] * tri_v[i, 2])
    sub_pos = pos[y0[i]:y0[i] + len(ys), x0[i]:x0[i] + len(xs)]
    sub_msk = mask[y0[i]:y0[i] + len(ys), x0[i]:x0[i] + len(xs)]
    sub_pos[inside] = p[inside].astype(np.float32)
    sub_msk[inside] = True
    if i % 20000 == 0:
        print(' ', i, '/', N, flush=True)

np.savez_compressed(os.path.join(OUT, '%s_%d.npz' % (MAT, SIZE)), pos=pos, mask=mask)

# debug colour map
def norm(a, lo, hi):
    return np.clip((a - lo) / (hi - lo), 0, 1)

v = tri_v.reshape(-1, 3)
dbg = np.zeros((SIZE, SIZE, 3), np.uint8)
dbg[..., 0] = (norm(pos[..., 2], v[:, 2].min(), v[:, 2].max()) * 255)
dbg[..., 1] = (norm(pos[..., 1], v[:, 1].min(), v[:, 1].max()) * 255)
dbg[..., 2] = (norm(np.abs(pos[..., 0]), 0, np.abs(v[:, 0]).max()) * 255)
dbg[~mask] = 0
Image.fromarray(dbg).resize((1000, 1000), Image.NEAREST).save(os.path.join(OUT, '%s_dbg.png' % MAT))
print('%s: %.1f%% covered' % (MAT, 100.0 * mask.mean()))

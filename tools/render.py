"""Software renderer for the A380X paintkit mesh.

Rasterises the OBJ with a z-buffer and samples whatever textures I have painted,
so a livery can be checked visually without the simulator.

usage: render.py <outfile> <az> <el> <dist> [texdir] [size]
   az  degrees around the aircraft (0 = looking from the nose, 90 = from port)
   el  degrees above the horizon
"""
import sys, os, glob, math
import numpy as np
from PIL import Image

OUT = sys.argv[1]
AZ = float(sys.argv[2])
EL = float(sys.argv[3])
DIST = float(sys.argv[4])
TEXDIR = sys.argv[5] if len(sys.argv) > 5 else 'paint'
W = int(sys.argv[6]) if len(sys.argv) > 6 else 1280
H = int(W * 0.62)
MESH = 'mesh'
TEXSIZE = 2048

CENTER = np.array([0.0, 3.0, -2.0])
FOV = 26.0
LIGHT = np.array([0.35, 0.78, 0.52])
LIGHT = LIGHT / np.linalg.norm(LIGHT)

# ---------------- camera ----------------
a = math.radians(AZ)
e = math.radians(EL)
eye = CENTER + DIST * np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])
fwd = CENTER - eye
fwd /= np.linalg.norm(fwd)
right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
right /= np.linalg.norm(right)
up = np.cross(right, fwd)
Rm = np.stack([right, up, -fwd])            # world -> camera rotation
f = 1.0 / math.tan(math.radians(FOV) / 2)

color = np.zeros((H, W, 3), np.float32)
zbuf = np.full((H, W), 1e30, np.float32)

# sky gradient
gy = np.linspace(0, 1, H)[:, None]
color[..., 0] = 0.42 + 0.34 * gy
color[..., 1] = 0.58 + 0.28 * gy
color[..., 2] = 0.82 + 0.14 * gy

DEFAULT = np.full((8, 8, 3), 200, np.uint8)

def load_tex(mat):
    for ext in ('png', 'jpg'):
        p = os.path.join(TEXDIR, mat + '.' + ext)
        if os.path.exists(p):
            im = Image.open(p).convert('RGB')
            if im.width > TEXSIZE:
                im = im.resize((TEXSIZE, TEXSIZE), Image.LANCZOS)
            return np.asarray(im, np.uint8)
    return DEFAULT

mats = sorted(glob.glob(os.path.join(MESH, '*.npz')))
for mp in mats:
    mat = os.path.basename(mp)[:-4]
    d = np.load(mp)
    tri_v = d['tri_v'].astype(np.float64)
    tri_uv = d['tri_uv'].astype(np.float64)
    tex = load_tex(mat)
    th, tw = tex.shape[:2]

    # normals + backface cull in world space
    e1 = tri_v[:, 1] - tri_v[:, 0]
    e2 = tri_v[:, 2] - tri_v[:, 0]
    nrm = np.cross(e1, e2)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    ln[ln == 0] = 1
    nrm /= ln

    cam = (tri_v - eye) @ Rm.T            # (N,3,3) camera space
    z = -cam[:, :, 2]
    ok = (z > 0.05).all(1)

    ctr = tri_v.mean(1)
    view = ctr - eye
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    facing = (nrm * view).sum(1) < 0
    ok &= facing
    if not ok.any():
        continue

    tri_v_o, tri_uv_o, cam, z, nrm = tri_v[ok], tri_uv[ok], cam[ok], z[ok], nrm[ok]

    sx = (cam[:, :, 0] * f / z) * (H / 2) + W / 2
    sy = (-cam[:, :, 1] * f / z) * (H / 2) + H / 2

    lam = np.clip((nrm * LIGHT).sum(1), 0, 1)
    shade = (0.55 + 0.45 * lam).astype(np.float32)

    x0 = np.clip(np.floor(sx.min(1)).astype(int), 0, W - 1)
    x1 = np.clip(np.ceil(sx.max(1)).astype(int) + 1, 0, W)
    y0 = np.clip(np.floor(sy.min(1)).astype(int), 0, H - 1)
    y1 = np.clip(np.ceil(sy.max(1)).astype(int) + 1, 0, H)
    big = ((x1 - x0) * (y1 - y0)) > 0

    for i in np.nonzero(big)[0]:
        ax, ay = sx[i, 0], sy[i, 0]
        bx, by = sx[i, 1], sy[i, 1]
        cx, cy = sx[i, 2], sy[i, 2]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-9:
            continue
        xs = np.arange(x0[i], x1[i]) + 0.5
        ys = np.arange(y0[i], y1[i]) + 0.5
        X, Y = np.meshgrid(xs, ys)
        l1 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / den
        l2 = ((cy - ay) * (X - cx) + (ax - cx) * (Y - cy)) / den
        l3 = 1.0 - l1 - l2
        ins = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not ins.any():
            continue
        zz = l1 * z[i, 0] + l2 * z[i, 1] + l3 * z[i, 2]
        sub_z = zbuf[y0[i]:y1[i], x0[i]:x1[i]]
        win = ins & (zz < sub_z)
        if not win.any():
            continue
        u = l1 * tri_uv_o[i, 0, 0] + l2 * tri_uv_o[i, 1, 0] + l3 * tri_uv_o[i, 2, 0]
        v = l1 * tri_uv_o[i, 0, 1] + l2 * tri_uv_o[i, 1, 1] + l3 * tri_uv_o[i, 2, 1]
        tu = np.clip((u * tw).astype(int), 0, tw - 1)
        tv = np.clip(((1 - v) * th).astype(int), 0, th - 1)
        px = tex[tv[win], tu[win]].astype(np.float32) / 255.0
        sub_c = color[y0[i]:y1[i], x0[i]:x1[i]]
        sub_c[win] = px * shade[i]
        sub_z[win] = zz[win]
    print('  drew', mat, flush=True)

img = (np.clip(color, 0, 1) * 255).astype(np.uint8)
Image.fromarray(img).save(OUT)
print('wrote', OUT, img.shape)

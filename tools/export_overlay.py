"""Pull the paintkit's own MARKINGS / SCUFFPLATES artwork out of the PSD as RGBA.

These carry the door outlines, scuff plates and stencils. My scheme only lays
down colour, so without these the aircraft comes out looking like a plastic toy.
"""
import sys, os, gc
import numpy as np
from PIL import Image
from psd_tools import PSDImage

PSD = sys.argv[1]
OUT = sys.argv[2]
SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
WANT = ('MARKINGS', 'SCUFFPLATES', 'SCUFF PLATES', 'WALKWAY')
W = H = 8192
os.makedirs(OUT, exist_ok=True)

psd = PSDImage.open(PSD)


def blit(canvas, layer):
    if layer.bbox == (0, 0, 0, 0):
        return
    a = layer.numpy()
    if a is None:
        return
    x1, y1, x2, y2 = layer.bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return
    a = a[:y2 - y1, :x2 - x1]
    if a.shape[2] == 3:
        a = np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], axis=2)
    src = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    del a
    alpha = src[:, :, 3:4].astype(np.uint16)
    dst = canvas[y1:y2, x1:x2]
    dst[:, :, :3] = ((src[:, :, :3].astype(np.uint16) * alpha +
                      dst[:, :, :3].astype(np.uint16) * (255 - alpha)) // 255).astype(np.uint8)
    dst[:, :, 3] = np.maximum(dst[:, :, 3], src[:, :, 3])
    del src, alpha
    gc.collect()


def walk(node, canvas):
    for l in node:
        if l.is_group():
            walk(l, canvas)
        else:
            blit(canvas, l)


for g in psd:
    if not g.is_group():
        continue
    sheet = g.name.strip().replace(' ', '')
    groups = [l for l in g if l.is_group() and l.name.strip().upper() in WANT]
    if not groups:
        continue
    canvas = np.zeros((H, W, 4), np.uint8)
    for grp in groups:
        walk(grp, canvas)
    im = Image.fromarray(canvas).resize((SIZE, SIZE), Image.LANCZOS)
    im.save(os.path.join(OUT, 'ov_%s.png' % sheet))
    cov = 100.0 * (canvas[..., 3] > 8).mean()
    print('%-14s %s  coverage %.2f%%' % (sheet, [x.name for x in groups], cov), flush=True)
    del canvas
    gc.collect()

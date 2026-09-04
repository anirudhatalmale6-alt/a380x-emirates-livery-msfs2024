import sys, os, glob
import numpy as np
from PIL import Image
import scheme as S
import importlib
importlib.reload(S)

SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
OUT = sys.argv[2] if len(sys.argv) > 2 else 'paint'
os.makedirs(OUT, exist_ok=True)

for f in sorted(glob.glob('bake/*_%d.npz' % SIZE)):
    mat = os.path.basename(f)[:-(len(str(SIZE)) + 5)]
    d = np.load(f)
    pos, mask = d['pos'], d['mask']
    c = S.scheme(mat, pos, mask)
    c = S.dilate_fill(c, mask, iters=max(6, SIZE // 256))

    # the paintkit's own door outlines / service-point stencils go on last
    ov = os.path.join('overlay', 'ov_%s.png' % mat.replace('A380X_', '').replace('A380_EXTERIOR_', ''))
    if os.path.exists(ov):
        o = np.asarray(Image.open(ov).convert('RGBA').resize((SIZE, SIZE), Image.LANCZOS), np.float32)
        a = o[..., 3:4] / 255.0
        c = c * (1 - a) + o[..., :3] * a
        print('  + markings overlay', flush=True)
    Image.fromarray(np.clip(c, 0, 255).astype(np.uint8)).save(os.path.join(OUT, mat + '.png'))
    print('painted', mat, flush=True)

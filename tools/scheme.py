"""Emirates A380 scheme, defined in aircraft space rather than in UV space.

Every texel knows its own (x lateral, y up, z nose-positive) position from the
bake, so the paint is a function of where it sits on the airframe. That way the
cheatlines and the red sweep stay continuous across five separate texture
sheets, which is the part that goes wrong if you draw them by hand per sheet.
"""
import numpy as np
import decals as D

WHITE      = (248, 247, 243)
GREY_BELLY = (214, 217, 219)
GREY_WING  = (170, 175, 178)
GREY_DARK  = (95, 100, 104)
RED        = (200, 16, 46)
GOLD       = (186, 144, 62)
GREEN      = (0, 115, 47)
BLACK      = (28, 28, 30)

FUSE_SHEETS = ('A380X_FUSE1', 'A380X_FUSE2', 'A380X_FUSE3', 'A380X_FUSE5')


def _fill(shape, rgb):
    out = np.empty(shape + (3,), np.float32)
    out[..., 0], out[..., 1], out[..., 2] = rgb
    return out


def red_sweep_top(z):
    """Top edge (y) of the red band that runs up the rear fuselage into the fin.

    Stays under the belly for most of the length, then climbs hard over the last
    few metres so the tailcone goes fully red into the fin root, as on the real
    aircraft. Exponent set by eye against the client's reference shots.
    """
    t = np.clip((-2.0 - z) / 28.0, 0.0, 1.0)
    return -2.6 + (t ** 2.6) * 9.5


def paint_fuselage(x, y, z, c):
    # light grey under the forward belly
    c[(y < -0.35) & (z > 4)] = GREY_BELLY
    # red sweep along the lower rear fuselage, climbing to swallow the tailcone
    c[y < red_sweep_top(z)] = RED
    return c


def paint_fin(x, y, z, c):
    """Vertical fin: red field with the UAE flag banding across the top."""
    fin = (y > 7.0) & (np.abs(x) < 4.6)
    t = np.clip((y - 7.0) / 13.2, 0, 1)
    c[fin] = RED
    band = fin & (t > 0.545) & (t <= 0.82)
    c[band] = WHITE
    # the black of the flag, broken into stripes the way Emirates render it
    s = (t - 0.545) / 0.275 * 5.0
    stripe = band & ((s % 1.0) < 0.33)
    c[stripe] = BLACK
    c[fin & (t > 0.82)] = GREEN
    # fin root fairing carries the fuselage red up; the stabilisers stay white
    root = (np.abs(x) < 2.8) & (y <= 7.0)
    c[root & (y < red_sweep_top(z))] = RED
    return c


def scheme(mat, pos, mask):
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    c = _fill(x.shape, WHITE)

    if mat in FUSE_SHEETS:
        c = paint_fuselage(x, y, z, c)
        c[np.abs(x) > 4.6] = WHITE           # flap-track fairings hanging off the sheet
        c = D.apply_fuselage_decals(c, x, y, z)

    elif mat == 'A380X_FUSE4':
        c = paint_fin(x, y, z, c)

    elif mat.startswith('A380_EXTERIOR_WING'):
        if 'FENCE' in mat:
            c[:] = RED
            c[y < 1.9] = WHITE
        else:
            c[:] = GREY_WING
            c[y < 1.2] = GREY_BELLY          # lower surface a touch lighter

    elif mat.startswith('A380_EXTERIOR_ENG'):
        c[:] = WHITE
        c[z < -2.2] = GREY_DARK              # exhaust cowl
        c[z > 12.0] = GREY_DARK              # intake lip
        c = D.apply_engine_decal(c, x, y, z, mask)

    elif mat == 'A380_EXTERTIOR_PYLON':
        c[:] = WHITE
        c[y < 0.4] = GREY_BELLY

    elif mat == 'A380_COCKPIT_WINDSHIELD_FRAME_4k':
        c[:] = GREY_DARK

    elif mat == 'A380X_ACCESSORIES':
        # mixed bag: gear doors, fairings, flap tracks. Treat by position.
        c[:] = WHITE
        wingish = np.abs(x) > 5.0
        c[wingish] = GREY_WING
        fus = np.abs(x) < 3.8            # fuselage proper, not the stab roots
        c[fus & (y < red_sweep_top(z))] = RED

    return c


def dilate_fill(c, mask, iters=10):
    """Bleed the painted colour outwards past each UV island edge.

    Texels outside the islands are never sampled head-on, but mipmapping and
    bilinear filtering do reach across the border, so leaving them unpainted
    puts a halo round every seam in the sim.
    """
    m = mask.copy()
    out = c.astype(np.float32).copy()
    for _ in range(iters):
        if m.all():
            break
        acc = np.zeros_like(out)
        cnt = np.zeros(m.shape, np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sm = np.roll(m, (dy, dx), (0, 1))
            acc += np.roll(out, (dy, dx), (0, 1)) * sm[..., None]
            cnt += sm
        new = (~m) & (cnt > 0)
        if not new.any():
            break
        out[new] = acc[new] / cnt[new][..., None]
        m |= new
    return out

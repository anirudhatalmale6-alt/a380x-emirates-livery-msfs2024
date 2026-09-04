"""Decal rasters (titles, wordmark, registration) and the projector that lays
them onto the airframe.

Decals are placed in aircraft coordinates - longitudinal station z and the angle
round the fuselage - not in UV space, so a title that crosses from one texture
sheet to the next stays continuous and the right way up.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LATIN = '/usr/share/fonts/opentype/urw-base35/P052-Bold.otf'   # closest match on this box to the Emirates wordmark
ARABIC = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
GOLD = (186, 144, 62)
DARKGREY = (74, 78, 82)

_cache = {}


def text_raster(txt, font_path, px, colour, squeeze=1.0, rtl=False, pad=2, track=0.0):
    """track: extra letter spacing as a fraction of the font size. The Emirates
    wordmark is noticeably wider than any stock face, so the Latin titles need
    some, or they come out short and stubby against the reference photos."""
    key = (txt, font_path, px, colour, squeeze, rtl, track)
    if key in _cache:
        return _cache[key]
    f = ImageFont.truetype(font_path, px)
    tmp = Image.new('L', (10, 10))
    kw = dict(font=f)
    if rtl:
        kw.update(direction='rtl', language='ar')
    box = ImageDraw.Draw(tmp).textbbox((0, 0), txt, **kw)
    extra = int(px * track)
    w = box[2] - box[0] + pad * 2 + extra * max(0, len(txt) - 1)
    h = box[3] - box[1] + pad * 2
    im = Image.new('RGBA', (w, h), colour + (0,))
    dr = ImageDraw.Draw(im)
    if extra and not rtl:
        cx = pad - box[0]
        for ch in txt:
            dr.text((cx, pad - box[1]), ch, fill=colour + (255,), **kw)
            cx += dr.textlength(ch, font=f) + extra
    else:
        dr.text((pad - box[0], pad - box[1]), txt, fill=colour + (255,), **kw)
    if squeeze != 1.0:
        im = im.resize((max(1, int(w * squeeze)), h), Image.LANCZOS)
    a = np.asarray(im, np.float32) / 255.0
    _cache[key] = a
    return a


def fuse_theta(x, y, yc=2.55):
    """Angle round the fuselage: 0 at the widest point, +90 at the crown."""
    return np.degrees(np.arctan2(y - yc, np.abs(x)))


def project(c, img, sel, x, z, theta, zA, zB, thA, thB):
    """Lay an RGBA raster onto the selected texels.

    Mirrored in z between the two sides so the text reads nose-to-tail on the
    port side and tail-to-nose on the starboard side, i.e. left-to-right on both.
    """
    ih, iw = img.shape[:2]
    L = float(zB - zA)
    for side in (1, -1):
        s = sel & ((x > 0) if side > 0 else (x < 0))
        if not s.any():
            continue
        zz, tt = z[s], theta[s]
        u = (zB - zz) / L if side > 0 else (zz - zA) / L
        v = (thB - tt) / float(thB - thA)
        good = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        if not good.any():
            continue
        px = np.clip((u[good] * iw).astype(int), 0, iw - 1)
        py = np.clip((v[good] * ih).astype(int), 0, ih - 1)
        samp = img[py, px]
        a = samp[:, 3:4]
        idx = np.nonzero(s.ravel())[0][np.nonzero(good)[0]]
        flat = c.reshape(-1, 3)
        flat[idx] = flat[idx] * (1 - a) + samp[:, :3] * 255.0 * a
    return c


FUSE_R = 3.9        # effective fuselage radius at the height the titles sit


def place(c, img, sel, x, z, theta, z_nose_end, th_bot, th_top, R=FUSE_R):
    """Anchor a decal at its nose-end station and let its own aspect ratio set
    how far aft it runs - otherwise the text gets stretched to fit the box."""
    ih, iw = img.shape[:2]
    arc = np.radians(th_top - th_bot) * R
    length = arc * (iw / float(ih))
    project(c, img, sel, x, z, theta, z_nose_end - length, z_nose_end, th_bot, th_top)
    return z_nose_end - length


def apply_fuselage_decals(c, x, y, z):
    th = fuse_theta(x, y)
    side = np.abs(x) > 1.2                      # skip the very top and keel

    # Main titles. The upper-deck window row measures out at y=4.82 on this
    # mesh, and on the real aircraft the cap height of the E sits just under it.
    t = text_raster('Emirates', LATIN, 300, GOLD, track=0.085)
    tail_of_titles = place(c, t, side, x, z, th, 25.5, 2.0, 38.0)

    # Arabic wordmark, aft of the titles and about half the height
    a = text_raster('الإمارات', ARABIC, 260, GOLD, rtl=True)
    place(c, a, side, x, z, th, tail_of_titles - 1.0, 11.0, 33.0)

    # registration on the rear fuselage
    r = text_raster('A6-EDY', LATIN, 150, DARKGREY, track=0.03)
    place(c, r, side, x, z, th, -13.0, 28.0, 36.0)
    return c


def apply_engine_decal(c, x, y, z, mask):
    """Gold wordmark on the outboard face of each nacelle.

    One sheet carries both engines on a wing, so the two nacelles are separated
    on |x| first and each one gets its own local centre.
    """
    ax = np.abs(x)
    live = mask & (ax > 1.0)
    if not live.any():
        return c
    lo, hi = ax[live].min(), ax[live].max()
    split = (lo + hi) / 2.0
    t = text_raster('الإمارات', ARABIC, 220, GOLD, rtl=True)
    for grp in (live & (ax < split), live & (ax >= split)):
        if not grp.any():
            continue
        xc = ax[grp].mean()
        outboard = grp & (ax > xc + 0.75)          # the face you see from outside
        if not outboard.any():
            continue
        u_side = 1 if x[outboard].mean() > 0 else -1
        ih, iw = t.shape[:2]
        yB, yA = -0.30, -1.55                      # top / bottom of the wordmark
        zB = 9.4
        zA = zB - (yB - yA) * (iw / float(ih))
        s = outboard
        zz, yy = z[s], y[s]
        u = (zB - zz) / (zB - zA) if u_side > 0 else (zz - zA) / (zB - zA)
        v = (yB - yy) / (yB - yA)
        good = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        if not good.any():
            continue
        ih, iw = t.shape[:2]
        px = np.clip((u[good] * iw).astype(int), 0, iw - 1)
        py = np.clip((v[good] * ih).astype(int), 0, ih - 1)
        samp = t[py, px]
        a = samp[:, 3:4]
        idx = np.nonzero(s.ravel())[0][np.nonzero(good)[0]]
        flat = c.reshape(-1, 3)
        flat[idx] = flat[idx] * (1 - a) + samp[:, :3] * 255.0 * a
    return c

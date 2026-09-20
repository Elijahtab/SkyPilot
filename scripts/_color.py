"""
Vehicle colour naming from pixels -- the exploratory, no-training baseline.

There are no colour labels anywhere in this project, so nothing here is learned.
The method is a weighted pixel vote over the vehicle's body, with three
corrections for what actually contaminates a traffic-cam box:

  1. corners and edges are road -> vote only inside a central ellipse
  2. road is still inside that ellipse on oblique views -> colours that are as
     common in a ring AROUND the box as inside it are down-weighted (histogram
     back-projection in Lab). A red car on grey asphalt keeps its red; the
     asphalt loses its vote.
  3. every vehicle has dark glass, tyres and shadow, and specular glare on
     glass reads as white -> dark and glare pixels vote at reduced weight

Chromatic colours win on a lower share than achromatic ones (CHROMA_SHARE),
because saturated pixels are rarely background while grey and black pixels
almost always partly are.

Limits worth stating: at night, under sodium light, or on an IR camera there is
no colour to recover, and silver vs grey vs white is decided by exposure as much
as paint. `confidence` is the winning weighted share, not a probability.

Usage:
    from _color import COLORS, name_color, white_balance_gains
    gains = white_balance_gains(frame_rgb)                 # once per frame
    color, conf, shares = name_color(frame_rgb, (x1, y1, x2, y2), gains)
"""

import cv2
import numpy as np

COLORS = ["white", "gray", "black", "red", "orange", "yellow", "green", "blue", "brown"]

# Box long side, native px, below which a colour answer is not attempted.
# Unmeasured below 48px -- see scripts/evaluation/explore_color.py.
MIN_COLOR_PX = 24

ELLIPSE      = 0.40     # semi-axes as a fraction of box width / height
RING_OUT     = 0.30     # background ring: box grown by this ...
RING_IN      = 0.05     # ... minus the box grown by this
BG_FLOOR     = 0.25     # never zero a pixel's vote entirely
DARK_WEIGHT  = 0.4      # glass, tyres, shadow: every vehicle has them
GLARE_WEIGHT = 0.5
CHROMA_SHARE = 0.25     # a chromatic colour wins with this weighted share

# achromatic / chromatic boundaries, HSV with S and V in 0..1, H in degrees
V_BLACK = 0.22
S_ACHRO = 0.22
S_BLUE  = 0.35          # shade and sky reflection tint white paint blue
# White is judged against the road around the vehicle, not an absolute level:
# the same white van reads V=0.9 at noon and V=0.55 under cloud.
WHITE_OVER_ROAD = 1.2
V_WHITE_RANGE   = (0.55, 0.80)
WB_GAIN_RANGE   = (0.7, 1.4)


def white_balance_gains(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Per-channel gains from the frame's near-neutral pixels (grey-world on road,
    kerb and roof). Traffic-cam frames are mostly asphalt, so this removes the
    camera's colour cast -- the cause of white vehicles reading blue. Compute
    once per frame and pass to name_color() for every box in it.
    """
    px = frame_rgb[::4, ::4].reshape(-1, 3).astype(np.float32) / 255.0
    mx, mn = px.max(1), px.min(1)
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    neutral = px[(s < 0.30) & (mx > 0.15) & (mx < 0.95)]
    if len(neutral) < 100:
        return np.ones(3, np.float32)
    m = neutral.mean(0)
    return np.clip(m.mean() / np.maximum(m, 1e-6), *WB_GAIN_RANGE).astype(np.float32)


def _hsv(rgb):
    rgb = rgb.astype(np.float32) / 255.0
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.select(
        [d == 0, mx == r, mx == g],
        [0.0, ((g - b) / np.maximum(d, 1e-6)) % 6, (b - r) / np.maximum(d, 1e-6) + 2],
        (r - g) / np.maximum(d, 1e-6) + 4) * 60.0
    return h, s, mx


def _names(h, s, v, v_white):
    """Per-pixel index into COLORS."""
    idx = np.full(h.shape, COLORS.index("gray"))
    chroma = (s >= S_ACHRO) & (v >= V_BLACK)
    chroma &= ~((h >= 170) & (h < 300) & (s < S_BLUE))   # faint blue is a tint, not paint
    idx[~chroma & (v >= v_white)] = COLORS.index("white")
    idx[v < V_BLACK] = COLORS.index("black")

    hue_bins = [                                   # (lo, hi, name)
        (0, 12, "red"), (12, 40, "orange"), (40, 75, "yellow"), (75, 170, "green"),
        (170, 300, "blue"), (300, 360, "red"),     # 40-75: lime-yellow buses too
    ]
    for lo, hi, name in hue_bins:
        idx[chroma & (h >= lo) & (h < hi)] = COLORS.index(name)
    # dark or washed-out orange is brown / tan / beige, not orange; washed-out
    # yellow is champagne / gold paint. Lane paint and school buses stay saturated.
    orange = chroma & (idx == COLORS.index("orange"))
    idx[orange & ((v < 0.55) | (s < 0.45))] = COLORS.index("brown")
    yellow = chroma & (idx == COLORS.index("yellow"))
    idx[yellow & (s < 0.40)] = COLORS.index("brown")
    return idx


def name_color(frame_rgb: np.ndarray, box, gains: np.ndarray = None):
    """
    frame_rgb: HxWx3 uint8 RGB. box: xyxy in pixels. gains: from
    white_balance_gains(frame_rgb); computed here if omitted, which is slow when
    called for many boxes in one frame.
    Returns (color, confidence, {color: weighted share}), or (None, 0.0, {}) if
    the box is too small or has no usable pixels.
    """
    H, W = frame_rgb.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if max(bw, bh) < MIN_COLOR_PX:
        return None, 0.0, {}
    if gains is None:
        gains = white_balance_gains(frame_rgb)

    # working window = the ring's outer edge, clamped to the frame
    ox1, oy1 = int(max(0, x1 - RING_OUT * bw)), int(max(0, y1 - RING_OUT * bh))
    ox2, oy2 = int(min(W, x2 + RING_OUT * bw)), int(min(H, y2 + RING_OUT * bh))
    win = frame_rgb[oy1:oy2, ox1:ox2]
    if win.size == 0:
        return None, 0.0, {}
    win = np.clip(win.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    yy, xx = np.mgrid[oy1:oy2, ox1:ox2] + 0.5

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    body = ((xx - cx) / (ELLIPSE * bw)) ** 2 + ((yy - cy) / (ELLIPSE * bh)) ** 2 <= 1
    inner = ((xx >= x1 - RING_IN * bw) & (xx < x2 + RING_IN * bw) &
             (yy >= y1 - RING_IN * bh) & (yy < y2 + RING_IN * bh))
    ring = ~inner
    if body.sum() < 20:
        return None, 0.0, {}

    # 2. background suppression: 8x8x8 Lab histogram back-projection
    lab = cv2.cvtColor(np.ascontiguousarray(win), cv2.COLOR_RGB2LAB)
    bins = (lab[..., 0] // 32).astype(int) * 64 + (lab[..., 1] // 32) * 8 + (lab[..., 2] // 32)
    w = np.ones(body.sum())
    if ring.sum() >= 20:
        hb = np.bincount(bins[body], minlength=512) / body.sum()
        hr = np.bincount(bins[ring], minlength=512) / ring.sum()
        p = hb / np.maximum(hb + hr, 1e-9)                  # 0.5 = as common outside as in
        w = np.clip(p[bins[body]] * 2 - 0.5, BG_FLOOR, 1.0)  # 0.5 -> 0.5, 0.75+ -> 1.0

    # 3. names, with dark and glare pixels voting at reduced weight
    road_v = np.median(win[ring].max(-1)) / 255.0 if ring.sum() >= 20 else 0.5
    v_white = float(np.clip(road_v * WHITE_OVER_ROAD, *V_WHITE_RANGE))
    h, s, v = _hsv(win[body])
    idx = _names(h, s, v, v_white)
    w = w * np.where(v < V_BLACK, DARK_WEIGHT, 1.0) * np.where((v > 0.95) & (s < 0.10), GLARE_WEIGHT, 1.0)

    tally = np.bincount(idx, weights=w, minlength=len(COLORS))
    shares = tally / max(tally.sum(), 1e-9)
    chromatic = [COLORS.index(c) for c in ("red", "orange", "yellow", "green", "blue", "brown")]
    best_chroma = max(chromatic, key=lambda i: shares[i])
    if shares[best_chroma] >= CHROMA_SHARE:
        winner = best_chroma
    else:
        winner = max((COLORS.index(c) for c in ("white", "gray", "black")), key=lambda i: shares[i])
    return COLORS[winner], float(shares[winner]), {c: round(float(x), 3) for c, x in zip(COLORS, shares)}

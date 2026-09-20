"""
Crop geometry shared by the Stage-2 type classifier: its dataset builder, its
evaluation, and the two-stage inference pipeline.

Training and inference MUST crop identically, or the classifier is measured on
one input distribution and deployed on another. So there is exactly one
implementation, and it lives here.

Usage from a script one level down:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _crops import TYPE_CLASSES, MIN_TYPE_PX, square_crop
"""

from PIL import Image

from _paths import CLASS_NAMES

# Stage-2 targets: every class in configs/vehicle_7class.yaml except the
# umbrella. 'Vehicle' is not a type -- it is what the pipeline answers when a box
# is below the gate or the classifier is unsure, which is a true-but-unspecific
# answer rather than a wrong one.
TYPE_CLASSES = [n for n in CLASS_NAMES if n != "Vehicle"]

# Box long side, in NATIVE image pixels, below which nobody can name a subtype.
# Same gate as scripts/labeling/build_crops.py, where the reasoning is recorded.
MIN_TYPE_PX = 48

PAD  = 0.15               # context on each side, as a fraction of the long side
FILL = (114, 114, 114)    # ultralytics letterbox gray, for area beyond the frame


def square_crop(im: Image.Image, box, pad: float = PAD, max_side: int = None) -> Image.Image:
    """
    Square crop centred on an xyxy pixel box, side = long side * (1 + 2*pad).

    Square on purpose: the classifier resizes and centre-crops to a square, so a
    rectangular crop would either lose the ends of the vehicle or distort its
    aspect ratio -- and aspect ratio is a real cue (a bus is long, a van tall).
    Area beyond the frame edge is filled gray rather than clamped, so a vehicle
    at the border keeps its position and scale inside the crop.
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(1, int(round(max(x2 - x1, y2 - y1) * (1 + 2 * pad))))
    sx, sy = int(round(cx - side / 2)), int(round(cy - side / 2))

    canvas = Image.new("RGB", (side, side), FILL)
    ix1, iy1 = max(0, sx), max(0, sy)
    ix2, iy2 = min(im.width, sx + side), min(im.height, sy + side)
    if ix2 > ix1 and iy2 > iy1:
        canvas.paste(im.crop((ix1, iy1, ix2, iy2)), (ix1 - sx, iy1 - sy))

    if max_side and side > max_side:
        canvas = canvas.resize((max_side, max_side), Image.LANCZOS)
    return canvas

# One vehicle detected by the two-stage pipeline.
# Mirrors the detection dict returned by scripts/pipeline/two_stage.py so the
# wire format and the offline pipeline cannot drift apart.

# Stage 1 -- v4 detector. Axis-aligned box in pixels, [x1, y1, x2, y2],
# origin top-left. The detector's own class output is deliberately ignored
# and NMS is class-agnostic; its job is finding vehicles, not naming them.
float32[4] box
float32    det_conf

# Longest side of the box in pixels. Both stage-2 gates key off this.
float32    size_px

# Stage 2 -- type classifier. 'Vehicle' is the umbrella class: true but
# unspecific. It is what you get when the box was too small to type or the
# classifier was not confident enough, NOT a generic leftover bucket.
string     type

# Which of those it was: 'typed', 'unsure', or 'too_small'.
string     type_status

# The classifier's best guess, kept even when the pipeline fell back to
# 'Vehicle' so downstream nodes can apply their own threshold. Empty string
# and NaN when stage 2 did not run at all.
string     type_guess
float32    type_conf

# Pixel-rule colour naming. Empty string and NaN below the colour gate.
# Rules only, scored about 0.62 -- treat as a hint, not a label.
string     color
float32    color_conf

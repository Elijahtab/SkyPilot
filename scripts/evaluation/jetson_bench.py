"""
Benchmark a YOLO .pt model on a folder of images: speed, power and temperature.

Written for the Jetson Orin Nano, and deliberately STANDALONE: it does not import
_paths or anything else from this repo, so it can be copied onto the board by
itself. Python 3.8 compatible (the ultralytics JetPack 5 container ships 3.8).

What it measures, in order:
  1. idle     board power with nothing running (baseline)
  2. warm-up  first N images, not timed (CUDA init and kernel selection are slow)
  3. run      every image, --passes times, one at a time (batch 1, like a live feed)

Timing per image:
  read       loading the file from disk (reported, but excluded from FPS)
  model      predict() wall time, CUDA-synchronised -- the FPS number
  pre/inf/post  ultralytics' own breakdown of that call

Power comes from the board's INA3221 sensors in /sys, sampled every 100 ms, so
only the timed phase is averaged. VDD_IN is total board input power. If the
sensors cannot be read, speed is still measured and the script says so --
use `sudo tegrastats` on the host instead.

Usage (inside the ultralytics container, home mounted at /work):
    python /work/jetson_bench.py /work/drone_images --model /work/best_vehicle.pt
    python /work/jetson_bench.py /work/drone_images --half          # FP16, still the .pt
    python /work/jetson_bench.py /work/drone_images --save-annotated /work/annotated

Writes a JSON report (all numbers + per-image timings) to --out.
"""

import argparse
import glob
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -- sensors ----------------------------------------------------------
def find_power_rails():
    """{rail name: (voltage file mV, current file mA)} from the INA3221 hwmon driver."""
    rails = {}
    for label in glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label"):
        try:
            name = Path(label).read_text().strip()
        except OSError:
            continue
        d = os.path.dirname(label)
        idx = os.path.basename(label)[2:-len("_label")]        # in1_label -> 1
        volt = os.path.join(d, "in%s_input" % idx)
        curr = os.path.join(d, "curr%s_input" % idx)
        if os.path.exists(volt) and os.path.exists(curr):
            rails[name] = (volt, curr)
    return rails


def read_mw(volt, curr):
    with open(volt) as fv, open(curr) as fc:
        return int(fv.read()) * int(fc.read()) / 1000.0


def max_temp_c():
    temps = []
    for f in glob.glob("/sys/devices/virtual/thermal/thermal_zone*/temp"):
        try:
            t = int(Path(f).read_text()) / 1000.0
        except (OSError, ValueError):
            continue
        if 0 < t < 150:                     # skip placeholder zones
            temps.append(t)
    return max(temps) if temps else None


class Sampler(threading.Thread):
    """Samples every rail and the hottest thermal zone, tagged with the current phase."""

    def __init__(self, rails, period=0.1):
        super().__init__(daemon=True)
        self.rails, self.period = rails, period
        self.phase = None
        self.power = {}                     # phase -> rail -> [mW]
        self.temps = {}                     # phase -> [C]
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            phase = self.phase
            if phase:
                for name, (v, c) in self.rails.items():
                    try:
                        self.power.setdefault(phase, {}).setdefault(name, []).append(read_mw(v, c))
                    except (OSError, ValueError):
                        pass
                t = max_temp_c()
                if t is not None:
                    self.temps.setdefault(phase, []).append(t)
            time.sleep(self.period)

    def stop(self):
        self._halt.set()
        self.join()

    def mean_w(self, phase, rail):
        vals = self.power.get(phase, {}).get(rail, [])
        return (statistics.mean(vals) / 1000.0, len(vals)) if vals else (None, 0)


# -- helpers ----------------------------------------------------------
def pct(sorted_vals, q):
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def summarise(vals):
    s = sorted(vals)
    return {"mean": statistics.mean(s), "p50": pct(s, 0.50), "p95": pct(s, 0.95),
            "min": s[0], "max": s[-1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", help="folder of images (searched recursively)")
    ap.add_argument("--model", default="best_vehicle.pt")
    ap.add_argument("--imgsz", type=int, default=640, help="v4 was trained at 640")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--half", action="store_true", help="FP16 inference on the .pt (no conversion)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--passes", type=int, default=3, help="timed passes over the folder")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--idle-seconds", type=float, default=15)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N images (0 = all)")
    ap.add_argument("--save-annotated", default=None, help="folder for images with boxes drawn (after timing)")
    ap.add_argument("--save-count", type=int, default=20)
    ap.add_argument("--out", default=None, help="JSON report path (default: next to the model)")
    args = ap.parse_args()

    img_dir = Path(args.images)
    imgs = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        imgs = imgs[:args.limit]
    if not imgs:
        sys.exit("[ERR] no images found in %s" % img_dir)
    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit("[ERR] model not found: %s" % model_path)

    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    cuda = torch.cuda.is_available() and args.device != "cpu"
    if not cuda and args.device != "cpu":
        sys.exit("[ERR] torch cannot see the GPU. Inside Docker, start the container with "
                 "--runtime=nvidia. Pass --device cpu to benchmark the CPU on purpose.")

    rails = find_power_rails()
    main_rail = "VDD_IN" if "VDD_IN" in rails else (sorted(rails)[0] if rails else None)

    print("=" * 72)
    print("model      %s" % model_path)
    print("images     %d from %s  (x%d passes)" % (len(imgs), img_dir, args.passes))
    print("settings   imgsz %d  conf %.2f  %s  device %s" %
          (args.imgsz, args.conf, "FP16" if args.half else "FP32", args.device))
    print("software   torch %s  ultralytics %s  cuda %s" %
          (torch.__version__, ultralytics.__version__, torch.cuda.get_device_name(0) if cuda else "-"))
    print("power      %s" % (", ".join(sorted(rails)) if rails else
                             "NO SENSORS READABLE -- speed only; use `sudo tegrastats` on the host"))
    print("=" * 72)

    sampler = Sampler(rails)
    sampler.start()

    # 1. idle baseline, before the model is loaded
    if rails and args.idle_seconds > 0:
        print("idle baseline: %.0f s, keep the board otherwise idle ..." % args.idle_seconds)
        sampler.phase = "idle"
        time.sleep(args.idle_seconds)
        sampler.phase = None

    def sync():
        if cuda:
            torch.cuda.synchronize()

    model = YOLO(str(model_path))
    kw = dict(imgsz=args.imgsz, conf=args.conf, half=args.half, device=args.device, verbose=False)

    # 2. warm-up
    print("warm-up: %d images ..." % min(args.warmup, len(imgs)))
    sampler.phase = "warmup"
    for p in imgs[:args.warmup]:
        model.predict(cv2.imread(str(p)), **kw)
    sync()
    sampler.phase = None

    # 3. timed run
    per_image, read_ms, model_ms, pre, inf, post, dets = [], [], [], [], [], [], []
    shapes = set()
    sampler.phase = "run"
    run_start = time.perf_counter()
    for n in range(args.passes):
        pass_start = time.perf_counter()
        for p in imgs:
            t0 = time.perf_counter()
            frame = cv2.imread(str(p))
            t1 = time.perf_counter()
            if frame is None:
                print("  [skip] unreadable: %s" % p)
                continue
            r = model.predict(frame, **kw)[0]
            sync()
            t2 = time.perf_counter()

            read_ms.append((t1 - t0) * 1000)
            model_ms.append((t2 - t1) * 1000)
            pre.append(r.speed["preprocess"])
            inf.append(r.speed["inference"])
            post.append(r.speed["postprocess"])
            dets.append(len(r.boxes))
            shapes.add(frame.shape[:2])
            if n == 0:
                per_image.append({"file": str(p.relative_to(img_dir)), "model_ms": round(model_ms[-1], 2),
                                  "detections": dets[-1]})
        print("  pass %d/%d: %.1f s" % (n + 1, args.passes, time.perf_counter() - pass_start))
    run_s = time.perf_counter() - run_start
    sampler.phase = None
    sampler.stop()

    if not model_ms:
        sys.exit("[ERR] no image could be read")

    # -- report -------------------------------------------------------
    m = summarise(model_ms)
    report = {
        "model": str(model_path), "images_dir": str(img_dir), "n_images": len(imgs),
        "passes": args.passes, "n_timed": len(model_ms), "imgsz": args.imgsz, "conf": args.conf,
        "precision": "FP16" if args.half else "FP32", "device": args.device,
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "image_shapes_hw": sorted(list(shapes)),
        "model_ms": m, "fps": 1000.0 / m["mean"],
        "read_ms": summarise(read_ms),
        "preprocess_ms": statistics.mean(pre), "inference_ms": statistics.mean(inf),
        "postprocess_ms": statistics.mean(post),
        "detections_per_image": statistics.mean(dets),
        "run_seconds": run_s, "power": {}, "temp_c": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_image_first_pass": per_image,
    }

    print("\n" + "=" * 72)
    print("SPEED  (batch 1, %d timed predictions, %s)" % (len(model_ms), report["precision"]))
    print("  model     mean %.1f ms | p50 %.1f | p95 %.1f | max %.1f   -> %.1f FPS"
          % (m["mean"], m["p50"], m["p95"], m["max"], report["fps"]))
    print("  breakdown preprocess %.1f + inference %.1f + postprocess %.1f ms"
          % (report["preprocess_ms"], report["inference_ms"], report["postprocess_ms"]))
    print("  disk read mean %.1f ms (not counted in FPS)" % report["read_ms"]["mean"])
    print("  image sizes %s   detections/image %.1f"
          % (", ".join("%dx%d" % (w, h) for h, w in report["image_shapes_hw"][:4]), report["detections_per_image"]))

    if main_rail:
        idle_w, n_idle = sampler.mean_w("idle", main_rail)
        run_w, n_run = sampler.mean_w("run", main_rail)
        print("\nPOWER  (%s = %s)" % (main_rail, "total board input" if main_rail == "VDD_IN" else "first rail found"))
        if idle_w is not None:
            print("  idle      %.2f W   (%d samples)" % (idle_w, n_idle))
        if run_w is not None:
            s_per_img = m["mean"] / 1000.0
            print("  running   %.2f W   (%d samples)" % (run_w, n_run))
            print("  energy    %.3f J per image (total board)" % (run_w * s_per_img))
            if idle_w is not None:
                print("  model     +%.2f W over idle -> %.3f J per image" %
                      (run_w - idle_w, (run_w - idle_w) * s_per_img))
        for rail in sorted(rails):
            w, _ = sampler.mean_w("run", rail)
            wi, _ = sampler.mean_w("idle", rail)
            report["power"][rail] = {"idle_w": wi, "run_w": w}
            if rail != main_rail and w is not None:
                print("  %-16s running %.2f W%s" % (rail, w, "" if wi is None else " (idle %.2f W)" % wi))
        if run_w is not None:
            report["joules_per_image"] = run_w * m["mean"] / 1000.0
    for phase in ("idle", "run"):
        t = sampler.temps.get(phase)
        if t:
            report["temp_c"][phase] = {"start": t[0], "end": t[-1], "max": max(t)}
    if "run" in report["temp_c"]:
        t = report["temp_c"]["run"]
        print("\nTEMP   hottest zone %.1f C at start of run, %.1f C at end, max %.1f C" % (t["start"], t["end"], t["max"]))
    print("=" * 72)

    out = Path(args.out) if args.out else model_path.parent / (
        "bench_%s_%s_%d_%s.json" % (model_path.stem, report["precision"], args.imgsz, time.strftime("%Y%m%d_%H%M%S")))
    out.write_text(json.dumps(report, indent=1))
    print("report -> %s" % out)

    # annotated copies, outside the timed and power-measured phases
    if args.save_annotated:
        dst = Path(args.save_annotated)
        dst.mkdir(parents=True, exist_ok=True)
        for p in imgs[:args.save_count]:
            r = model.predict(cv2.imread(str(p)), **kw)[0]
            cv2.imwrite(str(dst / (p.stem + "_pred.jpg")), r.plot())
        print("annotated -> %s (%d images)" % (dst, min(args.save_count, len(imgs))))


if __name__ == "__main__":
    main()

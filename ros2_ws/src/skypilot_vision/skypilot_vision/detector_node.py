"""ROS 2 node wrapping the two-stage vehicle pipeline.

    /camera/image_raw (sensor_msgs/Image) --> TwoStagePipeline --> vehicles
                                                       (skypilot_msgs/VehicleArray)

The pipeline is NOT reimplemented here. This node imports
scripts/pipeline/two_stage.py out of the repo so the offline tools
(index_frames.py, search_vehicles.py) and the live robot always run identical
detection, typing and colour code. If they ever disagree, that is a bug here.

Frame dropping is deliberate. Inference takes far longer than a camera frame
interval -- about 69 ms per frame on an Orin Nano at 640 -- so the image
subscription uses the sensor-data QoS profile: best effort, depth 1. Frames
arriving while inference is busy are dropped by the middleware instead of
queued, which keeps latency bounded rather than letting a backlog grow. A
tracker wants the freshest frame, not every frame.
"""

import os
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from skypilot_msgs.msg import Vehicle, VehicleArray

NAN = float("nan")


def _load_pipeline_class(repo_root):
    """Import TwoStagePipeline out of the repo's scripts/ tree.

    two_stage.py bootstraps its own sibling imports (_color, _crops, _paths) by
    inserting scripts/ on sys.path, so putting scripts/ on the path here is
    enough to make the whole chain resolve.
    """
    scripts = Path(repo_root).expanduser().resolve() / "scripts"
    if not scripts.is_dir():
        raise RuntimeError(
            "repo_root '%s' has no scripts/ directory. Point the repo_root "
            "parameter (or $SKYPILOT_REPO) at the SkyPilot checkout." % repo_root)
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from pipeline.two_stage import TwoStagePipeline
    return TwoStagePipeline


class DetectorNode(Node):

    def __init__(self):
        super().__init__("vehicle_detector")

        self.declare_parameter("repo_root", os.environ.get("SKYPILOT_REPO", "/workspace"))
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("det_conf", 0.25)
        self.declare_parameter("type_conf", 0.50)
        # "auto" picks the GPU when torch can see one, "cpu" forces CPU, a
        # digit pins a specific CUDA device.
        self.declare_parameter("device", "auto")
        self.declare_parameter("publish_annotated", False)
        self.declare_parameter("log_every", 30)

        repo_root = self.get_parameter("repo_root").value
        image_topic = self.get_parameter("image_topic").value
        self._log_every = int(self.get_parameter("log_every").value)

        device = self.get_parameter("device").value
        if device == "auto":
            device = None
        elif device.isdigit():
            device = int(device)

        self.get_logger().info("loading models from %s (this takes a moment) ..." % repo_root)
        pipeline_cls = _load_pipeline_class(repo_root)
        self.pipe = pipeline_cls(
            det_conf=float(self.get_parameter("det_conf").value),
            type_conf=float(self.get_parameter("type_conf").value),
            device=device,
        )
        self.get_logger().info("pipeline ready: %s" % self.pipe.versions)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(VehicleArray, "vehicles", 10)

        self._annotated_pub = None
        if bool(self.get_parameter("publish_annotated").value):
            # Debug aid only: drawing and re-encoding costs real time on the
            # Jetson, so it stays off unless asked for.
            self._annotated_pub = self.create_publisher(Image, "vehicles/annotated", 1)

        self.sub = self.create_subscription(
            Image, image_topic, self.on_image, qos_profile_sensor_data)

        self._frames = 0
        self._bad_frames = 0
        self.get_logger().info("subscribed to %s, publishing vehicles" % image_topic)

    def on_image(self, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:               # malformed frame: skip it, stay alive
            self._bad_frames += 1
            self.get_logger().warn("cv_bridge failed (%d so far): %s"
                                   % (self._bad_frames, exc))
            return

        # The pipeline was validated on RGB arrays; cv_bridge hands back BGR.
        # ascontiguousarray because the reversed view is a negative-stride
        # slice, which PIL.Image.fromarray refuses.
        import numpy as np
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])

        dets, (width, height) = self.pipe.from_array(rgb)

        out = VehicleArray()
        # Stamp from the source frame, not from now, so a consumer can match a
        # detection back to the image it came from.
        out.header = msg.header
        out.image_width = int(width)
        out.image_height = int(height)
        out.pipeline_versions = self.pipe.versions
        out.vehicles = [self._to_msg(d) for d in dets]
        self.pub.publish(out)

        if self._annotated_pub is not None:
            annotated = self.bridge.cv2_to_imgmsg(self._annotate(bgr, dets), encoding="bgr8")
            annotated.header = msg.header
            self._annotated_pub.publish(annotated)

        self._frames += 1
        if self._log_every and self._frames % self._log_every == 0:
            typed = sum(1 for d in dets if d["type_status"] == "typed")
            self.get_logger().info("frame %d: %d vehicles (%d typed)"
                                   % (self._frames, len(dets), typed))

    @staticmethod
    def _to_msg(d):
        """One detection dict -> one Vehicle message.

        None is not representable on the wire, so an absent confidence becomes
        NaN and an absent string becomes empty. Consumers should branch on
        type_status rather than sniffing for NaN.
        """
        v = Vehicle()
        v.box = [float(x) for x in d["box"]]
        v.det_conf = float(d["det_conf"])
        v.size_px = float(d["size_px"])
        v.type = d["type"]
        v.type_status = d["type_status"]
        v.type_guess = d["type_guess"] or ""
        v.type_conf = NAN if d["type_conf"] is None else float(d["type_conf"])
        v.color = d["color"] or ""
        v.color_conf = NAN if d["color_conf"] is None else float(d["color_conf"])
        return v

    @staticmethod
    def _annotate(bgr, dets):
        import cv2
        out = bgr.copy()
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d["box"])
            # Solid for a confident type, dim for the umbrella fallback, so the
            # distinction the pipeline works to preserve stays visible.
            typed = d["type_status"] == "typed"
            color = (0, 200, 0) if typed else (120, 120, 120)
            label = "%s %s" % (d["color"], d["type"]) if d["color"] else d["type"]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return out


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

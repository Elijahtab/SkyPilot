"""Bring up the vehicle detector on its own.

    ros2 launch skypilot_vision vision.launch.py image_topic:=/camera/image_raw

Every argument maps to a node parameter of the same name. Defaults match the
offline pipeline (det_conf 0.25, type_conf 0.50) so a bag replayed through
this launch file reproduces what index_frames.py would have written.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGS = (
    ("repo_root", os.environ.get("SKYPILOT_REPO", "/workspace"),
     "SkyPilot checkout that provides scripts/pipeline/two_stage.py"),
    ("image_topic", "/camera/image_raw", "sensor_msgs/Image topic to consume"),
    ("det_conf", "0.25", "stage-1 detector confidence"),
    ("type_conf", "0.50", "stage-2 threshold below which the answer is 'Vehicle'"),
    ("device", "auto", "auto | cpu | a CUDA device index"),
    ("publish_annotated", "false", "also publish drawn boxes (costs time on the Jetson)"),
)


def generate_launch_description():
    declared = [DeclareLaunchArgument(name, default_value=default, description=desc)
                for name, default, desc in ARGS]

    detector = Node(
        package="skypilot_vision",
        executable="vehicle_detector",
        name="vehicle_detector",
        output="screen",
        parameters=[{name: LaunchConfiguration(name) for name, _, _ in ARGS}],
    )

    return LaunchDescription(declared + [detector])

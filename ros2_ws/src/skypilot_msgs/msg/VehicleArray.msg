# Every vehicle found in one frame.
# header.stamp is copied from the incoming image, not the time inference
# finished, so consumers can align detections with the frame they came from.
std_msgs/Header header

# Size of the frame these boxes are in pixels. Boxes are absolute, not
# normalised, so a consumer that rescales the image needs these.
uint32 image_width
uint32 image_height

# Which weights produced this, e.g. 'det=Vehicle_type_detection_v4 cls=type_cls_v1'.
# Recorded per message so a rosbag stays interpretable after the models move on.
string pipeline_versions

Vehicle[] vehicles

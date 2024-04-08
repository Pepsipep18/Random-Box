#!/usr/bin/env python
import rospy
import tf
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2D
from geometry_msgs.msg import PoseStamped, PointStamped
from cv_bridge import CvBridge
import numpy as np

rospy.init_node('TargetPosition')

listener = tf.TransformListener()
bridge = CvBridge()

def handle_camera_info(data, camera_info={}):
    camera_info['K'] = data.K
    return camera_info

def handle_depth_image(data):
    try:
        return bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")
    except Exception as e:
        rospy.logerr("CV Bridge conversion error: {}".format(e))
        return None

def calculate_target_position(center, depth, camera_info):
    if not camera_info or depth is None:
        return None

    fx = camera_info['K'][0]
    fy = camera_info['K'][4]
    cx = camera_info['K'][2]
    cy = camera_info['K'][5]

    X = (center.x - cx) * depth / fx
    Y = (center.y - cy) * depth / fy
    Z = depth - 1.0  # Subtract to avoid collision
    return X, Y, max(Z, 0.0)  # Ensure Z is not negative

def handle_detection(detection_msg, camera_info, depth_image):
    if camera_info is None or depth_image is None:
        rospy.loginfo("Waiting for camera info and depth image...")
        return

    bbox = detection_msg.bbox
    if bbox.size_x == 0 or bbox.size_y == 0:
        return

    center = bbox.center
    depth = depth_image[int(center.y), int(center.x)]

    if np.isnan(depth) or np.isinf(depth):
        rospy.loginfo("Invalid depth value.")
        return

    target_pos = calculate_target_position(center, depth, camera_info)
    if target_pos is None:
        rospy.loginfo("Unable to calculate target position.")
        return

    # Convert to PointStamped for transformation
    point_stamped = PointStamped()
    point_stamped.header.stamp = rospy.Time.now()
    point_stamped.header.frame_id = "front_frame_optical"
    point_stamped.point.x, point_stamped.point.y, point_stamped.point.z = target_pos

    # Transform and publish the target position
    try:
        map_point = listener.transformPoint("map", point_stamped)
        publish_goal(map_point)
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logerr("TF error when converting point: {}".format(e))

def publish_goal(map_point):
    goal_pose = PoseStamped()
    goal_pose.header.stamp = rospy.Time.now()
    goal_pose.header.frame_id = "map"
    goal_pose.pose.position = map_point.point
    goal_pose.pose.orientation.w = 1.0
    goal_pub.publish(goal_pose)
    rospy.loginfo("Goal pose published successfully.")

camera_info_sub = rospy.Subscriber('/front/rgb/camera_info', CameraInfo, handle_camera_info)
depth_image_sub = rospy.Subscriber('/me5413/current_depth', Image, handle_depth_image)
detection_sub = rospy.Subscriber('/me5413/detected', Detection2D, handle_detection)
goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

camera_info = {}
rospy.spin()

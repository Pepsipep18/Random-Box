#!/usr/bin/env python
import rospy
import tf
import cv2
import numpy as np
import os
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2D
from geometry_msgs.msg import PoseStamped, PointStamped
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

# Set up paths and initial tracking coordinates
template_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1_resized.png")
initial_tracking_coordinates = [(0, 0, 106, 137)]
template_rectangle = initial_tracking_coordinates[0]

# Initialize the CV Bridge
cv_bridge = CvBridge()

# ROS subscribers and publishers
image_subscriber = None
depth_image_subscriber = None
template_subscriber = None
detection_publisher = None
current_depth_publisher = None
goal_pub = None

# Template, depth image, and camera info placeholders
current_template = None
depth_image_data = None
camera_info = {}

# Initialize TF listener
listener = tf.TransformListener()

def handle_template_message(data):
    global current_template
    if "box" not in data.data:
        current_template = None
        return

    try:
        template_image = cv2.imread(template_image_path)
        if template_image is None:
            rospy.logerr("Failed to load template image from path: {}".format(template_image_path))
        else:
            rospy.loginfo("Template image loaded successfully from path: {}".format(template_image_path))
            x, y, w, h = template_rectangle
            current_template = template_image[y:y+h, x:x+w]
    except Exception as e:
        rospy.logerr("Error loading template image: {}".format(e))

def handle_depth_image(data):
    global depth_image_data
    try:
        depth_image_data = cv_bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")
    except CvBridgeError as e:
        rospy.logerr("CV Bridge conversion error: {}".format(e))

def handle_input_image(data):
    global current_template, depth_image_data
    if current_template is None:
        rospy.loginfo("No template loaded.")
        return

    try:
        cv_image = cv_bridge.imgmsg_to_cv2(data, "bgr8")
        detect_and_publish(cv_image, depth_image_data)
    except CvBridgeError as e:
        rospy.logerr(str(e))

def handle_camera_info(data):
    global camera_info
    camera_info['K'] = data.K

def handle_detection(detection_msg):
    global depth_image_data, camera_info
    if camera_info is None or depth_image_data is None:
        rospy.loginfo("Waiting for camera info and depth image...")
        return

    bbox = detection_msg.bbox
    if bbox.size_x == 0 or bbox.size_y == 0:
        return

    center = bbox.center
    depth = depth_image_data[int(center.y), int(center.x)]

    if np.isnan(depth) or np.isinf(depth):
        rospy.loginfo("Invalid depth value.")
        return

    target_pos = calculate_target_position(center, depth, camera_info)
    if target_pos:
        publish_goal(target_pos)

def detect_and_publish(image, current_depth):
    global current_template
    if current_template is None:
        rospy.loginfo("No template for detection.")
        return

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(current_template, cv2.COLOR_BGR2GRAY)

    # Fix: Initialize original_height and original_width based on the input image size
    original_height, original_width = image_gray.shape[:2]

    scaled_template_height, scaled_template_width = template_gray.shape[:2]
    scales = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    max_matching_value = -1
    optimal_scale = None
    optimal_location = None

    for scale in scales:
        scaled_width = int(scaled_template_width * scale)
        scaled_height = int(scaled_template_height * scale)
        resized_template = cv2.resize(template_gray, (scaled_width, scaled_height), cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
        matching_result = cv2.matchTemplate(image_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        _, maximum_value, _, maximum_location = cv2.minMaxLoc(matching_result)

        if maximum_value > max_matching_value:
            max_matching_value = maximum_value
            optimal_scale = scale
            optimal_location = maximum_location

    if max_matching_value > 0.85:
        x, y = optimal_location
        width = int(scaled_template_width * optimal_scale)
        height = int(scaled_template_height * optimal_scale)
        publish_detection(x, y, width, height, image, current_depth)
    else:
        rospy.loginfo("No significant match found.")

def publish_detection(x, y, width, height, image, current_depth):
    detection = Detection2D()
    detection.bbox.size_x = width
    detection.bbox.size_y = height
    detection.bbox.center.x = x + width // 2
    detection.bbox.center.y = y + height // 2
    try:
        ros_image = cv_bridge.cv2_to_imgmsg(image, "bgr8")
        detection.source_img = ros_image
        detection_publisher.publish(detection)
        # Note: Publishing current_depth directly may not be correct without conversion
        # It's placeholder logic to indicate where depth information might be integrated
        current_depth_publisher.publish(cv_bridge.cv2_to_imgmsg(current_depth, "32FC1"))
    except CvBridgeError as e:
        rospy.logerr(str(e))

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
    return (X, Y, max(Z, 0.0))  # Ensure Z is not negative

def publish_goal(target_pos):
    X, Y, Z = target_pos
    goal_pose = PoseStamped()
    goal_pose.header.stamp = rospy.Time.now()
    goal_pose.header.frame_id = "map"
    goal_pose.pose.position.x = X
    goal_pose.pose.position.y = Y
    goal_pose.pose.position.z = Z
    goal_pose.pose.orientation.w = 1.0
    goal_pub.publish(goal_pose)
    rospy.loginfo("Goal pose published successfully.")

def initialize_ros_components():
    global image_subscriber, depth_image_subscriber, template_subscriber, detection_publisher, current_depth_publisher, goal_pub
    rospy.init_node('Detection', anonymous=True)

    # Subscribers
    image_subscriber = rospy.Subscriber("/front/rgb/image_raw", Image, handle_input_image)
    depth_image_subscriber = rospy.Subscriber('/front/depth/image_raw', Image, handle_depth_image)
    template_subscriber = rospy.Subscriber("/rviz_panel/goal_name", String, handle_template_message)
    camera_info_sub = rospy.Subscriber('/front/rgb/camera_info', CameraInfo, handle_camera_info)

    # Publishers
    detection_publisher = rospy.Publisher("/me5413/detected", Detection2D, queue_size=10)
    current_depth_publisher = rospy.Publisher("/me5413/current_depth", Image, queue_size=10)
    goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

initialize_ros_components()
rospy.spin()

#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from std_msgs.msg import String
from vision_msgs.msg import Detection2D
import os

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

# Template and depth image placeholders
current_template = None
depth_image_data = None

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
    depth_image_data = data

def handle_input_image(data):
    global current_template
    if current_template is None:
        empty_detection = Detection2D()
        empty_detection.bbox.size_x = 0
        empty_detection.bbox.size_y = 0
        empty_detection.bbox.center.x = 0
        empty_detection.bbox.center.y = 0
        empty_detection.source_img = data
        detection_publisher.publish(empty_detection)
        return

    try:
        cv_image = cv_bridge.imgmsg_to_cv2(data, "8UC3")
        current_depth = depth_image_data
        detect_and_publish(cv_image, current_depth)
    except CvBridgeError as e:
        rospy.logerr(str(e))

def detect_and_publish(image, current_depth):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(current_template, cv2.COLOR_BGR2GRAY)

    scaled_template_height, scaled_template_width = template_gray.shape[:2]
    original_height, original_width = original_height * 0.5, original_width * 0.5
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
    else:
        x, y, width, height = 0, 0, 0, 0

    publish_detection(x, y, width, height, image, current_depth)

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
        current_depth_publisher.publish(current_depth)
    except CvBridgeError as e:
        rospy.logerr(str(e))

def initialize_ros_components():
    global image_subscriber, depth_image_subscriber, template_subscriber, detection_publisher, current_depth_publisher
    rospy.init_node('Detection', anonymous=True)
    image_subscriber = rospy.Subscriber("/front/rgb/image_raw", Image, handle_input_image)
    depth_image_subscriber = rospy.Subscriber('/front/depth/image_raw', Image, handle_depth_image)
    template_subscriber = rospy.Subscriber("/rviz_panel/goal_name", String, handle_template_message)
    detection_publisher = rospy.Publisher("/me5413/detected", Detection2D, queue_size=10)
    current_depth_publisher = rospy.Publisher("/me5413/current_depth", Image, queue_size=10)

initialize_ros_components()
rospy.spin()

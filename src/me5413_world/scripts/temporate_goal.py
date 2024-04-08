import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from std_msgs.msg import String
from vision_msgs.msg import Detection2D
from geometry_msgs.msg import PoseStamped, PointStamped
import os
import tf

class SetGoal(object):
    def __init__(self):
        rospy.init_node('set_goal_node')  # 初始化ROS节点

        self.bridge = CvBridge()
        self.template_sub = rospy.Subscriber("/rviz_panel/goal_name", String, self.Judge_Target)
        
        self.listener = tf.TransformListener()
        # Template for matching, obtained from the first frame.
        self.template = None
        self.done = False
        self.depth_image = None

        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.goal_name_pub = rospy.Publisher('/rviz_panel/goal_name', String, queue_size=10)
        self.calculated_target_pub = rospy.Publisher('/calculated_target', PoseStamped, queue_size=10)  # 创建 calculated_target_pub 属性

    def Judge_Target(self, name):
        goal_name = name.data
        end = goal_name.rfind("_")
        self.goal_type_ = goal_name[1:end]
        goal_box_id = int(goal_name[end+1])

        if self.goal_type_ == "box":
            # 在这里执行 box 类型的处理逻辑
            X = 13
            Y = -2
            Yaw = -3.14/2

            point_stamped = PointStamped()
            current_time = rospy.Time.now()
            point_stamped.header.stamp = current_time
            point_stamped.header.frame_id = "world"
            point_stamped.point.x = X
            point_stamped.point.y = Y
            point_stamped.point.z = 0
    # 设置位置信息
            # point_stamped.pose.position.x = X
            # point_stamped.pose.position.y = Y
            # point_stamped.pose.position.z = 0  # 如果姿态信息不需要Z轴，则可以设置为0

            # # 设置姿态信息
            # quaternion = tf.transformations.quaternion_from_euler(0, 0, Yaw)
            # point_stamped.pose.orientation.x = quaternion[0]
            # point_stamped.pose.orientation.y = quaternion[1]
            # point_stamped.pose.orientation.z = quaternion[2]
            # point_stamped.pose.orientation.w = quaternion[3]

            try:
                self.listener.waitForTransform("map", point_stamped.header.frame_id, current_time, rospy.Duration(4.0))
                map_point = self.listener.transformPoint("map", point_stamped)

                goal_pose = PoseStamped()
                goal_pose.header.stamp = current_time
                goal_pose.header.frame_id = "map"
                goal_pose.pose.position.x = map_point.point.x
                goal_pose.pose.position.y = map_point.point.y
                goal_pose.pose.position.z = map_point.point.z
                goal_pose.pose.orientation.w = Yaw

                rospy.loginfo("Target position successfully calculated: x={}, y={}, z={}".format(
                    goal_pose.pose.position.x,
                    goal_pose.pose.position.y,
                    goal_pose.pose.position.z
                ))

                # Publish the goal pose
                self.goal_pub.publish(goal_pose)
                self.calculated_target_pub.publish(goal_pose)
                # Publish the goal name to substitute the goal name from the last goal
                goal_name_msg = String()
                goal_name_msg.data = "/done_1"
                self.goal_name_pub.publish(goal_name_msg)

                return
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logerr("TF error when converting point: %s", e)

if __name__ == '__main__':
    node = SetGoal()
    rospy.spin()

        

        

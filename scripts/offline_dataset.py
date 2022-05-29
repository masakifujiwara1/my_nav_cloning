#!/usr/bin/env python
from __future__ import print_function
import copy
import time
import os
import csv
import random
from std_srvs.srv import SetBool, SetBoolResponse
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Int8MultiArray
from nav_msgs.msg import Path
from std_srvs.srv import Trigger
from std_msgs.msg import Int8
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Twist
from skimage.transform import resize
from nav_cloning_net_cross import *
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import cv2
import rospy
import roslib
import sys
roslib.load_manifest('nav_cloning')


class offline():
    def __init__(self):
        rospy.init_node('offline_node', anonymous=True)
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.image_sub = rospy.Subscriber(
            "/camera_center/image_raw", Image, self.callback)
        self.image_left_sub = rospy.Subscriber(
            "/camera_left/image_raw", Image, self.callback_left_camera)
        self.image_right_sub = rospy.Subscriber(
            "/camera_right/image_raw", Image, self.callback_right_camera)
        self.cmd = rospy.Subscriber(
            "/icart_mini/cmd_vel", Twist, self.callback_cmd)
        self.dir_cmd_sub = rospy.Subscriber(
            "/cmd_dir", Int8MultiArray, self.callback_cmd_data)
        self.target_action = 0.0
        self.counter = 0
        self.bridge = CvBridge()
        self.path = roslib.packages.get_pkg_dir(
            'nav_cloning') + '/data/dataset/'
        os.makedirs(self.path + self.start_time)
        os.makedirs(self.path + self.start_time + '/image')
        self.image_path = roslib.packages.get_pkg_dir(
            'nav_cloning') + '/data/dataset/' + self.start_time + '/image/'
        self.cmd_path = roslib.packages.get_pkg_dir(
            'nav_cloning') + '/data/dataset/' + self.start_time + '/target_action'
        self.dir_cmd_data = (100, 0, 0, 0)

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_left_camera(self, data):
        try:
            self.cv_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_right_camera(self, data):
        try:
            self.cv_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_cmd(self, data):
        self.target_action = data.angular.z

    def callback_cmd_data(self, data):
        self.dir_cmd_data = data.data

    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.cv_left_image.size != 640 * 480 * 3:
            return
        if self.cv_right_image.size != 640 * 480 * 3:
            return

        img = cv2.resize(self.cv_image, (64, 48))
        r, g, b = cv2.split(img)
        imgobj = np.asanyarray([r, g, b])

        img_left = cv2.resize(self.cv_left_image, (64, 48))
        r, g, b = cv2.split(img_left)
        imgobj_left = np.asanyarray([r, g, b])

        img_right = cv2.resize(self.cv_right_image, (64, 48))
        r, g, b = cv2.split(img_right)
        imgobj_right = np.asanyarray([r, g, b])

        ros_time = str(rospy.Time.now())

        line = ["offline_mode", str(self.counter), str(
            self.target_action), str(self.dir_cmd_data)]
        with open(self.cmd_path + '.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(line)

        cv2.imwrite(self.image_path + "center" +
                    str(self.counter) + ".png", img)
        cv2.imwrite(self.image_path + "left" +
                    str(self.counter) + ".png", img_left)
        cv2.imwrite(self.image_path + "right" + str(self.counter) +
                    ".png", img_right)

        print(self.counter, type(self.dir_cmd_data))

        self.counter += 1

        if self.counter == 7190:
            os.system('killall roslaunch')
            sys.exit()


if __name__ == '__main__':
    rg = offline()
    DURATION = 0.25
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()

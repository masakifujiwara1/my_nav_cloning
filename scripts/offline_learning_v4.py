#!/usr/bin/env python
from __future__ import print_function
from calendar import c
import copy
import time
import os
import csv
import random
import math

# from sqlalchemy import false

# from pandas import read_clipboard, read_csv
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
from nav_cloning_net_offline import *
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import cv2
import rospy
import roslib
import sys
import ast
roslib.load_manifest('nav_cloning')

# control param
EPISODE = 30000  # more 7190
HZ = 8


class offline():
    def __init__(self):
        rospy.init_node('offline_node', anonymous=True)
        self.action_num = rospy.get_param(
            "/LiDAR_based_learning_node/action_num", 1)
        print("action_num: " + str(self.action_num))
        self.dl = deep_learning(n_action=self.action_num)
        self.image_sub = rospy.Subscriber(
            "/camera_center/image_raw", Image, self.callback)
        self.image_left_sub = rospy.Subscriber(
            "/camera_left/image_raw", Image, self.callback_left_camera)
        self.image_right_sub = rospy.Subscriber(
            "/camera_right/image_raw", Image, self.callback_right_camera)
        # self.cmd = rospy.Subscriber(
        #     "/icart_mini/cmd_vel", Twist, self.callback_cmd)
        self.cmd = rospy.Subscriber(
            "/nav_vel", Twist, self.callback_cmd)
        self.dir_cmd_sub = rospy.Subscriber(
            "/cmd_dir", Int8MultiArray, self.callback_cmd_data)
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.target_action = 0.0
        self.counter = 0
        self.bridge = CvBridge()
        self.path = roslib.packages.get_pkg_dir(
            'nav_cloning') + '/data/dataset/'
        # self.dir_cmd_data = Int8MultiArray()
        self.dir_cmd_data = tuple([100, 0, 0, 0])
        # print(type(self.dir_cmd_data))
        self.action_list = []
        self.cv_image = np.zeros((480, 640, 3), np.uint8)
        self.cv_left_image = np.zeros((480, 640, 3), np.uint8)
        self.cv_right_image = np.zeros((480, 640, 3), np.uint8)
        self.action = 0
        self.date_path = "4hz_v3/"
        self.episode = 0
        self.save_path = roslib.packages.get_pkg_dir(
            'nav_cloning') + '/data/dataset/' + self.date_path + 'model/'
        self.episode = 0
        self.rosbag = True

        # calc_episode
        # self.calc_count = int(math.ceil(EPISODE / 7190))
        # print(self.calc_count)

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
        if self.counter == 14845:
            self.rosbag = False

        if self.cv_image.size != 640 * 480 * 3:
            self.rosbag = False
            return
        if self.cv_left_image.size != 640 * 480 * 3:
            return
        if self.cv_right_image.size != 640 * 480 * 3:
            return

        img = resize(self.cv_image, (48, 64), mode='constant')
        r, g, b = cv2.split(img)
        imgobj = np.asanyarray([r, g, b])

        img_left = resize(self.cv_left_image, (48, 64), mode='constant')
        r, g, b = cv2.split(img_left)
        imgobj_left = np.asanyarray([r, g, b])

        img_right = resize(self.cv_right_image, (48, 64), mode='constant')
        r, g, b = cv2.split(img_right)
        imgobj_right = np.asanyarray([r, g, b])

        self.action = self.target_action

        dir_cmd = np.asanyarray(self.dir_cmd_data)

        ros_time = str(rospy.Time.now())

        # learning
        self.dl.make_dataset(imgobj, dir_cmd, self.action)
        self.dl.make_dataset(imgobj_left, dir_cmd, self.action - 0.2)
        self.dl.make_dataset(imgobj_right, dir_cmd, self.action + 0.2)

        # print(self.counter)
        print("count:" + str(self.counter),
              "action:" + str(self.action), "cmd_data:" + str(self.dir_cmd_data))
        self.counter += 1
        # self.episode += 1

        cv2.imshow("img", self.cv_image)
        cv2.waitKey(1)

    def training(self):
        if self.episode == EPISODE:
            self.dl.save(self.save_path)
            sys.exit()

        self.dl.trains()
        print("episode:" + str(self.episode))
        self.episode += 1

    def destroy(self):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    rg = offline()
    # DURATION = 0.25
    DURATION = float(1) / HZ
    print(DURATION)
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
        if rg.rosbag == False:
            rg.destroy()
            break

    while not rospy.is_shutdown():
        rg.training()
        r.sleep()

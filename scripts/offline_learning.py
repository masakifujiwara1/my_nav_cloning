#!/usr/bin/env python
from __future__ import print_function
from calendar import c
import copy
import time
import os
import csv
import random

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
from nav_cloning_net_4com import *
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import cv2
import rospy
import roslib
import sys
import ast
roslib.load_manifest('nav_cloning')

EPISODE = 60000


class offline():
    def __init__(self):
        rospy.init_node('offline_node', anonymous=True)
        self.action_num = rospy.get_param(
            "/LiDAR_based_learning_node/action_num", 1)
        print("action_num: " + str(self.action_num))
        self.dl = deep_learning(n_action=self.action_num)
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
        self.action = 0
        self.date_path = "4hz/"
        self.episode = 0
        self.save_path = roslib.packages.get_pkg_dir(
            'nav_cloning') + '/data/dataset/' + self.date_path + 'model/'
        

    def read_img(self):
        self.cv_image = cv2.imread(
            self.path + self.date_path + "image/center" + str(self.counter) + ".png")
        self.cv_left_image = cv2.imread(
            self.path + self.date_path + "image/left" + str(self.counter) + ".png")
        self.cv_right_image = cv2.imread(
            self.path + self.date_path + "image/right" + str(self.counter) + ".png")

    def read_csv(self):
        f = open(self.path + self.date_path + 'target_action.csv', 'r')
        for row in f:
            self.action_list.append(row)
        cur_action = self.action_list[self.counter]
        cur_actioned = cur_action.split(",")
        target = '"'
        idx = cur_action.find(target)
        cur_action1 = cur_action[idx+len(target):]
        cur_action1 = cur_action1.replace('"', "")
        cur_action1 = cur_action1.replace('\n', "")
        # print((str(cur_action1)))
        # print(cur_action)
       
        if cur_action1 == '(100, 0, 0, 0)':
            # print('success')
            self.dir_cmd_data = tuple([100, 0, 0, 0])
        elif cur_action1 == '(0, 100, 0, 0)':
            self.dir_cmd_data = tuple([0, 100, 0, 0])
        elif cur_action1 == '(0, 0, 100, 0)':
            self.dir_cmd_data = tuple([0, 0, 100, 0])
        elif cur_action1 == '(0, 0, 0, 100)':
            self.dir_cmd_data = tuple([0, 0, 0, 100])

        # print(cur_action1)
        # print(type(cur_action))

        self.action = float(cur_actioned[2])
        # self.dir_cmd_data = cur_action
        # print(cur_action)
        # print(self.action, self.dir_cmd_data)

    def loop(self):
        if self.counter == 7190: #7190
            self.dl.save(self.save_path)
            # os.system('killall rosrun')
            sys.exit()
        
        self.read_img()
        self.read_csv()

        r, g, b = cv2.split(self.cv_image)
        imgobj = np.asanyarray([r, g, b])

        r, g, b = cv2.split(self.cv_left_image)
        imgobj_left = np.asanyarray([r, g, b])

        r, g, b = cv2.split(self.cv_right_image)
        imgobj_right = np.asanyarray([r, g, b])

        ros_time = str(rospy.Time.now())

        

        # learning
        target_action = self.action
        dir_cmd = np.asanyarray(self.dir_cmd_data)
        # print(dir_cmd, type(dir_cmd))
        action, loss = self.dl.act_and_trains(
            imgobj, dir_cmd, target_action)
        if abs(target_action) < 0.1:
            action_left,  loss_left = self.dl.act_and_trains(
                imgobj_left, dir_cmd, target_action - 0.2)
            action_right, loss_right = self.dl.act_and_trains(
                imgobj_right, dir_cmd, target_action + 0.2)

        # print(self.counter)
        print("count:" + str(self.counter), "loss:" + str(loss) , "action:" + str(self.action), "cmd_data:" + str(self.dir_cmd_data))
        self.counter += 1


if __name__ == '__main__':
    rg = offline()
    DURATION = 0.25
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
